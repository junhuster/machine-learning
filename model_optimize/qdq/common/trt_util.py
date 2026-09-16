#!/usr/bin/env python3
"""TensorRT 构建与推理封装（ONNX → engine → 推理）。

设计要点：
  * 只用 tensorrt 官方 Python API，不依赖 pycuda（输入输出直接复用 torch 显存指针）。
  * 同时兼容 TRT 8.5+ 的 v3 API（set_tensor_address/execute_async_v3）与旧 binding API。
  * 支持动态 batch：构建时给 OptimizationProfile（min/opt/max），推理时 set_input_shape。

典型用法：
    from common.trt_util import build_engine, TRTEngine
    build_engine("m.onnx", "m_int8.plan", shapes={"user_id": ()}, min_batch=1,
                 opt_batch=32, max_batch=256, precision="int8")
    eng = TRTEngine("m_int8.plan")
    out = eng.infer({"user_id": idx_tensor, ...})
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

try:
    import tensorrt as trt
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "导入 tensorrt 失败，请先安装 TensorRT 的 Python 包（pip install tensorrt 或使用 "
        "NVIDIA 官方 tar 包里的 python 绑定）。原始错误: " + repr(exc)
    ) from exc


def _trt_dtype_to_torch(dtype) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    for attr in ("UINT8", "UINT32"):
        if hasattr(trt.DataType, attr):
            mapping[getattr(trt.DataType, attr)] = getattr(torch, attr.lower())
    if dtype not in mapping:
        raise TypeError(f"不支持的 TensorRT dtype: {dtype}")
    return mapping[dtype]


def _nbytes(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


def build_engine(
    onnx_path: str,
    engine_path: str,
    shapes: Optional[Dict[str, Sequence[int]]] = None,
    min_batch: int = 1,
    opt_batch: int = 32,
    max_batch: int = 256,
    precision: str = "fp32",
    workspace_gb: int = 4,
    strongly_typed: bool = False,
    verbose: bool = False,
) -> None:
    """把 ONNX 编译成 TensorRT engine。

    Args:
        shapes: 每个输入的「非 batch」维度，例如 {"user_id": (), "dense": (64,)}。
                batch 维由 min/opt/max_batch 自动补在最前面。
        precision: "fp32" | "fp16" | "int8"。
                   传 "int8" 且 ONNX 里已含 Q/DQ 时走「显式量化」，无需校准缓存。
    """
    precision = precision.lower()
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError(f"precision 必须是 fp32/fp16/int8，收到 {precision!r}")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        raw = f.read()
    if not parser.parse(raw):
        for i in range(parser.num_errors):
            print(f"  [ONNX parse error] {parser.get_error(i)}")
        raise RuntimeError(f"解析 ONNX 失败: {onnx_path}")

    config = builder.create_builder_config()

    # workspace（TRT 8.4+ 用 set_memory_pool_limit；旧版用 max_workspace_size）
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    except Exception:  # noqa: BLE001
        config.max_workspace_size = workspace_gb << 30

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if strongly_typed and hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # 动态 batch：给每个输入设置 profile
    if shapes is not None:
        profile = builder.create_optimization_profile()
        for name, dims in shapes.items():
            d = [int(x) for x in dims]
            if name not in [network.get_input(i).name for i in range(network.num_inputs)]:
                raise KeyError(f"ONNX 输入里没有 {name!r}，可用: "
                               f"{[network.get_input(i).name for i in range(network.num_inputs)]}")
            profile.set_shape(
                name,
                [min_batch, *d],
                [opt_batch, *d],
                [max_batch, *d],
            )
        config.add_optimization_profile(profile)

    print(f"[build] {os.path.basename(onnx_path)} -> {os.path.basename(engine_path)} "
          f"({precision}, batch {min_batch}/{opt_batch}/{max_batch}) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network 返回 None，构建失败（看上面日志）")

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(engine_path) / 1024 ** 2
    print(f"[build] 完成: {engine_path}  ({size_mb:.1f} MB)")


class TRTEngine:
    """TensorRT engine 推理封装（支持动态 batch）。"""

    def __init__(self, engine_path: str):
        self.engine_path = str(engine_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self.logger)
        with open(self.engine_path, "rb") as f:
            self.engine = self._runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"反序列化 engine 失败（版本不匹配？）: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self._v3 = hasattr(self.engine, "num_io_tensors")

        if self._v3:
            n = self.engine.num_io_tensors
            names = [self.engine.get_tensor_name(i) for i in range(n)]
            self.input_names = [
                x for x in names
                if self.engine.get_tensor_mode(x) == trt.TensorIOMode.INPUT
            ]
            self.output_names = [
                x for x in names
                if self.engine.get_tensor_mode(x) == trt.TensorIOMode.OUTPUT
            ]
        else:  # 旧 API 兜底
            n = self.engine.num_bindings
            self.input_names = [
                self.engine.get_binding_name(i)
                for i in range(n)
                if self.engine.binding_is_input(i)
            ]
            self.output_names = [
                self.engine.get_binding_name(i)
                for i in range(n)
                if not self.engine.binding_is_input(i)
            ]

    # ---------- 元信息 ----------
    def input_dtype(self, name: str) -> torch.dtype:
        if self._v3:
            return _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
        for i in range(self.engine.num_bindings):
            if self.engine.get_binding_name(i) == name:
                return _trt_dtype_to_torch(self.engine.get_binding_dtype(i))
        raise KeyError(name)

    def describe(self) -> str:
        lines = [f"engine: {self.engine_path}"]
        for name in self.input_names:
            lines.append(f"  IN   {name:20s} dtype={self.input_dtype(name)}")
        for name in self.output_names:
            lines.append(f"  OUT  {name:20s} dtype={self.input_dtype(name)}")
        return "\n".join(lines)

    # ---------- 推理 ----------
    def infer(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """inputs: {名字: torch.Tensor}（CPU 张量会自动搬到 GPU）。返回 {输出名: 张量}。"""
        ctx = self.context
        hold: Dict[str, torch.Tensor] = {}
        outputs: Dict[str, torch.Tensor] = {}

        if self._v3:
            for name in self.input_names:
                if name not in inputs:
                    raise KeyError(f"缺少输入 {name!r}，已提供: {list(inputs)}")
                t = inputs[name]
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"输入 {name!r} 必须是 torch.Tensor")
                t = t.cuda().contiguous()
                ctx.set_input_shape(name, tuple(t.shape))
                ctx.set_tensor_address(name, int(t.data_ptr()))
                hold[name] = t

            for name in self.output_names:
                shape = tuple(ctx.get_tensor_shape(name))
                out = torch.empty(shape, dtype=self.input_dtype(name), device="cuda")
                ctx.set_tensor_address(name, int(out.data_ptr()))
                outputs[name] = out

            ok = ctx.execute_async_v3(self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("execute_async_v3 返回 False（形状不合法？）")
        else:
            bindings = [0] * self.engine.num_bindings
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i):
                    t = inputs[name].cuda().contiguous()
                    ctx.set_binding_shape(i, tuple(t.shape))
                    bindings[i] = int(t.data_ptr())
                    hold[name] = t
                else:
                    shape = tuple(ctx.get_binding_shape(i))
                    out = torch.empty(shape, dtype=self.input_dtype(name), device="cuda")
                    bindings[i] = int(out.data_ptr())
                    outputs[name] = out
            ok = ctx.execute_async_v2(bindings, self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("execute_async_v2 返回 False（形状不合法？）")

        self.stream.synchronize()
        return outputs

    def warmup(self, inputs: Dict[str, torch.Tensor], iters: int = 10) -> None:
        for _ in range(iters):
            self.infer(inputs)


def trt_version() -> str:
    return trt.__version__


if __name__ == "__main__":
    print("TensorRT version:", trt_version())
