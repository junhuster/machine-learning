#!/usr/bin/env python3
"""Q/DQ 量化引擎（任务1）。

提供两种后端，**都产出「带 QuantizeLinear / DequantizeLinear 节点」的 ONNX**，
交给 TensorRT 按显式量化（explicit quantization）解析：

  backend = "pytorch_quantization"  NVIDIA 官方 TensorRT Quantization Toolkit
                                    （HF transformer-deploy 的 Q/DQ 插入步骤底层就是它）
  backend = "torch_fake"            内置实现：自己算 scale，用 torch.quantize_per_tensor
                                    表达 Q/DQ（torch 的 ONNX 导出会映射成 Q/DQ 节点）
  backend = "auto"                  能 import 到 pytorch_quantization 就用前者，否则后者

为什么要内置一份：pytorch_quantization 对「自定义模型」的自动替换不总是生效，
内置实现保证「torch 模型 -> 带 Q/DQ 的 ONNX -> TRT INT8」这条链路一定跑得通。

模式（每个 Linear 层各自持有）：
  off    不量化（原始 FP32）
  calib  只统计 amax / 分位数，前向仍是 FP32（PTQ 标准做法）
  quant  伪量化：量化再反量化，数值上是 INT8 的效果，纯 torch 运算
  export 与 quant 等价，但改用 quantize_per_tensor，使 ONNX 导出成 Q/DQ 节点
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QINT8_MAX = 127.0


# ---------------------------------------------------------------- 基本算子
def fake_qdq_sym(x: torch.Tensor, scale: float) -> torch.Tensor:
    """对称 INT8 伪量化（纯 torch 运算，快，用于敏感度扫描）。"""
    q = torch.clamp(torch.round(x / scale), -QINT8_MAX - 1, QINT8_MAX)
    return q * scale


def onnx_qdq_sym(x: torch.Tensor, scale: float) -> torch.Tensor:
    """对称 INT8 的 quant-dequant，导出 ONNX 时会变成 QuantizeLinear/DequantizeLinear。"""
    q = torch.quantize_per_tensor(x.contiguous(), float(scale), 0, torch.qint8)
    return torch.dequantize(q)


# ---------------------------------------------------------------- 单层包装
class QDQLinear(nn.Module):
    """把一个 nn.Linear 包成「可控量化」的层。"""

    def __init__(
        self,
        linear: nn.Linear,
        name: str,
        method: str = "max",
        percentile: float = 99.99,
        quant_weight: bool = True,
        quant_act: bool = True,
        sample_budget: int = 4096,
    ):
        super().__init__()
        self.name = name
        self.linear = linear
        self.method = method
        self.percentile = percentile
        self.quant_weight = quant_weight
        self.quant_act = quant_act
        self.sample_budget = sample_budget
        self.mode = "off"

        self.register_buffer("amax_in", torch.zeros(()))
        self.register_buffer("amax_w", torch.zeros(()))
        self.register_buffer("scale_in", torch.ones(()))
        self.register_buffer("scale_w", torch.ones(()))
        self._act_samples: List[torch.Tensor] = []

    # -- 校准 --
    def _observe(self, x: torch.Tensor) -> None:
        a = float(x.detach().abs().max())
        self.amax_in = torch.maximum(
            self.amax_in, torch.tensor(a, device=self.amax_in.device, dtype=self.amax_in.dtype)
        )
        if self.method == "percentile":
            flat = x.detach().abs().reshape(-1)
            if flat.numel() > self.sample_budget:
                idx = torch.randint(0, flat.numel(), (self.sample_budget,), device=flat.device)
                flat = flat[idx]
            self._act_samples.append(flat.to("cpu", torch.float32))

    @staticmethod
    def _to_scale(amax: float) -> float:
        return max(float(amax), 1e-8) / QINT8_MAX

    def finalize(self) -> None:
        """校准结束后调用：定下激活/权重的 scale。"""
        if self.method == "percentile" and self._act_samples:
            vals = torch.cat(self._act_samples).numpy()
            amax = float(np.percentile(vals, self.percentile))
        else:
            amax = float(self.amax_in)
        self.scale_in = torch.tensor(self._to_scale(amax))
        self.scale_w = torch.tensor(self._to_scale(float(self.linear.weight.detach().abs().max())))
        self._act_samples = []

    # -- 前向 --
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return self.linear(x)
        if self.mode == "calib":
            self._observe(x)
            return self.linear(x)

        fn = onnx_qdq_sym if self.mode == "export" else fake_qdq_sym
        w = self.linear.weight
        if self.quant_weight:
            w = fn(w, float(self.scale_w))
        if self.quant_act:
            x = fn(x, float(self.scale_in))
        return F.linear(x, w, self.linear.bias)

    def extra_repr(self) -> str:
        return (f"name={self.name}, mode={self.mode}, "
                f"scale_in={float(self.scale_in):.3e}, scale_w={float(self.scale_w):.3e}")


# ---------------------------------------------------------------- 模型级操作
def list_linear_names(model: nn.Module) -> List[str]:
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def wrap_model(
    model: nn.Module,
    method: str = "max",
    percentile: float = 99.99,
    quant_weight: bool = True,
    quant_act: bool = True,
) -> List[QDQLinear]:
    """把所有 nn.Linear 就地替换为 QDQLinear，返回包装层列表（顺序 = 网络顺序）。

    必须在「加载完 FP32 权重之后」调用——包装会改变 state_dict 的键名。
    """
    wrappers: List[QDQLinear] = []

    def _recurse(parent: nn.Module, prefix: str) -> None:
        for cname, child in list(parent.named_children()):
            full = f"{prefix}{cname}"
            if isinstance(child, nn.Linear):
                w = QDQLinear(child, full, method, percentile, quant_weight, quant_act)
                setattr(parent, cname, w)
                wrappers.append(w)
            else:
                _recurse(child, full + ".")

    _recurse(model, "")
    return wrappers


def set_modes(
    wrappers: Sequence[QDQLinear],
    mode: str,
    only: Optional[Set[str]] = None,
    disabled: Optional[Set[str]] = None,
) -> None:
    """批量设置模式。

    only     非空时，只有名字在其中的层被设为 mode，其余 off
    disabled 名字在其中的层强制 off（敏感层名单）
    """
    disabled = disabled or set()
    for w in wrappers:
        if w.name in disabled or (only is not None and w.name not in only):
            w.mode = "off"
        else:
            w.mode = mode


def calibrate(
    model: nn.Module,
    wrappers: Sequence[QDQLinear],
    calib_iter,
    disabled: Optional[Set[str]] = None,
) -> None:
    """跑校准集统计 amax。

    calib_iter 产出的是「模型输入字典」{输入名: 张量}，例如：
        (b.to(device).inputs(INPUT_NAMES) for b in iter_batches(calib_set, bs))
    """
    set_modes(wrappers, "calib", disabled=disabled)
    model.eval()
    with torch.no_grad():
        for feed in calib_iter:
            model(**feed)
    for w in wrappers:
        w.finalize()
    set_modes(wrappers, "off")


# ---------------------------------------------------------------- 与 ONNX 对接
def export_qdq_onnx(
    model: nn.Module,
    dummy_args,
    path: str,
    input_names,
    output_names,
    opset: int = 17,
    simplify: bool = False,
) -> None:
    """把处于 export 模式的模型导出成带 Q/DQ 的 ONNX，并校验节点确实存在。

    注：torch 导出时会在 QuantizeLinear 与 DequantizeLinear 之间插入一个
    Cast（int8→int8，等价于无操作）。ONNX Runtime 与 TensorRT 都能正确处理；
    若某个 TRT 版本对此报错，把 simplify=True（需要 onnxsim）即可抹平。
    """
    dynamic_axes = {n: {0: "batch"} for n in list(input_names) + list(output_names)}
    torch.onnx.export(
        model,
        dummy_args,
        str(path),
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )

    import onnx

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    n_q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
    n_dq = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear")
    if n_q == 0:
        raise RuntimeError(
            "导出的 ONNX 里没有 QuantizeLinear 节点 —— 说明 torch 没有把 quantize_per_tensor\n"
            "映射成 Q/DQ。请改用 backend=pytorch_quantization（NVIDIA 官方路径），\n"
            "或检查 torch 版本（建议 >= 2.0，opset >= 13）。"
        )
    print(f"[qdq] ONNX 校验通过: QuantizeLinear={n_q}, DequantizeLinear={n_dq}")

    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify

            ms, ok = onnx_simplify(m)
            if ok:
                onnx.save(ms, str(path))
                n_q2 = sum(1 for n in ms.graph.node if n.op_type == "QuantizeLinear")
                print(f"[qdq] onnxsim 简化完成（Q 节点数 {n_q} -> {n_q2}）")
            else:
                print("[qdq] onnxsim 认为简化后不一致，已放弃简化")
        except ImportError:
            print("[qdq] 未安装 onnxsim，跳过简化（pip install onnxsim 可启用）")


# ---------------------------------------------------------------- 可选：NVIDIA 后端
def has_pytorch_quantization() -> bool:
    try:
        import pytorch_quantization  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def wrap_model_pytorch_quantization(model: nn.Module, calib_method: str = "max") -> List[nn.Module]:
    """用 NVIDIA pytorch_quantization 替换 nn.Linear。

    返回替换后的模块列表（每个含 _input_quantizer/_weight_quantizer，
    可用 set_pq_modes() 控制开关）。
    """
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.tensor_quant import QuantDescriptor

    qdesc = QuantDescriptor(calib_method="histogram" if calib_method == "histogram" else "max")
    replaced: List[nn.Module] = []

    def _recurse(parent: nn.Module, prefix: str) -> None:
        for cname, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                q = quant_nn.QuantLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, quant_desc_input=qdesc,
                )
                q.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    q.bias.data.copy_(child.bias.data)
                setattr(parent, cname, q)
                q._qdq_name = f"{prefix}{cname}"  # type: ignore[attr-defined]
                replaced.append(q)
            else:
                _recurse(child, f"{prefix}{cname}.")
    _recurse(model, "")
    return replaced


def set_pq_modes(modules: Sequence[nn.Module], enabled: bool, disabled: Optional[Set[str]] = None) -> None:
    """开关 pytorch_quantization 的量化器。"""
    disabled = disabled or set()
    for m in modules:
        name = getattr(m, "_qdq_name", "")
        on = enabled and name not in disabled
        for attr in ("_input_quantizer", "_weight_quantizer"):
            q = getattr(m, attr, None)
            if q is None:
                continue
            q.enable() if on else q.disable()
