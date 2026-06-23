"""
onnx2trt.py — ONNX → TensorRT INT8 Engine 构建工具脚本

读取含 QDQ 节点的 ONNX 模型，TensorRT 直接从 QDQ 节点提取
量化参数（scale/zero_point），无需再做 calibration，构建 INT8 engine。

也支持从 FP32 ONNX 构建 FP16 engine（--precision fp16）。

用法：
  # QDQ ONNX → INT8 TRT engine（推荐）
  python onnx2trt.py --config config.yaml --precision int8

  # FP32 ONNX → FP16 TRT engine
  python onnx2trt.py --config config.yaml --precision fp16 \
      --onnx checkpoints/model_fp32.onnx

  # 构建后立即做 benchmark
  python onnx2trt.py --config config.yaml --precision int8 --benchmark
"""

import argparse
import logging
import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

import yaml
from logger_setup import init_logger

logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ------------------------------------------------------------------ #
#  Engine 构建
# ------------------------------------------------------------------ #

def build_engine(onnx_path, engine_path, precision, cfg):
    """
    从 ONNX 构建 TensorRT engine。

    Args:
        onnx_path:   输入 ONNX 路径（QDQ 格式或 FP32）
        engine_path: 输出 TRT engine 路径
        precision:   "int8" 或 "fp16"
        cfg:         全局配置

    Returns:
        engine_path (str)
    """
    trt_cfg   = cfg["tensorrt"]
    max_batch = trt_cfg["max_batch_size"]
    workspace = trt_cfg["workspace_mb"] * (1 << 20)

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    builder      = trt.Builder(TRT_LOGGER)
    network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network      = builder.create_network(network_flag)
    parser       = trt.OnnxParser(network, TRT_LOGGER)

    logger.info("Parsing ONNX: %s", onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("TRT parse error[%d]: %s", i, parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model.")

    build_config = builder.create_builder_config()
    build_config.max_workspace_size = workspace

    if precision == "int8":
        build_config.set_flag(trt.BuilderFlag.INT8)
        # QDQ ONNX 已含量化参数，直接使用 EXPLICIT_PRECISION 模式
        # TRT 从 QuantizeLinear / DequantizeLinear 节点读取 scale，无需 Calibrator
        build_config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        logger.info("Building INT8 engine from QDQ ONNX (no calibration needed)...")
    elif precision == "fp16":
        build_config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Building FP16 engine...")
    else:
        logger.info("Building FP32 engine...")

    # 设置动态 batch size profile
    profile = builder.create_optimization_profile()
    d = cfg["data"]
    nf = d["num_sparse_fields"]
    nd = d["num_dense_features"]
    profile.set_shape("sparse", (1, nf), (max_batch // 2, nf), (max_batch, nf))
    profile.set_shape("dense",  (1, nd), (max_batch // 2, nd), (max_batch, nd))
    build_config.add_optimization_profile(profile)

    logger.info("Building TRT engine (this may take a while)...")
    engine = builder.build_engine(network, build_config)
    if engine is None:
        raise RuntimeError("TRT engine build failed.")

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    logger.info("TRT engine saved to: %s", engine_path)
    return engine_path


def load_engine(engine_path):
    """从文件反序列化加载 TRT engine。"""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    logger.info("TRT engine loaded: %s", engine_path)
    return engine


# ------------------------------------------------------------------ #
#  TRT 推理封装
# ------------------------------------------------------------------ #

class TRTRunner:
    """封装 TRT engine 推理，提供统一的 infer(sparse, dense) 接口。"""

    def __init__(self, engine):
        self.engine  = engine
        self.context = engine.create_execution_context()
        self.stream  = cuda.Stream()
        self._alloc_buffers()

    def _alloc_buffers(self):
        self.bindings  = []
        self.host_ins  = {}
        self.dev_ins   = {}
        self.host_outs = {}
        self.dev_outs  = {}

        for i in range(self.engine.num_bindings):
            name  = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            # 动态 shape 时 shape[0] == -1，先用 max_shape 分配
            max_shape = self.engine.get_profile_shape(0, i)[2]
            size      = int(np.prod(max_shape))
            host_mem  = cuda.pagelocked_empty(size, dtype)
            dev_mem   = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))

            if self.engine.binding_is_input(i):
                self.host_ins[name]  = host_mem
                self.dev_ins[name]   = dev_mem
            else:
                self.host_outs[name] = host_mem
                self.dev_outs[name]  = dev_mem

    def infer(self, sparse, dense):
        """
        Args:
            sparse: np.ndarray (B, F) int32 / int64
            dense:  np.ndarray (B, D) float32

        Returns:
            dict[str, np.ndarray]: {"pred_ctr": ..., "pred_cvr": ...}
        """
        B = sparse.shape[0]

        # 设置动态 batch size
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                if "sparse" in name:
                    self.context.set_binding_shape(i, sparse.shape)
                elif "dense" in name:
                    self.context.set_binding_shape(i, dense.shape)

        # H2D
        sp_flat = sparse.flatten().astype(np.int32)
        de_flat = dense.flatten().astype(np.float32)
        np.copyto(self.host_ins["sparse"][:len(sp_flat)], sp_flat)
        np.copyto(self.host_ins["dense"][:len(de_flat)],  de_flat)
        for name in self.host_ins:
            cuda.memcpy_htod_async(self.dev_ins[name], self.host_ins[name], self.stream)

        self.context.execute_async_v2(self.bindings, self.stream.handle)

        # D2H
        for name in self.host_outs:
            cuda.memcpy_dtoh_async(self.host_outs[name], self.dev_outs[name], self.stream)
        self.stream.synchronize()

        return {name: self.host_outs[name][:B].copy() for name in self.host_outs}


# ------------------------------------------------------------------ #
#  Benchmark
# ------------------------------------------------------------------ #

def benchmark(runner, cfg, iters=100):
    """测量单 batch 平均推理延迟（ms）和吞吐量（QPS）。"""
    d         = cfg["data"]
    batch_size = d["batch_size"]
    sparse = np.zeros((batch_size, d["num_sparse_fields"]), dtype=np.int32)
    dense  = np.zeros((batch_size, d["num_dense_features"]), dtype=np.float32)

    # warmup
    for _ in range(5):
        runner.infer(sparse, dense)

    start = time.perf_counter()
    for _ in range(iters):
        runner.infer(sparse, dense)
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / iters * 1000
    qps    = batch_size * iters / elapsed
    logger.info("Benchmark: batch_size=%d  avg_latency=%.3f ms  QPS=%.1f",
                batch_size, avg_ms, qps)
    return avg_ms, qps


# ------------------------------------------------------------------ #
#  主入口
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="ONNX → TensorRT engine builder")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--onnx",      default=None,
                        help="输入 ONNX 路径，不指定则从 config 读取")
    parser.add_argument("--output",    default=None,
                        help="输出 TRT engine 路径，不指定则从 config 读取")
    parser.add_argument("--precision", choices=["int8", "fp16", "fp32"],
                        default="int8")
    parser.add_argument("--benchmark", action="store_true",
                        help="构建后立即做性能测试")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    init_logger(cfg.get("logging", {}))

    # 确定输入/输出路径
    if args.onnx:
        onnx_path = args.onnx
    elif args.precision == "int8":
        onnx_path = cfg["export"]["onnx_qdq_path"]
    else:
        onnx_path = cfg["export"]["onnx_fp32_path"]

    engine_path = args.output or cfg["tensorrt"]["engine_path"]

    logger.info("ONNX: %s  precision: %s  engine: %s",
                onnx_path, args.precision, engine_path)

    build_engine(onnx_path, engine_path, args.precision, cfg)

    do_benchmark = args.benchmark or cfg["tensorrt"].get("benchmark", False)
    if do_benchmark:
        engine = load_engine(engine_path)
        runner = TRTRunner(engine)
        iters  = cfg["tensorrt"].get("benchmark_iters", 100)
        benchmark(runner, cfg, iters=iters)

    logger.info("Done.")


if __name__ == "__main__":
    main()
