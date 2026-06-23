"""
quantize_trt.py — ONNX → TensorRT INT8 量化模块

核心 API：
  - build_engine(onnx_path, engine_path, layers_to_keep_fp32,
                 val_dataset, calibration_samples, cfg)
      将 ONNX 转换为 TRT INT8 engine，指定层保持 FP32。

  - TRTRunner
      封装 TRT engine 推理，提供 infer(sparse, dense) 接口。
"""

import logging
import os
import struct
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  自动初始化 CUDA context

logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ------------------------------------------------------------------ #
#  Calibrator
# ------------------------------------------------------------------ #

class RecInt8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    TensorRT INT8 Entropy Calibrator。
    从 RecDataset 读取 calibration 数据，送入 TRT 统计激活分布。
    """

    def __init__(self, val_dataset, input_shapes, batch_size, num_samples,
                 cache_file="checkpoints/trt_calib.cache"):
        super().__init__()
        self.val_dataset   = val_dataset
        self.input_shapes  = input_shapes   # {"sparse": (B,F), "dense": (B,D)}
        self.batch_size    = batch_size
        self.num_samples   = num_samples
        self.cache_file    = cache_file
        self._data_iter    = val_dataset.as_numpy_batches(batch_size)
        self._collected    = 0

        # 在 GPU 上预分配输入 buffer
        self._buffers = {}
        for name, shape in input_shapes.items():
            dtype = np.int32 if name == "sparse" else np.float32
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self._buffers[name] = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self._collected >= self.num_samples:
            return None
        try:
            batch = next(self._data_iter)
        except StopIteration:
            return None

        self._collected += batch["sparse"].shape[0]
        result = []
        for name in names:
            if "sparse" in name.lower():
                data = batch["sparse"].astype(np.int32)
            else:
                data = batch["dense"].astype(np.float32)
            cuda.memcpy_htod(self._buffers[name], np.ascontiguousarray(data))
            result.append(int(self._buffers[name]))
        return result

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            logger.debug("Reading TRT calibration cache: %s", self.cache_file)
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        logger.debug("TRT calibration cache saved: %s", self.cache_file)


# ------------------------------------------------------------------ #
#  Engine 构建
# ------------------------------------------------------------------ #

def build_engine(onnx_path, engine_path, layers_to_keep_fp32,
                 val_dataset, calibration_samples, cfg):
    """
    将 ONNX 模型构建为 TRT INT8 engine。

    Args:
        onnx_path:            FP32 ONNX 模型路径
        engine_path:          TRT engine 输出路径
        layers_to_keep_fp32:  list[str]，需要保持 FP32 的层名（黑名单）
        val_dataset:          RecDataset，提供 calibration 数据
        calibration_samples:  calibration 样本数
        cfg:                  全局配置 dict

    Returns:
        engine_path (str)
    """
    q_cfg = cfg["quantize"]
    d_cfg = cfg["data"]
    batch_size = q_cfg["trt_max_batch_size"]

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    logger.info("Parsing ONNX model: %s", onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("TRT ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model.")

    config = builder.create_builder_config()
    config.max_workspace_size = q_cfg["trt_workspace_mb"] * (1 << 20)
    config.set_flag(trt.BuilderFlag.INT8)

    # calibrator
    input_shapes = {
        "sparse": (batch_size, d_cfg["num_sparse_fields"]),
        "dense":  (batch_size, d_cfg["num_dense_features"]),
    }
    calibrator = RecInt8Calibrator(
        val_dataset=val_dataset,
        input_shapes=input_shapes,
        batch_size=min(512, batch_size),
        num_samples=calibration_samples,
        cache_file=os.path.join(os.path.dirname(engine_path), "trt_calib.cache"),
    )
    config.int8_calibrator = calibrator

    # 对黑名单层强制设置 FP32 动态范围，使其跳过 INT8
    if layers_to_keep_fp32:
        logger.info("Keeping %d layers in FP32: %s",
                    len(layers_to_keep_fp32), layers_to_keep_fp32)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name in layers_to_keep_fp32:
                layer.precision = trt.DataType.FLOAT
                layer.set_output_type(0, trt.DataType.FLOAT)

    logger.info("Building TRT INT8 engine (this may take a while)...")
    engine = builder.build_engine(network, config)
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
#  推理封装
# ------------------------------------------------------------------ #

class TRTRunner:
    """
    封装 TRT engine 推理，提供和 onnxruntime 一致的 infer() 接口。
    """

    def __init__(self, engine):
        self.engine  = engine
        self.context = engine.create_execution_context()
        self._allocate_buffers()

    def _allocate_buffers(self):
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size  = int(np.prod(shape))
            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem,
                                    "name": binding})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem,
                                     "name": binding})

    def infer(self, sparse, dense):
        """
        推理一个 batch。

        Args:
            sparse: np.ndarray (B, F) int32
            dense:  np.ndarray (B, D) float32

        Returns:
            list[np.ndarray]: [pred_ctr, pred_cvr]
        """
        for inp in self.inputs:
            if "sparse" in inp["name"].lower():
                np.copyto(inp["host"], sparse.flatten().astype(np.int32))
            else:
                np.copyto(inp["host"], dense.flatten().astype(np.float32))
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )

        results = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            results.append(out["host"].copy())
        self.stream.synchronize()
        return results
