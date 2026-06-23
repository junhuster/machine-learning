"""
quantize_onnx.py — ONNX → onnxruntime INT8 静态量化模块

核心 API：
  - quantize(onnx_path, output_path, nodes_to_exclude, val_dataset, calibration_samples)
      将 ONNX 模型量化为 INT8，跳过 nodes_to_exclude 中的节点。

  - load_session(onnx_path)
      加载 onnxruntime InferenceSession。
"""

import logging
import os
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Calibration Data Reader
# ------------------------------------------------------------------ #

class RecCalibrationReader(CalibrationDataReader):
    """
    为 onnxruntime 静态量化提供 calibration 数据。
    迭代产出 {input_name: np.ndarray} 字典。
    """

    def __init__(self, val_dataset, input_names, batch_size, num_samples):
        """
        Args:
            val_dataset:  RecDataset 实例
            input_names:  ONNX 模型的输入节点名列表
            batch_size:   calibration batch size
            num_samples:  最多使用的样本数
        """
        self.val_dataset = val_dataset
        self.input_names = input_names
        self.batch_size  = batch_size
        self.num_samples = num_samples
        self._data_iter  = None
        self._collected  = 0
        self._reset()

    def _reset(self):
        self._data_iter = self.val_dataset.as_numpy_batches(self.batch_size)
        self._collected = 0

    def get_next(self):
        if self._collected >= self.num_samples:
            return None
        try:
            batch = next(self._data_iter)
        except StopIteration:
            return None

        self._collected += batch["sparse"].shape[0]
        # onnxruntime 期望输入名和 ONNX graph input 对应
        # tf2onnx 导出的输入名通常是 "serving_default_sparse:0" 等形式
        feeds = {}
        for name in self.input_names:
            if "sparse" in name.lower():
                feeds[name] = batch["sparse"]
            elif "dense" in name.lower():
                feeds[name] = batch["dense"]
        return feeds


# ------------------------------------------------------------------ #
#  量化函数
# ------------------------------------------------------------------ #

def quantize(onnx_path, output_path, nodes_to_exclude,
             val_dataset, input_names, calibration_samples, batch_size=512):
    """
    对 ONNX 模型做静态 INT8 量化。

    Args:
        onnx_path:           FP32 ONNX 模型路径
        output_path:         INT8 ONNX 输出路径
        nodes_to_exclude:    list[str]，不参与量化的节点名（黑名单）
        val_dataset:         RecDataset，提供 calibration 数据
        input_names:         ONNX 模型输入名列表
        calibration_samples: calibration 样本数
        batch_size:          calibration batch size

    Returns:
        output_path (str)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    reader = RecCalibrationReader(
        val_dataset=val_dataset,
        input_names=input_names,
        batch_size=batch_size,
        num_samples=calibration_samples,
    )

    logger.debug(
        "quantize_static: exclude %d nodes: %s",
        len(nodes_to_exclude), nodes_to_exclude,
    )

    quantize_static(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        per_channel=False,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude if nodes_to_exclude else None,
    )

    logger.debug("INT8 ONNX saved to: %s", output_path)
    return output_path


def load_session(onnx_path):
    """
    加载 onnxruntime InferenceSession（CPU EP）。

    Returns:
        ort.InferenceSession
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info("ORT session loaded: %s", onnx_path)
    return session


def infer(session, sparse, dense):
    """
    用 onnxruntime session 推理一个 batch。

    Args:
        session: ort.InferenceSession
        sparse:  np.ndarray (B, F) int32
        dense:   np.ndarray (B, D) float32

    Returns:
        list[np.ndarray]: [pred_ctr, pred_cvr]
    """
    input_names = [inp.name for inp in session.get_inputs()]
    feeds = {}
    for name in input_names:
        if "sparse" in name.lower():
            feeds[name] = sparse
        elif "dense" in name.lower():
            feeds[name] = dense
    return session.run(None, feeds)
