#!/usr/bin/env python3
"""PyTorch 官方工具链做 PTQ（任务2，训练后量化）。

## 用 torch.ao 的哪一部分

量化参数的求取用的是 **`torch.ao.quantization.observer.MinMaxObserver`** ——
也就是 `torch.ao.quantization.quantize_fx` 在 FX graph mode PTQ 内部用的**同一套观察器与
`calculate_qparams()` 数学**（激活 per-tensor 非对称 uint8 / 权重 per-tensor 对称 int8）。

把参数写进 ONNX 用标准 Q/DQ 表达：`torch.quantize_per_tensor` + `torch.dequantize`
（导出时映射为 ONNX 的 `QuantizeLinear` / `DequantizeLinear`，支持非零 zero_point）。

## 为什么不直接导出 torch.ao 的 FX 量化模型

在 torch 2.x 上，FX 量化图直接 `torch.onnx.export` 有几处硬限制（本项目实测，见 README 第 6 节）：

  1. 量化张量的 `Concat`：`ONNX symbolic expected the output of onnx::Concat to be a quantized tensor`
  2. 权重 per-channel：`aten::quantize_per_channel` 没有 ONNX symbolic，直接 `UnsupportedOperatorError`
  3. `convert_fx(..., is_reference=True)` 得到的 reference 模型**无法直接导出**
     （`RuntimeError: ... Got int`；且先 `torch.jit.trace` 再导出会得到数值错误的图）

因此本项目分层处理：**torch.ao 负责"算量化参数"，标准 Q/DQ 负责"写进 ONNX"**。
这样既用到了 torch 官方工具，又能保证导出的图数值正确（用 ONNX Runtime 逐位核对过）。

## 一个必须知道的坑：激活别用 uint8 非对称

torch 2.2 导出 `quint8`（非对称，zero_point≠0）的激活 Q/DQ 时，会在
`QuantizeLinear`（输出 uint8）后插一个 **Cast**，且当 `zero_point=0` 时数值会出错——
实测误差从 5e-08 飙到 0.26（6 层累积）。

所以本项目**激活与权重统一用对称 int8**（`qint8` + `per_tensor_symmetric`，zero_point=0），
此时图里只有 `INT8` 一种 Cast，实测误差 ~1e-02 量级（≈ 一个量化步长，属于
torch 的 `quantize_per_tensor` 与 ONNX Runtime 的 `QuantizeLinear` 舍入模式差异，
可接受）。**权威精度以 bench_compare.py 里 TRT engine 实测的 AUC 为准。**
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.ao.quantization.observer import MinMaxObserver

_DTYPE_MAP = {"quint8": torch.quint8, "qint8": torch.qint8}
_QSCHEME_MAP = {
    "per_tensor_affine": torch.per_tensor_affine,
    "per_tensor_symmetric": torch.per_tensor_symmetric,
    "per_channel_affine": torch.per_channel_affine,
    "per_channel_symmetric": torch.per_channel_symmetric,
}


def resolve_dtypes(act_dtype: str, act_qscheme: str, w_dtype: str, w_qscheme: str):
    """把配置里的字符串解析成 torch 的 dtype / qscheme。"""
    try:
        return (_DTYPE_MAP[act_dtype], _QSCHEME_MAP[act_qscheme],
                _DTYPE_MAP[w_dtype], _QSCHEME_MAP[w_qscheme])
    except KeyError as exc:
        raise KeyError(
            f"未知取值 {exc}；dtype 可选 {list(_DTYPE_MAP)}，qscheme 可选 {list(_QSCHEME_MAP)}"
        ) from exc


@dataclass
class QParams:
    act_scale: float
    act_zero_point: int
    act_dtype: torch.dtype
    w_scale: float
    w_zero_point: int
    w_dtype: torch.dtype

    def __str__(self) -> str:
        return (f"act(scale={self.act_scale:.5f}, zp={self.act_zero_point}, {self.act_dtype}) "
                f"w(scale={self.w_scale:.5f}, zp={self.w_zero_point}, {self.w_dtype})")


# ---------------------------------------------------------------- Q/DQ 算子
def qdq(x: torch.Tensor, scale: float, zero_point: int, dtype: torch.dtype) -> torch.Tensor:
    """量化-反量化。导出 ONNX 时会变成 QuantizeLinear + DequantizeLinear。"""
    return torch.dequantize(torch.quantize_per_tensor(x.contiguous(), scale, zero_point, dtype))


class QDQLinear(nn.Module):
    """把 nn.Linear 包成可开关的量化层（qparams 由 torch.ao 的 observer 算出）。"""

    def __init__(self, linear: nn.Linear, name: str, qp: QParams):
        super().__init__()
        self.name = name
        self.linear = linear
        self.qp = qp
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.linear(x)
        w = qdq(self.linear.weight, self.qp.w_scale, self.qp.w_zero_point, self.qp.w_dtype)
        x = qdq(x, self.qp.act_scale, self.qp.act_zero_point, self.qp.act_dtype)
        return F.linear(x, w, self.linear.bias)

    def extra_repr(self) -> str:
        return f"name={self.name}, enabled={self.enabled}, {self.qp}"


# ---------------------------------------------------------------- 求量化参数
class _ActHook:
    """把某个 Linear 的输入喂给它的激活观察器。"""

    def __init__(self, obs: MinMaxObserver):
        self.obs = obs

    def __call__(self, module, inputs, output) -> None:  # noqa: ARG002
        self.obs(inputs[0].detach())


def list_linear_names(model: nn.Module) -> List[str]:
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n]


def compute_qparams(
    model: nn.Module,
    calib_iter: Iterable[Dict[str, torch.Tensor]],
    input_names: Sequence[str],
    act_dtype: torch.dtype = torch.qint8,
    act_qscheme=torch.per_tensor_symmetric,
    w_dtype: torch.dtype = torch.qint8,
    w_qscheme=torch.per_tensor_symmetric,
    reduce_range: bool = False,
    only: Optional[Set[str]] = None,
) -> Dict[str, QParams]:
    """用 torch.ao 的 MinMaxObserver 在校准集上求每个 Linear 的量化参数。

    返回 {层名: QParams}。不在 `only` 里的层不参与。
    """
    targets = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and n}
    if only is not None:
        targets = {n: m for n, m in targets.items() if n in only}
    if not targets:
        raise ValueError("没有可量化的 Linear 层")

    act_obs = {
        n: MinMaxObserver(dtype=act_dtype, qscheme=act_qscheme, reduce_range=reduce_range)
        for n in targets
    }
    handles = [m.register_forward_hook(_ActHook(act_obs[n])) for n, m in targets.items()]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for feed in calib_iter:
                model(**feed)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    out: Dict[str, QParams] = {}
    for n, m in targets.items():
        a_s, a_zp = act_obs[n].calculate_qparams()
        w_obs = MinMaxObserver(dtype=w_dtype, qscheme=w_qscheme, reduce_range=reduce_range)
        w_obs(m.weight.detach())
        w_s, w_zp = w_obs.calculate_qparams()
        out[n] = QParams(
            act_scale=float(a_s), act_zero_point=int(a_zp), act_dtype=act_dtype,
            w_scale=float(w_s), w_zero_point=int(w_zp), w_dtype=w_dtype,
        )
    return out


# ---------------------------------------------------------------- 包装与开关
def wrap_model(model: nn.Module, qparams: Dict[str, QParams]) -> List[QDQLinear]:
    """把所有 Linear 换成 QDQLinear（有 qparams 的才带量化能力，其余 enabled 恒为 False）。

    必须在加载完 FP32 权重之后调用（包装会改 state_dict 键名）。
    """
    wrappers: List[QDQLinear] = []

    def _recurse(parent: nn.Module, prefix: str) -> None:
        for cname, child in list(parent.named_children()):
            full = f"{prefix}{cname}"
            if isinstance(child, nn.Linear):
                qp = qparams.get(full)
                w = QDQLinear(child, full, qp) if qp is not None else QDQLinear(
                    child, full, QParams(1.0, 0, torch.quint8, 1.0, 0, torch.qint8)
                )
                w.has_qparams = qp is not None  # type: ignore[attr-defined]
                setattr(parent, cname, w)
                wrappers.append(w)
            else:
                _recurse(child, full + ".")

    _recurse(model, "")
    return wrappers


def set_enabled(
    wrappers: Sequence[QDQLinear],
    enabled: Optional[Set[str]] = None,
    disabled: Optional[Set[str]] = None,
) -> None:
    """控制哪些层真正做量化。

    enabled 为 None  -> 除 disabled 外全部开启
    enabled 非 None  -> 只开启 enabled 里且在 disabled 外的层
    """
    disabled = disabled or set()
    for w in wrappers:
        if not getattr(w, "has_qparams", False) or w.name in disabled:
            w.enabled = False
        elif enabled is None:
            w.enabled = True
        else:
            w.enabled = w.name in enabled


# ---------------------------------------------------------------- ONNX 导出
def export_qdq_onnx(
    model: nn.Module,
    dummy_args: Sequence[torch.Tensor],
    path: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset: int = 17,
) -> None:
    """导出带 Q/DQ 的 ONNX，并校验节点确实存在。"""
    dynamic_axes = {n: {0: "batch"} for n in list(input_names) + list(output_names)}
    torch.onnx.export(
        model, tuple(dummy_args), str(path),
        input_names=list(input_names), output_names=list(output_names),
        dynamic_axes=dynamic_axes, opset_version=opset,
        do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL,
    )

    import onnx

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    n_q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
    n_dq = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear")
    if n_q == 0:
        raise RuntimeError("导出的 ONNX 里没有 QuantizeLinear 节点，检查是否有层被 enabled。")
    print(f"[ao] ONNX 校验通过: QuantizeLinear={n_q}, DequantizeLinear={n_dq}")
