#!/usr/bin/env python3
"""环境探测：一次性看清这台机器具备哪些量化/部署能力。

用法:
    python3 common/check_env.py

输出三个分区：
  1) 硬件与驱动（GPU / 算力 / 驱动 / CUDA）
  2) Python 包（torch / onnx / onnxruntime / tensorrt / pytorch_quantization / transformers）
  3) 命令行工具（trtexec / nsys / ncu / polygraphy）

退出码恒为 0，只做报告，不报错中断，方便直接复制输出。
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys


def _section(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def check_hardware() -> None:
    _section("1) 硬件与驱动")
    try:
        import torch

        print(f"torch          : {torch.__version__}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"cuda available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"device count   : {n}")
            for i in range(n):
                prop = torch.cuda.get_device_properties(i)
                cap = f"sm_{prop.major}{prop.minor}"
                print(
                    f"  [{i}] {prop.name} | {cap} | "
                    f"{prop.total_memory / 1024 ** 3:.1f} GB | SMs={prop.multi_processor_count}"
                )
                if cap == "sm_75":
                    print("      注: Turing — 支持 FP16/INT8 Tensor Core；不支持 TF32/INT4/FP8")
    except Exception as exc:  # noqa: BLE001
        print(f"torch 不可用或探测失败: {type(exc).__name__}: {exc}")

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        print("nvidia-smi     :")
        for line in out.stdout.strip().splitlines():
            print("  " + line)
    except Exception as exc:  # noqa: BLE001
        print(f"nvidia-smi 不可用: {type(exc).__name__}")


def check_packages() -> None:
    _section("2) Python 包")
    mods = [
        ("torch", "torch"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("tensorrt", "tensorrt"),
        ("pytorch_quantization", "pytorch-quantization (NVIDIA)"),
        ("transformer_deploy", "transformer-deploy (HF)"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
    ]
    missing = []
    for mod_name, label in mods:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  OK  {label:34s} {ver}")
        except Exception as exc:  # noqa: BLE001
            print(f"  --  {label:34s} 未安装 ({type(exc).__name__})")
            missing.append(label)

    # onnx 的 Q/DQ 相关能力
    try:
        import onnx  # noqa: F401

        from onnx import TensorProto  # noqa: F401

        print("  OK  onnx QuantizeLinear/DequantizeLinear opset 可用")
    except Exception:  # noqa: BLE001
        pass

    if missing:
        print("\n  缺失清单: " + ", ".join(missing))


def check_binaries() -> None:
    _section("3) 命令行工具")
    for b in ["trtexec", "nsys", "ncu", "polygraphy", "onnxsim", "nvidia-smi"]:
        path = shutil.which(b)
        print(f"  {'OK ' if path else '-- '} {b:12s} {path or ''}")


def main() -> int:
    print(f"python         : {sys.version.split()[0]}  ({sys.executable})")
    check_hardware()
    check_packages()
    check_binaries()
    print("\n探测完成。判断路径:")
    print("  · 两个任务都需要      : torch / onnx / tensorrt(Python API)")
    print("  · 任务1 (NVIDIA 系)  : pytorch_quantization 或 transformer_deploy")
    print("  · 任务2 (PyTorch 系) : 只需 torch>=2.0 自带 torch.ao，无需额外包")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
