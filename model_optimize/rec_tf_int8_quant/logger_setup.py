"""
logger_setup.py — 统一日志初始化
所有模块的日志都汇聚到同一个文件 + 控制台，不再各自分散。
"""

import logging
import os


def init_logger(log_cfg):
    """
    配置 root logger，同时输出到控制台和统一日志文件。
    各模块通过 logging.getLogger(__name__) 获取子 logger，
    自动继承 root logger 的 handler，无需单独配置。

    Args:
        log_cfg (dict): config.yaml 中的 logging 节点，字段：
            level    - 日志级别，默认 INFO
            format   - 日志格式
            log_file - 日志文件路径，默认 logs/run.log
    """
    level   = getattr(logging, log_cfg.get("level", "INFO"))
    fmt     = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    logfile = log_cfg.get("log_file", "logs/run.log")

    os.makedirs(os.path.dirname(logfile) if os.path.dirname(logfile) else ".", exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # 避免重复添加 handler（predict.py 单独调用时也安全）
    if root.handlers:
        root.handlers.clear()

    formatter = logging.Formatter(fmt)

    # 文件 handler（追加模式，同一次运行所有模块日志写同一文件）
    file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logger initialized: level=%s  file=%s", log_cfg.get("level", "INFO"), logfile
    )
