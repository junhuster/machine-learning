"""
logger_setup.py — 统一日志初始化，所有模块日志写入同一文件
"""
import logging
import os


def init_logger(log_cfg):
    level   = getattr(logging, log_cfg.get("level", "INFO"))
    fmt     = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    logfile = log_cfg.get("log_file", "logs/run.log")

    os.makedirs(os.path.dirname(logfile) if os.path.dirname(logfile) else ".", exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        root.handlers.clear()

    formatter    = logging.Formatter(fmt)
    file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logger initialized: level=%s  file=%s", log_cfg.get("level", "INFO"), logfile
    )
