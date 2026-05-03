import logging as log
from logging.handlers import TimedRotatingFileHandler
_handler = None
_formatter = log.Formatter(
      fmt="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d %(funcName)s] - %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S"
  )
  
def init_logger(logfile: str, level=log.DEBUG, backupCount=5):
      """初始化日志，指定日志文件路径"""
      global _handler

      # 移除旧的 handler
      if _handler:
          log.getLogger().removeHandler(_handler)

      _handler = TimedRotatingFileHandler(
          filename=logfile,
          when="midnight",
          interval=1,
          backupCount=backupCount,
          encoding="utf-8"
      )
      _handler.suffix = "%Y-%m-%d"
      _handler.setFormatter(_formatter)

      log.basicConfig(
          level=level,
          handlers=[_handler]
      )