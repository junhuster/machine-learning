import logging as log
from logging.handlers import TimedRotatingFileHandler

logfile="/home/ubuntu/work/logs/gpt2-train.log.merge.ft1"
handler = TimedRotatingFileHandler(
    filename=logfile,
    when="midnight",      # 每天午夜轮转
    interval=1,           # 每1天
    backupCount=5,        # 保留5个备份
    encoding="utf-8"
)
handler.suffix = "%Y-%m-%d"
formatter = log.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d %(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
log.basicConfig(
    level=log.DEBUG,
    handlers=[handler]
)
