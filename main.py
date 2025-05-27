from nnaf_logger.v2 import Loggerv2, LogLevel
import colorama
import time

colorama.init()

logger = Loggerv2(
    name="test_logger",
    log_level=LogLevel.DEBUG,
    log_save_for_man=True,
    log_save_as_json=True,
)
logger.debug("This is a debug message", epoch=1, step=1, metric1="value1", metric2="value2")
logger.info("This is an info message", epoch=1, step=2)
logger.warn("This is a warning message", epoch=1, step=3)

try:
    1 / 0
except Exception as e:
    logger.error("An exception occurred", epoch=1, step=5)

logger.watch_pytorch_module