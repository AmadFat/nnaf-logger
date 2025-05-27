import enum
import logging


class LogLevel(enum.IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARN = logging.WARN
    ERROR = logging.ERROR


class TimestampFmt(enum.Enum):
    DETAILED = "%w %m-%d-%Y %H:%M:%S.%f%:z"
    NORMAL = "%m-%d-%Y %H:%M:%S"
    SHORT = "%m/%d@%H:%M"
    MAN = "%c"
