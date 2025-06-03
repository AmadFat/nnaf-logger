import json
from dataclasses import asdict

from nnaf_utils.pytype import *
from structlog import PrintLogger, _native

from .configs import RESET_STYLE, DefaultLogStyles, LogLevel

_native.NAME_TO_LEVEL["train"] = LogLevel.TRAIN.value
_native.NAME_TO_LEVEL["test"] = LogLevel.TEST.value
_native.LEVEL_TO_NAME[LogLevel.TRAIN.value] = "train"
_native.LEVEL_TO_NAME[LogLevel.TEST.value] = "test"


class ExtendedLogger(PrintLogger):
    debug = train = info = test = warn = error = PrintLogger.msg


class ExtendedLoggerFactory:
    def __call__(self, *args: Any) -> ExtendedLogger:
        return ExtendedLogger(*args)


def add_log_level(
    logger,
    method_name,
    event_dict,
):
    if method_name.upper() == "WARNING":
        method_name = "WARN"
    event_dict["level"] = method_name.upper()
    return event_dict


def get_console_renderer(
    loggername_style: str,
    timestamp_style: str,
    epoch_style: str,
    epoch_format: str,
    step_style: str,
    step_format: str,
    level_styles: dict[str, str],
    event_style: str,
    metric_style: str,
):
    from structlog.dev import (
        Column,
        ConsoleRenderer,
        KeyValueColumnFormatter,
        LogLevelColumnFormatter,
        RichTracebackFormatter,
    )

    return ConsoleRenderer(
        columns=[
            Column("logger", KeyValueColumnFormatter(None, loggername_style, RESET_STYLE, str)),
            Column("timestamp", KeyValueColumnFormatter(None, timestamp_style, RESET_STYLE, str)),
            Column("level", LogLevelColumnFormatter(level_styles, RESET_STYLE)),
            Column("epoch", KeyValueColumnFormatter(None, epoch_style, RESET_STYLE, lambda x: f"E{x:{epoch_format}}")),
            Column("step", KeyValueColumnFormatter(None, step_style, RESET_STYLE, lambda x: f"S{x:{step_format}}")),
            Column("event", KeyValueColumnFormatter(None, event_style, RESET_STYLE, str)),
            Column("", KeyValueColumnFormatter(metric_style, metric_style, RESET_STYLE, str)),
        ],
        exception_formatter=RichTracebackFormatter(color_system="truecolor"),
        sort_keys=False,
    )


def get_manfile_renderer(
    epoch_format: str,
    step_format: str,
):
    from structlog.dev import (
        Column,
        ConsoleRenderer,
        KeyValueColumnFormatter,
        LogLevelColumnFormatter,
        RichTracebackFormatter,
    )

    max_level_len = max(len(k) for k, v in asdict(DefaultLogStyles()).items())
    return ConsoleRenderer(
        columns=[
            Column("logger", KeyValueColumnFormatter(None, "", "", str)),
            Column("timestamp", KeyValueColumnFormatter(None, "", "", str)),
            Column("level", LogLevelColumnFormatter({" " * max_level_len: ""}, "")),
            Column("epoch", KeyValueColumnFormatter(None, "", "", lambda x: f"E{x:{epoch_format}}")),
            Column("step", KeyValueColumnFormatter(None, "", "", lambda x: f"S{x:{step_format}}")),
            Column("event", KeyValueColumnFormatter(None, "", "", str)),
            Column("", KeyValueColumnFormatter("", "", "", str)),
        ],
        exception_formatter=RichTracebackFormatter(color_system="auto"),
        sort_keys=False,
    )


class ManFileHandler:
    def __init__(
        self,
        fileobj: StrPath,
        epoch_format: str,
        step_format: str,
    ):
        self.renderer = get_manfile_renderer(epoch_format, step_format)

        from nnaf_utils.filesystem import refresh_obj

        refresh_obj(fileobj, strict=True)
        self.fileobj = Path(fileobj)

    def __call__(
        self,
        logger,
        method_name,
        event_dict,
    ):
        with self.fileobj.open("a+", encoding="utf-8") as f:
            f.write(self.renderer(logger, method_name, event_dict.copy()) + "\n")
        return event_dict


class JsonFileHandler:
    def __init__(
        self,
        fileobj: StrPath,
    ):
        from structlog.processors import ExceptionRenderer

        self.exception_renderer = ExceptionRenderer()

        from nnaf_utils.filesystem import refresh_obj

        refresh_obj(fileobj, strict=True)
        self.fileobj = Path(fileobj)

    def __call__(
        self,
        logger,
        method_name,
        event_dict,
    ):
        with self.fileobj.open("a+", encoding="utf-8") as f:
            f.write(json.dumps(self.exception_renderer(logger, method_name, event_dict.copy())) + "\n")
        return event_dict
