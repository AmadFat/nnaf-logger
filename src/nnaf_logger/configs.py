import enum
import logging
from dataclasses import asdict, dataclass, field

import colorama
from nnaf_utils.pytype import *

RESET_STYLE = colorama.Style.RESET_ALL


logging.addLevelName(15, "TRAIN")
logging.addLevelName(25, "TEST")


class LogLevel(enum.IntEnum):
    DEBUG = logging.getLevelNamesMapping()["DEBUG"]
    TRAIN = logging.getLevelNamesMapping()["TRAIN"]
    INFO = logging.getLevelNamesMapping()["INFO"]
    TEST = logging.getLevelNamesMapping()["TEST"]
    WARN = logging.getLevelNamesMapping()["WARN"]
    ERROR = logging.getLevelNamesMapping()["ERROR"]


class TimestampFmt(enum.StrEnum):
    DETAILED = "%w %m-%d-%Y %H:%M:%S.%f%:z"
    NORMAL = "%m-%d-%Y %H:%M:%S"
    SHORT = "%m-%d@%H:%M"
    MAN = "%c"


@dataclass(frozen=True)
class DefaultLogStyles:
    DEBUG: str = colorama.Fore.GREEN + colorama.Style.NORMAL
    TRAIN: str = colorama.Fore.GREEN + colorama.Style.BRIGHT
    INFO: str = colorama.Fore.CYAN + colorama.Style.NORMAL
    TEST: str = colorama.Fore.CYAN + colorama.Style.BRIGHT
    WARN: str = colorama.Fore.YELLOW + colorama.Style.NORMAL
    ERROR: str = colorama.Fore.RED + colorama.Style.NORMAL


@dataclass
class LogConfig:
    refresh_dir: bool = False
    level: LogLevel = LogLevel.INFO
    # saving configs
    dir: StrPath = "nnaf-logs"
    save_for_man: bool = True
    save_as_json: bool = False
    # terminal configs
    loggername_style: str = colorama.Fore.WHITE + colorama.Style.DIM
    timestamp_style: str = colorama.Fore.WHITE + colorama.Style.DIM
    timestamp_format: str = TimestampFmt.MAN.value
    timestamp_utc: bool = False
    epoch_style: str = colorama.Fore.BLUE + colorama.Style.NORMAL
    epoch_format: str = "04d"
    step_style: str = colorama.Fore.BLUE + colorama.Style.BRIGHT
    step_format: str = "04d"
    level_styles: dict[str, str] = field(default_factory=lambda: asdict(DefaultLogStyles()))
    event_style: str = colorama.Fore.WHITE + colorama.Style.NORMAL
    metric_style: str = colorama.Fore.WHITE + colorama.Style.BRIGHT


@dataclass
class WandbConfig:
    refresh_dir: bool = False
    mode: Literal["online", "offline"] = "offline"
    dir: StrPath = None
    # identity configs
    anonymous: Literal["never", "allow", "must"] = "never"
    api_key: str = None
    # run configs
    entity: str = None
    project: str = None
    id: str = None
    name: str = None
    group: str = None
    tags: Sequence[str] = None
    notes: str = None
    config: dict[str, Any] | str = None
    config_exclude_keys: Sequence[str] = None
    config_include_keys: Sequence[str] = None
