from nnaf._types import *
from nnaf.parallel import run_in_another_thread
from .miscs import *
import colorama
colorama.init()


class Loggerv2:
    def __init__(
        self,
        name: str = None,
        print_info_interval: int = 1,

        log_level: Literal[LogLevel.DEBUG, LogLevel.INFO] = LogLevel.INFO,
        log_dir: str = "./nnaf-logs",
        log_save_for_man: bool = True,
        log_save_as_json: bool = False,
        log_name_style: str = colorama.Back.WHITE + colorama.Fore.BLACK + colorama.Style.BRIGHT,
        log_timestamp_style: str = colorama.Fore.WHITE + colorama.Style.DIM,
        log_timestamp_format: TimestampFmt = TimestampFmt.MAN,
        log_timestamp_utc: bool = False,
        log_epoch_style: str = colorama.Fore.CYAN + colorama.Style.BRIGHT,
        log_epoch_format: str = "04d",
        log_step_style: str = colorama.Fore.CYAN + colorama.Style.NORMAL,
        log_step_format: str = "04d",
        log_level_style: str = colorama.Style.BRIGHT,
        log_message_style: str = colorama.Back.MAGENTA + colorama.Fore.LIGHTWHITE_EX + colorama.Style.BRIGHT,
        log_metric_style: str = colorama.Fore.WHITE + colorama.Style.BRIGHT,

        use_wandb: bool = False,
        wandb_mode: Literal["online", "offline"] = "offline",
        wandb_dir: str = "./wandb",
        wandb_entity: str = None,
        wandb_project: str = None,
        wandb_run_id: str = None,
        wandb_run_name: str = None,
        wandb_run_group: str = None,
        wandb_run_tags: list[str] = None,
        wandb_run_notes: str = None,
        wandb_job_type: Literal["train", "eval"] = "train",
        wandb_anonymous_log: Literal["allow", "must", "never"] = "never",
    ):
        import structlog, logging, sys
        from structlog.dev import (
            ConsoleRenderer,
            Column,
            KeyValueColumnFormatter,
            LogLevelColumnFormatter
        )
        from structlog.stdlib import (
            ProcessorFormatter,
            PositionalArgumentsFormatter
        )
    
        def block_root_name(logger, name, event_dict):
            if event_dict.get("logger") == "root":
                event_dict.pop("logger")
            return event_dict

        structlog_base_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            block_root_name,
            PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(
                fmt=log_timestamp_format.value,
                utc=log_timestamp_utc,
            ),
            ProcessorFormatter.wrap_for_formatter,
        ]

        console_renderer = ConsoleRenderer(
            columns=[
                Column("logger", KeyValueColumnFormatter(None, log_name_style, colorama.Style.RESET_ALL, str)),
                Column("timestamp", KeyValueColumnFormatter(None, log_timestamp_style, colorama.Style.RESET_ALL, str)),
                Column("epoch", KeyValueColumnFormatter(None, log_epoch_style, colorama.Style.RESET_ALL, lambda x: f"{x:{log_epoch_format}}", prefix="E")),
                Column("step", KeyValueColumnFormatter(None, log_step_style, colorama.Style.RESET_ALL, lambda x: f"{x:{log_step_format}}", prefix="S")),
                Column("level", LogLevelColumnFormatter(
                    level_styles={
                        "debug": colorama.Fore.GREEN + log_level_style,
                        "info": colorama.Fore.GREEN + log_level_style,
                        "warn": colorama.Fore.YELLOW + log_level_style,
                        "error": colorama.Fore.RED + log_level_style,
                    },
                    reset_style=colorama.Style.RESET_ALL,
                )),
                Column("event", KeyValueColumnFormatter(None, log_message_style, colorama.Style.RESET_ALL, str)),
                Column("", KeyValueColumnFormatter(log_metric_style, log_metric_style, colorama.Style.RESET_ALL, str)),
            ],
            exception_formatter=structlog.dev.rich_traceback,
            sort_keys=False,
        )

        structlog.configure(
            processors=structlog_base_processors,
            wrapper_class=structlog.make_filtering_bound_logger(log_level.value),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True
        )

        try:
            import wandb
        except ImportError:
            self.warn("`wandb` is not installed. Related features will be disabled.")
            use_wandb = False

        if use_wandb and wandb_project is not None:
            self.wandb_run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                dir=wandb_dir,
                id=wandb_run_id,
                name=wandb_run_name,
                group=wandb_run_group,
                tags=wandb_run_tags,
                notes=wandb_run_notes,
                job_type=wandb_job_type,
                mode=wandb_mode,
                force=False,
                anonymous=wandb_anonymous_log,
                reinit="create_new",
                resume="auto",
            )
        else:
            self.wandb_run = None

        self.std_logger = logging.getLogger(name)
        self.std_logger.setLevel(log_level.value)
        self.std_logger.propagate = False
        self.std_logger.handlers.clear()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ProcessorFormatter(console_renderer))
        self.std_logger.addHandler(console_handler)
        self.logger = structlog.get_logger(name)

        if log_save_for_man or log_save_as_json:
            from logging import handlers

            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            if self.wandb_run is not None:
                log_file_name = self.wandb_run.id
            else:
                import xxhash
                from datetime import datetime
                log_file_name = xxhash.xxh32_hexdigest(datetime.now().isoformat().encode())

            if log_save_for_man:
                man_file_handler = handlers.RotatingFileHandler(
                    filename=log_dir / f"{log_file_name}@man.log",
                    maxBytes=1 << 25,  # 32 MiB
                    backupCount=5,
                    encoding="utf-8",
                )
                man_file_renderer = ConsoleRenderer(
                    columns=[
                        Column("logger", KeyValueColumnFormatter(None, "", "", str)),
                        Column("timestamp", KeyValueColumnFormatter(None, "", "", str)),
                        Column("epoch", KeyValueColumnFormatter(None, "", "", lambda x: f"{x:{log_epoch_format}}", prefix="E")),
                        Column("step", KeyValueColumnFormatter(None, "", "", lambda x: f"{x:{log_step_format}}", prefix="S")),
                        Column("level", LogLevelColumnFormatter(None, None)),
                        Column("event", KeyValueColumnFormatter(None, "", "", str)),
                        Column("", KeyValueColumnFormatter("", "", "", str)),
                    ],
                    exception_formatter=structlog.dev.RichTracebackFormatter(color_system="auto"),
                    sort_keys=False,
                    colors=False,
                )
                man_file_handler.setFormatter(
                    structlog.stdlib.ProcessorFormatter(
                        processor=man_file_renderer,
                        keep_exc_info=True,
                    )
                )
                self.std_logger.addHandler(man_file_handler)
            
            if log_save_as_json:
                json_file_handler = handlers.RotatingFileHandler(
                    filename=log_dir / f"{log_file_name}@json.log",
                    maxBytes=1 << 25,  # 32 MiB
                    backupCount=5,
                    encoding="utf-8",
                )
                json_file_handler.setFormatter(
                    structlog.stdlib.ProcessorFormatter(
                        processor=structlog.processors.JSONRenderer(),
                        keep_exc_info=True,
                    )
                )
                self.std_logger.addHandler(json_file_handler)
        
        if name is None:
            self.warn("Logger with `name=None` is default as root logger in logging." \
                      "This may raise unexpected conflicts.")

    def log(
        self,
        msg,
        level: Literal["debug", "info", "warn", "error"] = None,
        timeout: float = 5,
        **kwargs,
    ):
        assert level is not None
        if level == "debug":
            self.debug(msg, timeout=timeout, **kwargs)
        if level == "info":
            self.info(msg, timeout=timeout, **kwargs)
        if level == "warn":
            self.warn(msg, timeout=timeout, **kwargs)
        if level == "error":
            self.error(msg, **kwargs)
    
    def debug(self, msg, timeout=5, **kwargs):
        run_in_another_thread(
            self.logger.debug,
            msg,
            level="debug",
            timeout=timeout,
            error_callback=self.error,
            **kwargs,
        )
    
    def info(self, msg, timeout=5, **kwargs):
        run_in_another_thread(
            self.logger.info,
            msg,
            level="info",
            timeout=timeout,
            error_callback=self.error,
            **kwargs,
        )

    def warn(self, msg, timeout=5, **kwargs):
        run_in_another_thread(
            self.logger.warn,
            msg,
            level="warn",
            timeout=timeout,
            error_callback=self.error,
            **kwargs,
        )
    
    def error(self, *args, exc_info=True, **kwargs):
        self.logger.error(*args, level="error", exc_info=exc_info, **kwargs)
        self.close()
        exit(1)

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
        colorama.deinit()
        logging.shutdown()
