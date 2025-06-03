from nnaf_utils.pytype import *

from .configs import LogConfig, LogLevel, WandbConfig


class Loggerv2:
    def __init__(
        self,
        name: str = None,
        print_info_interval: int = 1,

        log_config: LogConfig = None,
        wandb_config: WandbConfig = None,
    ):
        import structlog

        from .structlog_extentsions import ExtendedLogger

        self._init_wandb(wandb_config)
        self._init_structlog(log_config)
        self.logger: ExtendedLogger = structlog.get_logger()
        if name:
            self.logger = self.logger.bind(logger=name)

        self.now_index = 1
        self.print_info_interval = print_info_interval
        self.single_step_dict = dict()

    def _init_structlog(self, log_config: LogConfig):
        import structlog
        from structlog._native import _make_filtering_bound_logger
        from structlog.processors import TimeStamper

        from .structlog_extentsions import (
            ExtendedLoggerFactory,
            add_log_level,
            get_console_renderer,
        )

        processors = [
            add_log_level,
            TimeStamper(
                fmt=log_config.timestamp_format,
                utc=log_config.timestamp_utc,
            ),
        ]

        if log_config.save_for_man or log_config.save_as_json:

            from .structlog_extentsions import JsonFileHandler, ManFileHandler

            Path(log_config.dir).mkdir(parents=True, exist_ok=True)
            if log_config.refresh_dir:
                from nnaf_utils.filesystem import refresh_obj
                refresh_obj(Path(log_config.dir))

            if self.wandb_run:
                filename = self.wandb_run.id
            
            else:
                from datetime import datetime

                import xxhash

                filename = xxhash.xxh32_hexdigest(datetime.now().isoformat().encode())
            
            if log_config.save_for_man:
                processors.append(
                    ManFileHandler(
                        fileobj=Path(log_config.dir) / f"{filename}@man.log",
                        epoch_format=log_config.epoch_format,
                        step_format=log_config.step_format,
                    )
                )
            
            if log_config.save_as_json:
                processors.append(
                    JsonFileHandler(
                        fileobj=Path(log_config.dir) / f"{filename}@json.log",
                    )
                )

        processors.append(
            get_console_renderer(
                loggername_style=log_config.loggername_style,
                timestamp_style=log_config.timestamp_style,
                epoch_style=log_config.epoch_style,
                epoch_format=log_config.epoch_format,
                step_style=log_config.step_style,
                step_format=log_config.step_format,
                level_styles=log_config.level_styles,
                event_style=log_config.event_style,
                metric_style=log_config.metric_style,
            )
        )

        structlog.configure(
            processors=processors,
            wrapper_class=_make_filtering_bound_logger(log_config.level.value),
            context_class=dict,
            logger_factory=ExtendedLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _init_wandb(self, wandb_config: WandbConfig = None):

        if wandb_config:
            try:
                import wandb
            except Exception as e:
                self.warn(f"`W&B` features will be disabled: {e}")
                wandb_config = None

        try:
            if wandb_config:

                if wandb_config.api_key is None:
                    import os
                    os.environ["WANDB_MODE"] = "offline"
            
                wandb.login(
                    anonymous=wandb_config.anonymous,
                    key=wandb_config.api_key,
                    relogin=True,
                    force=True,
                    timeout=5
                )

                (Path(wandb_config.dir) / "wandb").mkdir(parents=True, exist_ok=True)
                if wandb_config.refresh_dir:
                    from nnaf_utils.filesystem import refresh_obj
                    refresh_obj(Path(wandb_config.dir) / "wandb")

                self.wandb_run = wandb.init(
                    entity=wandb_config.entity,
                    project=wandb_config.project,
                    dir=wandb_config.dir,
                    id=wandb_config.id,
                    name=wandb_config.name,
                    group=wandb_config.group,
                    tags=wandb_config.tags,
                    notes=wandb_config.notes,
                    mode=wandb_config.mode,
                    anonymous=wandb_config.anonymous,
                    config=wandb_config.config,
                    config_exclude_keys=wandb_config.config_exclude_keys,
                    config_include_keys=wandb_config.config_include_keys,
                    force=True,
                    reinit="create_new",
                    resume="auto",
                )

                self.wandb_watch_pytorch_module = self.wandb_run.watch
                self.wandb_log_model = self.wandb_run.log_model
                self.wandb_log = self.wandb_run.log

            else:
                self.wandb_run = None

        except Exception as e:
            self.error(f"Exception occurred: {e}", shutdown=False)
            self.close()

    def close(self):
        import colorama
        colorama.deinit()
        del self.logger
        if self.wandb_run is not None:
            self.wandb_run.log({})
            self.wandb_run.finish()
            del self.wandb_run

    """Normal logging methods."""

    def debug(self, event=None, **kwargs):
        self.logger.debug(**self._filter_kwargs_none(event=event), **kwargs)

    def train(self, event=None, **kwargs):
        self.logger.train(**self._filter_kwargs_none(event=event), **kwargs)
    
    def info(self, event=None, **kwargs):
        self.logger.info(**self._filter_kwargs_none(event=event), **kwargs)
    
    def test(self, event=None, **kwargs):
        self.logger.test(**self._filter_kwargs_none(event=event), **kwargs)
    
    def warn(self, event=None, **kwargs):
        self.logger.warn(**self._filter_kwargs_none(event=event), **kwargs)
    
    def error(self, event=None, exc_info=True, shutdown=True, **kwargs):
        try:
            self.logger.error(**self._filter_kwargs_none(event=event), **kwargs, exc_info=exc_info)
        finally:
            if shutdown:
                self.close()
                exit(1)

    """Neural network logging methods.
    
    This is where level ``train`` and ``eval`` are really used. In general, we use a window tracer and global
    index keepers to track the whole process. We now use tracer for batch uploading instead of smoothing.

    """ 

    def _process_level(
        self,
        level: LogLevel | str,
        lift: LogLevel | str = None,
    ) -> str:
        if lift:
            level = max(
                isinstance(level, str) and getattr(LogLevel, level.upper()) or level,
                isinstance(lift, str) and getattr(LogLevel, lift.upper()) or lift,
            )
        
        match level:
            case LogLevel():
                return level.name.lower()
            case str():
                return getattr(LogLevel, level.upper()).name.lower()

    def _filter_kwargs_none(
        self,
        **kwargs,
    ) -> dict:
        popks = [k for k, v in kwargs.items() if v is None]
        for k in popks:
            kwargs.pop(k)
        return kwargs

    def add(
        self,
        event: Any = None,
        level: LogLevel | str = "train",
        epoch: int = None,
        step: int = None,
        tag: str = None,
        **kwargs,
    ):
        """In every epoch you can call :func:`add` many times."""
        for k, v in kwargs.items():
            k = (f"{tag}/" if tag else "") + k
            assert k.count("/") <= 1
            self.single_step_dict[k] = v

        if event:
            lift = "info" if self.now_index % self.print_info_interval == 0 else None
            level = self._process_level(level, lift=lift)
            getattr(self, level)(event=event, **self._filter_kwargs_none(epoch=epoch, step=step))

    def commit(
        self,
        event: Any = None,
        level: LogLevel | str = "train",
        epoch: int = None,
        step: int = None,
    ):
        """In the end of every epoch you should call :func:`commit`."""
        lift = "info" if self.now_index % self.print_info_interval == 0 else None
        level = self._process_level(level=level, lift=lift)
        getattr(self, level)(**self._filter_kwargs_none(event=event, epoch=epoch, step=step, **self.single_step_dict))

        if self.wandb_run:
            self.wandb_log(
                self.single_step_dict,
                step=self.now_index,
                commit=(self.now_index % self.print_info_interval == 0),
            )

        self.single_step_dict.clear()
        self.now_index += 1
