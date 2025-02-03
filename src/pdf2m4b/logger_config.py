import os
import logging
import structlog
from structlog.stdlib import BoundLogger
from structlog.typing import EventDict
# from controller.aws_utils import aws_instance_id  # adjust or remove if not used

# ANSI escape codes for colors.
_ANSI_BLUE = "\033[34m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"

def _add_color_to_level_str(level: str) -> str:
    level = level.lower()
    padded_level = level.ljust(9)
    if level == "debug":
        return f"{_ANSI_BLUE}{_ANSI_BOLD}{padded_level}{_ANSI_RESET}"
    elif level == "info":
        return f"{_ANSI_GREEN}{_ANSI_BOLD}{padded_level}{_ANSI_RESET}"
    elif level == "warning":
        return f"{_ANSI_YELLOW}{_ANSI_BOLD}{padded_level}{_ANSI_RESET}"
    elif level == "error":
        return f"{_ANSI_RED}{_ANSI_BOLD}{padded_level}{_ANSI_RESET}"
    elif level == "critical":
        return f"{_ANSI_MAGENTA}{_ANSI_BOLD}{padded_level}{_ANSI_RESET}"
    return level

def _add_log_level(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    if method_name == "warn":
        method_name = "warning"
    elif method_name == "exception":
        method_name = "error"
    event_dict["level"] = _add_color_to_level_str(method_name)
    return event_dict

# Use colorized logs if NICE_COLOR_LOGS is set (default true).
NICE_COLOR_LOGS = os.getenv("NICE_COLOR_LOGS", "1") == "1"

_processors = [
    structlog.processors.CallsiteParameterAdder(
        [structlog.processors.CallsiteParameter.MODULE,
         structlog.processors.CallsiteParameter.FUNC_NAME]
    ),
    _add_log_level,
    structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
    # Render logs as colorized output (or JSON if not).
    structlog.dev.ConsoleRenderer() if NICE_COLOR_LOGS else structlog.processors.JSONRenderer()
]

# Set up a logger that writes both to a file and to the console.
_system_logger = logging.getLogger("pdf2m4b")
_system_logger.setLevel(logging.DEBUG)
_file_handler = logging.FileHandler("app.log")
_file_handler.setLevel(logging.DEBUG)
_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.DEBUG)
_system_logger.addHandler(_file_handler)
_system_logger.addHandler(_stream_handler)

structlog.configure(
    processors=_processors,  # type: ignore
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

def get_logger() -> BoundLogger:
    return structlog.wrap_logger(_system_logger)

logger = get_logger()
