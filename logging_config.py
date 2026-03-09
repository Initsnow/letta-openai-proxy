import logging
import os
import sys

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Route all stdlib logging records (uvicorn, httpx, haystack...) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore[assignment]

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


_configured = False


def configure_logging() -> None:
    """
    Configure loguru as the single logging backend for the whole application.

    - Reads LOG env var (default INFO) — same var hayhooks reads.
    - Re-routes all stdlib logging (uvicorn, httpx, haystack…) through loguru
      via InterceptHandler so everything appears in one unified stream.
    - Idempotent: safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    log_level = os.getenv("LOG", "INFO").upper()

    # Remove all existing sinks to ensure a clean state
    logger.remove()
    _configured = True

    use_json = os.getenv("LOG_JSON", "false").lower() == "true"
    if use_json:
        logger.add(sys.stderr, level=log_level, serialize=True)
    else:
        logger.add(
            sys.stderr,
            level=log_level,
            colorize=True,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
        )

    # Intercept stdlib logging so uvicorn / httpx / haystack go through loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
