import logging
import os
import structlog


def configure_logging():
    """
    Configures logging for the application using structlog.
    Respects LOG_LEVEL and HAYSTACK_LOGGING_USE_JSON environment variables.
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    use_json = os.getenv("HAYSTACK_LOGGING_USE_JSON", "false").lower() == "true"

    try:
        log_level = getattr(logging, log_level_name)
    except AttributeError:
        log_level = logging.INFO

    # 1. Configure Standard Library Logging
    # We want standard logging (used by libraries) to be captured by structlog

    # Shared processors for both structlog and stdlib
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Specific processors for structlog
    # IMPORTANT: We must NOT render to string/json here if we are passing to stdlib logging!
    # Instead, we wrap the event dict for the ProcessorFormatter to handle.
    processors = shared_processors + [
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter
    ]

    if use_json:
        # For stdlib, we need a formatter that outputs JSON
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
    else:
        # Console rendering (pretty)
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(),
            foreign_pre_chain=shared_processors,
        )

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # BUG-11: First remove any existing handlers to avoid duplicate log output,
    # then add our configured handler.
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)

    # Set levels for specific noisy libraries if needed
    # logging.getLogger("httpx").setLevel(logging.WARNING)

    # Log that logging is configured
    logger = structlog.get_logger()
    logger.info("Logging configured", level=log_level_name, json_mode=use_json)
