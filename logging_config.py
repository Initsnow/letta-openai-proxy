import logging
import os
import sys
import structlog

def configure_logging():
    """
    Configures logging for the application using structlog.
    Respects LOG_LEVEL and HAYSTACK_LOGGING_USE_JSON environment variables.
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    use_json = os.getenv("HAYSTACK_LOGGING_USE_JSON", "false").lower() == "true"
    
    # If not explicitly set to use JSON, check if we are in a TTY
    if not use_json and not sys.stdout.isatty():
        # Default to JSON in non-interactive environments (unless explicitly disabled, but we'll stick to a simple rule)
        # Actually, let's just stick to the env var or default to console for now unless in prod.
        pass

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
    if use_json:
        # JSON rendering
        processors = shared_processors + [
            structlog.processors.JSONRenderer()
        ]
        # For stdlib, we need a formatter that outputs JSON
        # effectively, we'll just use structlog's ProcessorFormatter
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
    else:
        # Console rendering (pretty)
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer()
        ]
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

    # Apply configuration to startdard library root logger
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    
    # Avoid duplicate logs if other handlers are already attached (e.g. from uvicorn)
    # But usually creating a fresh handler is fine if we clear others or if this is the main entrypoint
    # For safe measure, let's remove existing handlers from root to avoid double logging
    if root_logger.handlers:
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
    root_logger.addHandler(handler)

    # Set levels for specific noisy libraries if needed
    # logging.getLogger("httpx").setLevel(logging.WARNING)

    # Log that logging is configured
    logger = structlog.get_logger()
    logger.info("Logging configured", level=log_level_name, json_mode=use_json)

