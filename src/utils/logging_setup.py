"""Centralized logging configuration for ContentEngineAI.

This module provides standardized logging setup to avoid duplication across
producer, scraper, and other components. Includes automatic secret masking
to prevent accidental credential exposure in logs.
"""

import contextvars
import logging
import re
import sys
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

from .secrets import SECRET_KEY_PATTERNS, mask_secret

# Pre-compiled patterns for detecting secrets in log messages
# These match common secret formats (API keys, tokens, etc.)
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Generic API key patterns (sk-xxx, pk-xxx, api-xxx)
    re.compile(r"\b(sk|pk|api|key)[-_]?[a-zA-Z0-9]{16,}\b", re.IGNORECASE),
    # Bearer tokens
    re.compile(r"\bBearer\s+[a-zA-Z0-9\-_.]+\b", re.IGNORECASE),
    # Base64-like tokens (32+ chars)
    re.compile(r"\b[a-zA-Z0-9+/]{32,}={0,2}\b"),
    # UUID-like tokens
    re.compile(r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b"),
)


class SecretMaskingFilter(logging.Filter):
    """Logging filter that masks secrets in log records.

    This filter scans log messages for patterns that look like secrets
    (API keys, tokens, passwords) and masks them before output.

    The filter is thread-safe as it only modifies the current log record
    and uses immutable pattern matching.

    Examples
    --------
        >>> filter = SecretMaskingFilter()
        >>> # Applied automatically via setup_debug_logging()

    """

    __slots__ = ("_patterns", "_key_patterns")

    def __init__(self, name: str = "") -> None:
        """Initialize the secret masking filter.

        Parameters
        ----------
        name : str, optional
            Filter name (default: "")

        """
        super().__init__(name)
        self._patterns = _SECRET_VALUE_PATTERNS
        self._key_patterns = SECRET_KEY_PATTERNS

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and mask secrets in the log record.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to process

        Returns
        -------
        bool
            Always True (record is never filtered out, only modified)

        """
        # Format the message first (msg % args), then mask the result.
        # This avoids destroying %-format specifiers like %s/%d in the
        # format string (e.g., "keyword: %s" contains "KEY" which would
        # be falsely matched as a secret key pattern).
        try:
            formatted_msg = record.getMessage()
            record.msg = self._mask_string(formatted_msg)
            record.args = None
        except (TypeError, ValueError):
            # If formatting fails, fall back to masking raw parts
            if record.msg and isinstance(record.msg, str):
                record.msg = self._mask_string(record.msg)
            if record.args:
                record.args = self._mask_args(record.args)

        return True

    def _mask_string(self, text: str) -> str:
        """Mask secret patterns in a string.

        Parameters
        ----------
        text : str
            Text to scan for secrets

        Returns
        -------
        str
            Text with secrets masked

        """
        if not text:
            return text

        result = text

        # Check for key=value patterns (e.g., API_KEY=xxx, AUTH_TOKEN: yyy).
        # Require the key to be SCREAMING_SNAKE_CASE so product keywords
        # like "wireless" (contains "key") aren't falsely masked.
        result = re.sub(
            r"(\b[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)"
            r"[A-Z0-9_]*\s*[=:]\s*)(\S+)",
            lambda m: m.group(1) + mask_secret(m.group(2)),
            result,
        )

        # Mask standalone secret-like values
        for pattern in self._patterns:
            result = pattern.sub(lambda m: mask_secret(m.group(0)), result)

        return result

    def _mask_args(
        self, args: tuple[object, ...] | Mapping[str, object]
    ) -> tuple[object, ...] | dict[str, object]:
        """Mask secrets in log record args.

        Parameters
        ----------
        args : tuple or dict
            Log record args (for % formatting)

        Returns
        -------
        tuple or dict
            Args with secrets masked

        """
        if isinstance(args, dict):
            return {
                k: self._mask_string(str(v)) if isinstance(v, str) else v
                for k, v in args.items()
            }
        # Default to tuple handling
        return tuple(
            self._mask_string(str(arg)) if isinstance(arg, str) else arg for arg in args
        )


# A run appends, so the file is a history rather than only the last run.
# One file per component per day, kept this long. Size rotation gave
# `scraper.log.2` with no idea what dates it covered, and the pipeline runs
# on a daily cadence, so the questions are by date. A dated file opened in
# append mode is also safe for two processes at once (the analytics timer
# and a manual publisher run), where a rotating handler in each would
# rename the same file from under the other.
LOG_RETENTION_DAYS = 45
_DATED_LOG = re.compile(r"^(?P<stem>[A-Za-z0-9_]+)-(?P<date>\d{4}-\d{2}-\d{2})\.log$")


def _today() -> date:
    return date.today()


def dated_log_path(log_file: Path) -> Path:
    """`logs/producer.log` -> `logs/producer-2026-09-19.log`; a dated name stays."""
    if _DATED_LOG.match(log_file.name):
        return log_file
    return log_file.with_name(
        f"{log_file.stem}-{_today().isoformat()}{log_file.suffix}"
    )


def prune_old_logs(log_dir: Path, keep_days: int = LOG_RETENTION_DAYS) -> list[Path]:
    """Remove dated log files older than `keep_days`, by the date in the name.

    Only `<stem>-YYYY-MM-DD.log` names are candidates; the size-rotated
    files of earlier releases and anything else in the directory stay.
    """
    removed: list[Path] = []
    if not log_dir.is_dir():
        return removed
    today = _today()
    for path in log_dir.iterdir():
        match = _DATED_LOG.match(path.name)
        if not match or not path.is_file():
            continue
        try:
            written = date.fromisoformat(match["date"])
        except ValueError:
            continue
        if (today - written).days > keep_days:
            try:
                path.unlink()
            except OSError as error:
                logging.getLogger(__name__).debug("Could not prune %s: %s", path, error)
                continue
            removed.append(path)
    return removed


# The run and product a record belongs to. Context variables rather than
# globals so an awaited step logs under the product that awaited it, and a
# value bound inside `log_context` is gone when the block ends.
UNBOUND = "-"
RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "log_run_id", default=UNBOUND
)
PRODUCT_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "log_product_id", default=UNBOUND
)


# A thread started by an executor begins with an empty context, so the run
# id also lives process-wide: a record from such a thread still names the
# run. The product id has no such fallback; a thread that should carry it
# is handed `contextvars.copy_context().run` as its callable.
_process_run_id: str | None = None


def new_run_id() -> str:
    return uuid.uuid4().hex[:8]


def current_run_id() -> str | None:
    """The run id bound by the entry point, or None before one is."""
    bound = RUN_ID.get()
    return _process_run_id if bound == UNBOUND else bound


@contextmanager
def log_context(
    *, run_id: str | None = None, product_id: str | None = None
) -> Iterator[None]:
    """Bind ids to every record logged inside the block; restored on exit."""
    tokens = []
    if run_id is not None:
        tokens.append((RUN_ID, RUN_ID.set(run_id)))
    if product_id is not None:
        tokens.append((PRODUCT_ID, PRODUCT_ID.set(product_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class ContextFilter(logging.Filter):
    """Stamp the bound run and product ids onto each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = current_run_id() or UNBOUND
        record.product_id = PRODUCT_ID.get()
        return True


class IsoFormatter(logging.Formatter):
    """ISO 8601 timestamps with milliseconds and the local UTC offset.

    The schedule is in the publisher's configured zone and the provider
    reports UTC, so the offset belongs on every line; `datefmt` alone
    would drop the milliseconds.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        stamp = datetime.fromtimestamp(record.created).astimezone()
        return stamp.isoformat(timespec="milliseconds")


FILE_FORMAT = (
    "%(asctime)s - %(run_id)s - %(product_id)s - %(name)s - %(levelname)s"
    " - %(funcName)s:%(lineno)d - %(message)s"
)
VERBOSE_CONSOLE_FORMAT = (
    "%(asctime)s - %(product_id)s - %(name)s - %(levelname)s - %(message)s"
)

# Third-party loggers held at WARNING in every mode. The list used to apply
# only outside debug mode, and every documented command passes --debug, so it
# never applied in practice: Pillow's PNG plugin logs each chunk at DEBUG
# (45% of a producer log), and the Gemini SDK logs "AFC is enabled" at INFO
# on every call. `google_genai` is its own top-level logger, not `google`.
QUIET_LOGGERS: tuple[str, ...] = (
    "PIL",
    "google_genai",
    "httpx",
    "httpcore",
    "hpack",
    "numba",
    "TTS",
    "TTS.tts.utils.text.phonemizers",
)
# INFO under --debug, WARNING otherwise: their INFO lines are worth having
# when debugging, their DEBUG lines never are.
DEBUG_INFO_LOGGERS: tuple[str, ...] = ("google", "aiohttp", "asyncio", "urllib3")


def setup_debug_logging(
    log_file: Path,
    debug_mode: bool = False,
    verbose: bool = False,
    component_name: str = "ContentEngineAI",
    mark_run: bool = True,
    run_id: str | None = None,
) -> Path:
    """Configure standardized logging with console and file handlers.

    Parameters
    ----------
    log_file : Path
        The file records go to, appended. Entry points pass
        `dated_log_path(<logs dir>/<component>.log)`, one file per day.
    debug_mode : bool, optional
        Enable DEBUG level logging (default: False = INFO level)
    verbose : bool, optional
        Enable verbose console formatting (default: False)
    component_name : str, optional
        Name of the component for logging messages (default: "ContentEngineAI")
    mark_run : bool, optional
        Write an INFO run marker (default: True). Pass False when configuring
        logging at import rather than at the start of a run.
    run_id : str, optional
        The id bound to this run's records; a fresh one is made when marking
        a run without one. Not bound when `mark_run` is False.

    Notes
    -----
    - Console output uses simplified format by default, detailed format when verbose
    - File output always uses detailed format with function names and line numbers
    - The file is appended to, so importing a module that configures logging
      cannot destroy an earlier run's log; dated files in its directory
      older than LOG_RETENTION_DAYS are removed here
    - A run marker is written at INFO unless `mark_run` is False. Pass False
      when configuring logging at import rather than at the start of a run,
      or the marker records the import and misleads whoever reads it
    - The loggers in `QUIET_LOGGERS` sit at WARNING in every mode; those in
      `DEBUG_INFO_LOGGERS` at INFO under debug mode and WARNING otherwise

    """
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # Clear any existing handlers to avoid duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler configuration
    console_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        console_formatter: logging.Formatter = IsoFormatter(VERBOSE_CONSOLE_FORMAT)
    else:
        console_formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    # Appending, not overwriting. `mode="w"` truncated the file when the
    # handler was constructed, so anything that merely imported a module which
    # configures logging destroyed the previous run's log before writing a
    # line -- which is how a scraper log was lost to a tool that only meant to
    # read the source. The day in the file name is the size bound now.
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(IsoFormatter(FILE_FORMAT))
    file_handler.setLevel(log_level)

    # Create shared secret masking filter
    secret_filter = SecretMaskingFilter()

    # Apply the filters to both handlers
    context_filter = ContextFilter()
    for handler in (console_handler, file_handler):
        handler.addFilter(context_filter)
        handler.addFilter(secret_filter)

    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    for lib in QUIET_LOGGERS:
        logging.getLogger(lib).setLevel(logging.WARNING)
    for lib in DEBUG_INFO_LOGGERS:
        logging.getLogger(lib).setLevel(logging.INFO if debug_mode else logging.WARNING)
    # Websocket cleanup "goodbye" errors are noise at any level
    logging.getLogger("websocket").setLevel(logging.CRITICAL)

    # Run boundary, at INFO on purpose. The file is appended to, so without a
    # marker visible at the default level a reader greping the log -- which is
    # how the docs say to verify a render -- can match the previous run's line
    # instead of this one's.
    #
    # Gated because configuring logging is not the same event as starting a
    # run. The scraper configures it at module import, so every producer,
    # publisher and batch invocation imports it -- and so does `--help` --
    # which wrote a marker claiming a scrape had begun, with no completion
    # line after it. That is worse than no marker: it is a boundary an
    # operator would trust.
    logger = logging.getLogger(component_name)
    if mark_run:
        global _process_run_id
        _process_run_id = run_id or new_run_id()
        RUN_ID.set(_process_run_id)
        logger.info("=== %s run starting (run %s) ===", component_name, RUN_ID.get())
    logger.debug(
        "Logging initialized: level=%s, log_file=%s, verbose=%s",
        logging.getLevelName(log_level),
        log_file,
        verbose,
    )
    for path in prune_old_logs(log_file.parent):
        logger.debug("Pruned log older than %d days: %s", LOG_RETENTION_DAYS, path)
    return log_file
