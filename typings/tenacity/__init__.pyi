from collections.abc import Callable
from typing import Any, TypeVar

_RetValT = TypeVar("_RetValT")

def retry_if_exception_type(
    exception_types: type[BaseException] | tuple[type[BaseException], ...],
) -> Callable[[Exception], bool]: ...
def stop_after_attempt(max_attempt_number: int) -> Callable[[RetryState], bool]: ...
def wait_exponential(
    multiplier: float = 1,
    max: float = 10,
    min: float = 0,
) -> Callable[[RetryState], float]: ...

class RetryState:
    attempt_number: int
    outcome: Any
    next_action: Any
    retry_object: Any

class Retrying:
    retry_state: RetryState
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, fn: Callable[..., _RetValT]) -> Callable[..., _RetValT]: ...

class RetryError(Exception):
    last_attempt: Any
    def __init__(self, last_attempt: Any) -> None: ...
