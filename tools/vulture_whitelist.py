# Vulture whitelist for ContentEngineAI project
# This file contains patterns that Vulture should ignore

# Async context manager parameters (required by protocol but unused)
# These are false positives - parameters are required by async context manager protocol
# but intentionally unused in __aexit__ implementations

# ruff: noqa: F821, B018
# pyright: reportUndefinedVariable=false
# type: ignore

from typing import Any

_ = Any  # Placeholder for vulture pattern matching

_.exc_type  # Async context manager __aexit__ parameter
_.exc_val  # Async context manager __aexit__ parameter
_.exc_tb  # Async context manager __aexit__ parameter
