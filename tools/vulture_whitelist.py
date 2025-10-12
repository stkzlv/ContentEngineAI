# Vulture whitelist for ContentEngineAI project
# This file contains patterns that Vulture should ignore

# Async context manager parameters (required by protocol but unused)
_.exc_type  # Async context manager __aexit__ parameter
_.exc_val   # Async context manager __aexit__ parameter
_.exc_tb    # Async context manager __aexit__ parameter
