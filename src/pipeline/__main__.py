"""Entry point for running global batch pipeline as a module.

Enables execution via:
    python -m src.pipeline.global_batch
"""

import asyncio

from src.pipeline.global_batch import main

if __name__ == "__main__":
    asyncio.run(main())
