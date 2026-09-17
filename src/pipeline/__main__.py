"""Entry point for running the global batch pipeline as a package.

Enables execution via:
    python -m src.pipeline
"""

import asyncio

from src.pipeline.cli import main

if __name__ == "__main__":
    asyncio.run(main())
