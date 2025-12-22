"""Entry point for running Late.dev publisher as a module.

Enables execution via:
    python -m src.publisher.late
"""

import asyncio

from src.publisher.late.cli import main

if __name__ == "__main__":
    asyncio.run(main())
