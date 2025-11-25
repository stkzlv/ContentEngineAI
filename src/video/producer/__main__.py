# src/video/producer/__main__.py
"""Entry point for running video producer as a module."""

import asyncio

from src.video.producer.cli import main

if __name__ == "__main__":
    asyncio.run(main())
