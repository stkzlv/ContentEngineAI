"""The global HTTP pool survives event-loop turnover.

The pool is module-level and outlives ``asyncio.run`` calls by design, but a
cached session is bound to the loop that created it: ``session.closed`` stays
False when that loop dies, so the cache handed a dead-loop session to the
next loop and aiohttp raised ``RuntimeError: Event loop is closed`` from deep
inside a request -- or from ``close_global_pool`` cancelling a cleanup task
that belonged to the previous loop. Surfaced as order-dependent test
poisoning; the same hazard applies to any process running two loops in
sequence.
"""

from __future__ import annotations

import asyncio

from src.utils import connection_pool


def _fresh_pool() -> connection_pool.GlobalConnectionPool:
    return connection_pool.GlobalConnectionPool()


class TestLoopTurnover:
    def test_a_new_loop_gets_a_new_session(self):
        pool = _fresh_pool()

        async def grab():
            return await pool.get_session()

        first = asyncio.run(grab())
        second = asyncio.run(grab())

        assert first is not second, "a dead-loop session was served to a new loop"
        asyncio.run(pool.close())

    def test_close_on_a_new_loop_does_not_raise(self):
        """The batch's shutdown calls close_global_pool at the end of its
        own loop; leftovers from an earlier loop must be dropped, not
        awaited.
        """
        pool = _fresh_pool()

        async def grab():
            await pool.get_session()

        asyncio.run(grab())
        asyncio.run(pool.close())  # different loop than the session's

    def test_same_loop_reuses_the_session(self):
        pool = _fresh_pool()

        async def grab_twice():
            one = await pool.get_session()
            two = await pool.get_session()
            assert one is two
            await pool.close()

        asyncio.run(grab_twice())
