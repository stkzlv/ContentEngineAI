"""The global batch pipeline's phases, one module each.

Each phase is a function taking the batch configuration and the few
orchestrator reads it needs as parameters, and returning its typed
summary. The orchestrator in `src.pipeline.global_batch` keeps a
delegating method per phase, so `run_pipeline` and the tests that stub a
phase on the instance are unchanged.
"""
