"""Person 4, approach C1: independent-correctness selection on top of the C0 pipeline.

The core pipeline (capstone_person4) carries only the method names and a one-line seam;
this package holds the question-only generation request/ingest, the mistake indicator,
the exact integer C1 utilities for the shared CP-SAT selector, and the correctness
diagnostics. It is outside the core fingerprints by design.
"""
