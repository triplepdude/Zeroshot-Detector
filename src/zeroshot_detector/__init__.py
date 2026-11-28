"""Zero-shot detection of military content in short-form video frames.

The package is split so that the light-weight parts (taxonomy, rules, decision
logic, aggregation, evaluation and analysis) import only ``numpy``/``pandas``;
the heavy model and I/O dependencies (``torch``, ``transformers``, ``opencv``,
``minio``) are imported lazily by the modules that need them.
"""

__version__ = "1.0.0"
