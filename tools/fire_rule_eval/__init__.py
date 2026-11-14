"""Offline evaluation harness for the classical fire-detection rule (issue #104).

Two stages keep experiments cheap:

* ``extract`` decodes videos once at the production check rate and stores per-check features
  (the same colour / motion operations as ``CPUFireDetector``, plus extra statistics used by
  candidate rules).
* ``report`` applies rules and threshold sweeps to the stored features and prints recall / false
  positive tables.

Run ``PYTHONPATH=. python -m tools.fire_rule_eval --help`` for usage.
"""
