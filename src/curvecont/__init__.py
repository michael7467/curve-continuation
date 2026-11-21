"""Curve continuation for reassembling painted fragments.

Pipeline: draw a test picture and cut it (synth), find the curve ends on every piece (extraction), score how well the
ends of two pieces continue each other (scoring), give the scores to the group's RL solver (solver_bridge), and check
the result (evaluation). pipeline.py ties the steps together; the scripts/ folder holds the command-line entry points.
"""

__version__ = "1.0.0"
