"""Compatibility of two pieces from their curve ends.

ends.py           the End data class and loading the _ends.json files
geometry.py       cost of joining two ends: biarc, co-circularity, clothoid
features.py       the full join cost with width, colour and curvature terms
matching.py       from join scores to one score per placement: best, lap, order
significance.py   the a contrario chance model and the number of false alarms (NFA)
compatibility.py  the score S of one placement, and the matrix R for the solver
"""
