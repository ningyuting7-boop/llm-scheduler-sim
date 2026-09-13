"""Pluggable TIE scheduling for stock vLLM.

Loaded through vLLM's ``--scheduler-cls`` flag; nothing here patches or
forks vLLM itself. See docs/Phase3_Scheduling_Evaluation_Plan.md.

Parts of this package are adapted from the reference implementation
accompanying Zheng et al., "Scheduling LLM Inference with Uncertainty-Aware
Output Length Predictions" (ICML 2026, OpenReview I5IMkvVKd7), which is
distributed under Apache 2.0. Each module's docstring states what was taken
and what was changed.
"""
