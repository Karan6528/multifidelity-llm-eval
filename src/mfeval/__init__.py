"""Multi-fidelity evaluation for an in-car LLM assistant.

The idea: running an expensive LLM judge on every test is slow, while cheap checks
miss subtle problems. This package scores each test at three levels of cost and then
asks which mix of levels catches the most failures for the least compute.
"""

__version__ = "0.1.0"
