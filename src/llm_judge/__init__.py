"""wassname's small, composable defaults for LLM-as-judge. -- PI/OpenAI"""
from .pairs import compare
from .questions import label
from .trace import Trace

__all__ = ["Trace", "compare", "label"]
