"""Executor interface and implementations for running experiments."""

from folio.executors.base import Executor
from folio.executors.claude_light import ClaudeLightExecutor
from folio.executors.human import HumanExecutor

__all__ = ["Executor", "HumanExecutor", "ClaudeLightExecutor"]
