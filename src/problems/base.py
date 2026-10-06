"""Contract between problem implementations and the shared pipeline."""
from __future__ import annotations

from typing import Protocol
import numpy as np
from ..core.representation import PreparedCompression


class ProblemAdapter(Protocol):
    def compress(self, problem: object, seed: int = 0) -> PreparedCompression:
        """Return ordered paths, source groups and recovery data."""
        ...

    def optimize(self, problem: object, compression: object,
                 routes: list[np.ndarray], seed: int = 0) -> np.ndarray | list[np.ndarray]:
        """Refine restored routes without changing the public output format."""
        ...
