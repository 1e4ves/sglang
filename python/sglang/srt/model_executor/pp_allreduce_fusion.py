from __future__ import annotations

from abc import ABC, abstractmethod


class PPAllReduceFusionSupport(ABC):
    """Explicit model contract for PP-boundary all-reduce fusion state."""

    @abstractmethod
    def get_pp_allreduce_fusion(
        self, num_tokens: int, can_run_tbo: bool
    ) -> bool:
        raise NotImplementedError
