from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PPAllReduceFusionSupport(ABC):
    """Explicit model contract for PP-boundary all-reduce fusion state."""

    @abstractmethod
    def get_pp_allreduce_fusion_for_capture(
        self, forward_batch: ForwardBatch
    ) -> bool:
        raise NotImplementedError
