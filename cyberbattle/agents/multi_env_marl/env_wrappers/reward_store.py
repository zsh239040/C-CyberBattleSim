from abc import abstractmethod
from typing import List


class IRewardStore:
    """Stores the rewards given by an environment during a single episode."""

    @property
    @abstractmethod
    def episode_rewards(self) -> List[float]:
        raise NotImplementedError
