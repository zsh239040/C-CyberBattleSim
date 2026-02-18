from abc import abstractmethod
from typing import List


class IEnvironmentObserver:
    """Interface for classes that want to subscribe to environment events."""

    @abstractmethod
    def on_reset(self, last_reward: float):
        """Called when environment is reset."""
        raise NotImplementedError


class EnvironmentEventSource:
    """Source of environment events that observers can attach themselves to."""

    def __init__(self) -> None:
        self.observers: List[IEnvironmentObserver] = []

    def add_observer(self, observer: IEnvironmentObserver) -> None:
        self.observers.append(observer)

    def notify_reset(self, last_reward: float) -> None:
        for observer in self.observers:
            observer.on_reset(last_reward)
