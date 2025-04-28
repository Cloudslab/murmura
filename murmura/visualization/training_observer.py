from abc import ABC, abstractmethod
from typing import List

from murmura.visualization.training_event import TrainingEvent


class TrainingObserver(ABC):
    """Interface for observers that react to training events."""

    @abstractmethod
    def on_event(self, event: TrainingEvent) -> None:
        """Handle a training event.

        Args:
            event (TrainingEvent): The training event to handle.
        """
        pass


class TrainingMonitor:
    """
    Central monitoring system that tracks the training process.

    This class collects events from the learning process and distributes them to registered observers.
    """

    def __init__(self):
        """Initialize the Training Monitor."""
        self.observers: List[TrainingObserver] = []
        self.events: List[TrainingEvent] = []

    def register_observer(self, observer: TrainingObserver) -> None:
        """
        Register an observer to receive training events.

        :param observer: The observer to register.
        """
        self.observers.append(observer)

    def emit_event(self, event: TrainingEvent) -> None:
        """
        Emit a training event to all registered observers.

        :param event: The training event to emit.
        """
        self.events.append(event)
        for observer in self.observers:
            observer.on_event(event)
