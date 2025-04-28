import pytest

from murmura.visualization.training_event import TrainingEvent, EvaluationEvent
from murmura.visualization.training_observer import TrainingMonitor, TrainingObserver


class MockObserver(TrainingObserver):
    """Mock observer for testing"""
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def test_training_monitor_initialization():
    """Test initializing the TrainingMonitor"""
    monitor = TrainingMonitor()

    assert monitor.observers == []
    assert monitor.events == []


def test_register_observer():
    """Test registering an observer with the monitor"""
    monitor = TrainingMonitor()
    observer = MockObserver()

    monitor.register_observer(observer)

    assert observer in monitor.observers
    assert len(monitor.observers) == 1


def test_emit_event():
    """Test emitting an event to registered observers"""
    monitor = TrainingMonitor()
    observer1 = MockObserver()
    observer2 = MockObserver()

    monitor.register_observer(observer1)
    monitor.register_observer(observer2)

    event = TrainingEvent(round_num=1, step_name="test")
    monitor.emit_event(event)

    # Event should be stored in the monitor
    assert event in monitor.events
    assert len(monitor.events) == 1

    # Event should be received by both observers
    assert event in observer1.events
    assert event in observer2.events
    assert len(observer1.events) == 1
    assert len(observer2.events) == 1


def test_multiple_events():
    """Test emitting multiple events"""
    monitor = TrainingMonitor()
    observer = MockObserver()

    monitor.register_observer(observer)

    event1 = TrainingEvent(round_num=1, step_name="step1")
    event2 = TrainingEvent(round_num=2, step_name="step2")

    monitor.emit_event(event1)
    monitor.emit_event(event2)

    assert len(monitor.events) == 2
    assert len(observer.events) == 2
    assert observer.events[0] == event1
    assert observer.events[1] == event2


def test_register_multiple_observers():
    """Test registering multiple observers"""
    monitor = TrainingMonitor()
    observers = [MockObserver() for _ in range(5)]

    for observer in observers:
        monitor.register_observer(observer)

    assert len(monitor.observers) == 5

    # Emit an event
    event = EvaluationEvent(round_num=1, metrics={"accuracy": 0.9})
    monitor.emit_event(event)

    # All observers should receive the event
    for observer in observers:
        assert len(observer.events) == 1
        assert observer.events[0] == event


def test_abstract_observer_implementation():
    """Test that implementing TrainingObserver requires on_event method"""
    # Creating a direct instance should fail since it's an abstract class
    with pytest.raises(TypeError):
        TrainingObserver()

    # Creating a subclass without implementing on_event should fail
    with pytest.raises(TypeError):
        class InvalidObserver(TrainingObserver):
            pass

        InvalidObserver()

    # Creating a valid subclass should work
    class ValidObserver(TrainingObserver):
        def on_event(self, event):
            pass

    observer = ValidObserver()
    assert isinstance(observer, TrainingObserver)
