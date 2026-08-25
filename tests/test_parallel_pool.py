"""GPU-free orchestration tests over the worker-fleet protocol."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from vulkanocr.detection import TextRegion
from vulkanocr.engine import OcrModels
from vulkanocr.parallel import ParallelOcr, PrimaryEngine
from vulkanocr.workers import WorkerAnswer, WorkerFleet


def _region(y: float) -> TextRegion:
    return TextRegion(10.0, y, 10.0, 90.0, 90.0, False, 0.9)


class FakeFleet:
    def __init__(self, costs, replies, names=None):
        self._costs = dict(costs)
        self._replies = list(replies)
        self._names = names or {index: f"GPU {index}" for index in costs}
        self.sent = {index: [] for index in costs}
        self.closed = False

    @property
    def count(self):
        return len(self._names)

    @property
    def device_names(self):
        return tuple(self._names[index] for index in sorted(self._names))

    @property
    def costs(self):
        return dict(self._costs)

    def send(self, device_index, generation, crop_index, patch):
        self.sent[device_index].append((generation, crop_index, patch))

    def answer(self):
        return self._replies.pop(0)

    def update_cost(self, device_index, cost):
        self._costs[device_index] = cost

    def close(self):
        self.closed = True


class FakePrimary:
    device_name = "Fast GPU"

    def __init__(self, pairs: list[tuple], result: tuple[str, float] = ("fallback", 0.9)):
        self.pairs = pairs
        self.result = result
        self.close_calls = 0

    def crops(self, rgb: np.ndarray) -> list[tuple]:
        del rgb
        return self.pairs

    def recognise(self, patch: np.ndarray) -> tuple[str, float]:
        del patch
        return self.result

    def close(self):
        self.close_calls += 1


def _pool(fleet: WorkerFleet | None, pairs):
    primary = FakePrimary(pairs)
    engines: list[PrimaryEngine] = []

    def build_engine(*_args, **_kwargs) -> PrimaryEngine:
        engine = primary if not engines else FakePrimary([], ("recovered", 0.9))
        engines.append(engine)
        return engine

    device_count = 1 if fleet is None else 2
    devices = tuple(
        SimpleNamespace(index=index, name=f"GPU {index}") for index in range(device_count)
    )
    pool = ParallelOcr(
        cast(OcrModels, object()),
        runtime=object(),
        device_provider=lambda _runtime: devices,
        engine_factory=build_engine,
        fleet_factory=None if fleet is None else lambda *_args: fleet,
    )
    return pool, primary, engines


def _pairs():
    return [
        (_region(10.0), np.zeros((48, 300, 3), dtype=np.uint8)),
        (_region(20.0), np.zeros((48, 200, 3), dtype=np.uint8)),
        (_region(30.0), np.zeros((48, 100, 3), dtype=np.uint8)),
    ]


def test_stale_reply_cannot_poison_the_next_page():
    replies = [
        WorkerAnswer(0, 1, 0, ("OLD PAGE", 0.9), 90.0),
        WorkerAnswer(0, 2, 0, ("first", 0.9), 300.0),
        WorkerAnswer(1, 2, 1, ("second", 0.9), 200.0),
        WorkerAnswer(0, 2, 2, ("third", 0.9), 100.0),
    ]
    fleet = FakeFleet({0: 1.0, 1: 1.0}, replies, {0: "Fast GPU", 1: "iGPU"})
    pool, primary, _engines = _pool(fleet, [])
    pool.read(np.zeros((1, 1, 3), dtype=np.uint8))
    primary.pairs = _pairs()

    result = pool.read(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [line.text for line in result.lines] == ["first", "second", "third"]
    sent = [message for messages in fleet.sent.values() for message in messages]
    assert {message[0] for message in sent} == {2}
    assert sorted(message[1] for message in sent) == [0, 1, 2]


def test_real_crops_refine_a_wrong_probe_seed():
    replies = [
        WorkerAnswer(0, 1, 0, ("one", 0.9), 1200.0),
        WorkerAnswer(1, 1, 1, ("two", 0.9), 200.0),
        WorkerAnswer(1, 1, 2, ("three", 0.9), 100.0),
    ]
    fleet = FakeFleet({0: 1.0, 1: 1.0}, replies)
    pool, _primary, _engines = _pool(fleet, _pairs())

    pool.read(np.zeros((100, 100, 3), dtype=np.uint8))

    assert fleet.costs == {0: pytest.approx(2.5), 1: pytest.approx(1.0)}


def test_slow_device_gets_no_work_fast_device_finishes_sooner():
    replies = [
        WorkerAnswer(0, 1, 0, ("first", 0.9), 300.0),
        WorkerAnswer(0, 1, 1, ("second", 0.9), 200.0),
        WorkerAnswer(0, 1, 2, ("third", 0.9), 100.0),
    ]
    fleet = FakeFleet({0: 1.0, 1: 10.0}, replies)
    pool, _primary, _engines = _pool(fleet, _pairs())

    pool.read(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [message[1] for message in fleet.sent[0]] == [0, 1, 2]
    assert fleet.sent[1] == []


def test_near_equal_devices_split_the_page():
    replies = [
        WorkerAnswer(0, 1, 0, ("first", 0.9), 300.0),
        WorkerAnswer(1, 1, 1, ("second", 0.9), 240.0),
        WorkerAnswer(0, 1, 2, ("third", 0.9), 100.0),
    ]
    fleet = FakeFleet({0: 1.0, 1: 1.2}, replies)
    pool, _primary, _engines = _pool(fleet, _pairs())

    pool.read(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [message[1] for message in fleet.sent[0]] == [0, 2]
    assert [message[1] for message in fleet.sent[1]] == [1]


def test_absent_fleet_reads_on_single_device_primary():
    pool, _primary, _engines = _pool(None, _pairs())

    result = pool.read(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result.device_name == "Fast GPU"
    assert [line.text for line in result.lines] == ["fallback"] * 3


def test_empty_multi_gpu_fleet_builds_and_reuses_lazy_recognition_fallback():
    pool, _primary, built = _pool(FakeFleet({}, [], {}), _pairs())

    first = pool.read(np.zeros((10, 10, 3), dtype=np.uint8))
    second = pool.read(np.zeros((10, 10, 3), dtype=np.uint8))

    assert [line.text for line in first.lines] == ["recovered"] * 3
    assert [line.text for line in second.lines] == ["recovered"] * 3
    assert len(built) == 2


def test_close_is_safe_to_repeat():
    fleet = FakeFleet({0: 1.0, 1: 1.0}, [])
    pool, primary, _engines = _pool(fleet, [])

    pool.close()
    pool.close()

    assert fleet.closed is True
    assert primary.close_calls == 1


def test_one_device_builds_no_worker_fleet(monkeypatch):
    from vulkanocr import parallel

    class FakeEngine:
        device_name = "Only GPU"

        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def crops(self, rgb: np.ndarray) -> list[tuple]:
            del rgb
            return [(_region(10.0), np.zeros((48, 100, 3), dtype=np.uint8))] * 2

        def recognise(self, patch: np.ndarray) -> tuple[str, float]:
            del patch
            return ("alone", 0.9)

        def close(self):
            self.closed = True

    monkeypatch.setattr(parallel, "OcrEngine", FakeEngine)
    monkeypatch.setattr(
        parallel,
        "hardware_devices",
        lambda _runtime: [SimpleNamespace(index=0, name="Only GPU")],
    )
    monkeypatch.setattr(
        parallel,
        "MultiprocessingWorkerFleet",
        lambda *_args: pytest.fail("single device must not build a fleet"),
    )

    pool = ParallelOcr(cast(OcrModels, object()))
    assert pool.device_name == "Only GPU"
    assert [line.text for line in pool.read(np.zeros((10, 10, 3), np.uint8)).lines] == [
        "alone",
        "alone",
    ]
    pool.close()


def test_runtime_devices_engine_and_fleet_are_injected_without_module_patches():
    runtime = object()
    models = cast(OcrModels, object())
    devices = (
        SimpleNamespace(index=0, name="Fast GPU"),
        SimpleNamespace(index=1, name="iGPU"),
    )
    built = []
    fleet = FakeFleet({0: 1.0, 1: 1.0}, [])

    class FakeEngine:
        device_name = "Fast GPU"

        def close(self):
            return None

        def crops(self, rgb: np.ndarray) -> list[tuple]:
            del rgb
            return []

        def recognise(self, patch: np.ndarray) -> tuple[str, float]:
            del patch
            return ("", 0.0)

    def build_engine(received_models, **kwargs) -> PrimaryEngine:
        built.append((received_models, kwargs))
        return FakeEngine()

    def build_fleet(received_models, received_devices, options):
        built.append((received_models, received_devices, options))
        return fleet

    pool = ParallelOcr(
        models,
        runtime=runtime,
        device_provider=lambda received: devices if received is runtime else (),
        engine_factory=build_engine,
        fleet_factory=build_fleet,
    )

    assert built[0][0] is models
    assert built[0][1]["runtime"] is runtime
    assert built[0][1]["device"] is devices[0]
    assert built[0][1]["nets"] == ("det",)
    assert built[1][0] is models and built[1][1] == devices
    options = built[1][2]
    pool._fallback_recognizer()
    assert built[2][0] is models
    assert built[2][1]["runtime"] is runtime
    assert built[2][1]["device"] is devices[0]
    assert built[2][1]["options"] is options
    assert built[2][1]["nets"] == ("rec",)


def test_detection_parent_and_lazy_fallback_release_every_loaded_net():
    devices = (
        SimpleNamespace(index=0, name="Fast GPU"),
        SimpleNamespace(index=1, name="iGPU"),
    )
    engines = []
    fleet = FakeFleet({}, [], {})

    class RecordingEngine:
        device_name = "Fast GPU"

        def __init__(self, *_args, nets, **_kwargs):
            self.nets = nets
            self.closed = False
            engines.append(self)

        def crops(self, rgb: np.ndarray) -> list[tuple]:
            del rgb
            return [(_region(10.0), np.zeros((48, 100, 3), dtype=np.uint8))]

        def recognise(self, patch: np.ndarray) -> tuple[str, float]:
            del patch
            return ("recovered", 0.9)

        def close(self):
            self.closed = True

    pool = ParallelOcr(
        cast(OcrModels, object()),
        runtime=object(),
        device_provider=lambda _runtime: devices,
        engine_factory=RecordingEngine,
        fleet_factory=lambda *_args: fleet,
    )

    first = pool.read(np.zeros((10, 10, 3), dtype=np.uint8))
    second = pool.read(np.zeros((10, 10, 3), dtype=np.uint8))
    pool.close()

    assert [line.text for line in first.lines] == ["recovered"]
    assert [line.text for line in second.lines] == ["recovered"]
    assert [engine.nets for engine in engines] == [("det",), ("rec",)]
    assert all(engine.closed for engine in engines)
    assert fleet.closed is True
