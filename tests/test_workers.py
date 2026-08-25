"""Worker-process adapter behavior without a GPU or child process."""

import sys
from queue import Queue
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from vulkanocr.device import VulkanDevice
from vulkanocr.engine import OcrEngineError, OcrModels
from vulkanocr.options import InferenceOptions
from vulkanocr.workers import MultiprocessingWorkerFleet, worker_main


class FakeProcess:
    def __init__(self, *, alive=True, exitcode=None):
        self._alive = alive
        self.exitcode = exitcode
        self.started = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self._alive = False

    def terminate(self):
        self._alive = False


class FakeQueue(Queue):
    def __init__(self):
        super().__init__()
        self.closed = False
        self.cancelled = False
        self.sent = []

    def put(self, item, *args, **kwargs):
        self.sent.append(item)
        return super().put(item, *args, **kwargs)

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        self.cancelled = True


class SilentContext:
    def __init__(self):
        self.queues = []
        self.processes = []
        self.Queue = self._queue
        self.Process = self._process

    def _queue(self):
        queue = FakeQueue()
        self.queues.append(queue)
        return queue

    def _process(self, **_kwargs):
        process = FakeProcess()
        self.processes.append(process)
        return process


def _fleet(replies, devices, processes):
    fleet = MultiprocessingWorkerFleet.__new__(MultiprocessingWorkerFleet)
    fleet._replies = replies
    fleet._devices = {device.index: device for device in devices}
    fleet._requests = {device.index: FakeQueue() for device in devices}
    fleet._processes = {
        device.index: process for device, process in zip(devices, processes, strict=True)
    }
    fleet._names = {device.index: device.name for device in devices}
    fleet._costs = dict.fromkeys(fleet._names, 1.0)
    fleet._inflight = {}
    fleet._response_timeout_s = 120.0
    fleet._closed = False
    return fleet


def test_worker_error_retires_device_and_preserves_cause():
    replies = FakeQueue()
    replies.put(("error", 1, "RuntimeError('driver reset')"))
    fleet = _fleet(
        replies,
        [SimpleNamespace(index=1, name="iGPU")],
        [FakeProcess(alive=True)],
    )

    with pytest.raises(OcrEngineError) as caught:
        fleet.answer()

    assert caught.value.code == "worker-failed"
    assert "iGPU" in caught.value.detail and "driver reset" in caught.value.detail
    assert fleet.count == 0


def test_dead_worker_is_a_named_refusal():
    fleet = _fleet(
        FakeQueue(),
        [SimpleNamespace(index=0, name="Fast GPU")],
        [FakeProcess(alive=False, exitcode=-9)],
    )

    with pytest.raises(OcrEngineError) as caught:
        fleet.answer()

    assert caught.value.code == "worker-died"
    assert "Fast GPU" in caught.value.detail and "-9" in caught.value.detail


def test_close_is_idempotent_and_reaps_every_process():
    processes = [FakeProcess(), FakeProcess()]
    fleet = _fleet(
        FakeQueue(),
        [SimpleNamespace(index=0, name="A"), SimpleNamespace(index=1, name="B")],
        processes,
    )
    channels = list(fleet._requests.values())

    fleet.close()
    fleet.close()

    assert [channel.sent for channel in channels] == [[None], [None]]
    assert all(not process.is_alive() for process in processes)
    assert all(channel.closed and channel.cancelled for channel in channels)


def test_each_worker_loads_and_closes_one_recognition_only_engine(monkeypatch):
    from vulkanocr import workers

    built = []

    class FakeEngine:
        device_name = "GPU"

        def __init__(self, *_args, **kwargs):
            self.nets = kwargs["nets"]
            self.closed = False
            built.append(self)

        def recognise(self, _patch):
            return ("", 0.0)

        def close(self):
            self.closed = True

    runtime = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "ncnn", runtime)
    monkeypatch.setattr(workers, "OcrEngine", FakeEngine)
    monkeypatch.setattr(
        workers,
        "hardware_devices",
        lambda _runtime: (
            SimpleNamespace(index=0, name="GPU 0"),
            SimpleNamespace(index=1, name="GPU 1"),
        ),
    )
    for index in (0, 1):
        requests, replies = Queue(), Queue()
        requests.put(None)
        worker_main(
            index,
            cast(OcrModels, object()),
            InferenceOptions(),
            requests,
            replies,
        )
        assert replies.get()[0] == "ready"

    assert [engine.nets for engine in built] == [("rec",), ("rec",)]
    assert all(engine.closed for engine in built)


def test_live_silent_worker_hits_configured_startup_deadline_and_closes_fleet():
    context = SilentContext()
    device = cast(VulkanDevice, SimpleNamespace(index=0, name="Silent GPU"))

    with pytest.raises(OcrEngineError) as caught:
        MultiprocessingWorkerFleet(
            cast(OcrModels, object()),
            (device,),
            InferenceOptions(),
            context=context,
            ready_timeout_s=0.01,
        )

    assert caught.value.code == "worker-start-timeout"
    assert "0.01 seconds" in caught.value.detail
    assert context.queues[1].sent == [None]
    assert all(queue.closed and queue.cancelled for queue in context.queues)
    assert all(not process.is_alive() for process in context.processes)


def test_live_silent_worker_hits_inference_response_deadline():
    fleet = _fleet(
        FakeQueue(),
        [SimpleNamespace(index=0, name="Silent GPU")],
        [FakeProcess(alive=True)],
    )
    fleet._response_timeout_s = 0.01
    fleet.send(0, 4, 7, np.zeros((48, 96, 3), dtype=np.uint8))

    with pytest.raises(OcrEngineError) as caught:
        fleet.answer()

    assert caught.value.code == "worker-response-timeout"
    assert "0.01 seconds" in caught.value.detail
    assert (4, 7) in fleet._inflight
