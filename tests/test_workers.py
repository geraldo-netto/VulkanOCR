"""Worker-process adapter behavior without a GPU or child process."""

from queue import Queue
from types import SimpleNamespace

import pytest

from vulkanocr.engine import OcrEngineError
from vulkanocr.workers import MultiprocessingWorkerFleet


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
