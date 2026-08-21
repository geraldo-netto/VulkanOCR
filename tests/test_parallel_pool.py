"""GPU-free tests of the pool's reply handling and dispatch logic."""

from queue import Queue
from types import SimpleNamespace

import pytest

from vulkanocr.engine import OcrEngineError
from vulkanocr.parallel import ParallelOcr


class _Worker:
    def __init__(self, *, alive: bool, exitcode: int | None = None):
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout=None):
        self._alive = False

    def terminate(self):
        self._alive = False


class _Channel:
    def __init__(self):
        self.sent = []
        self.closed = False

    def put(self, message):
        self.sent.append(message)

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        return None


def _pool(*, replies: Queue, devices, workers) -> ParallelOcr:
    pool = ParallelOcr.__new__(ParallelOcr)
    pool._replies = replies
    pool._devices = list(devices)
    pool._workers = list(workers)
    pool._names = {device.index: device.name for device in devices}
    pool._cost = dict.fromkeys(pool._names, 1.0)
    pool._requests = {device.index: _Channel() for device in devices}
    pool._generation = 0
    return pool


def test_a_workers_own_raise_is_reported_with_its_device_and_cause():
    """The child's exception text crosses the queue (VOCR-0037).

    An unhandled raise used to kill the child and the parent could only
    say "exited with code 1", the cause stranded on the child's stderr.
    """

    replies = Queue()
    replies.put(("error", 1, "OcrEngineError('image-invalid', 'expected uint8 pixels')"))
    pool = _pool(
        replies=replies,
        devices=[SimpleNamespace(index=0, name="Fast GPU"), SimpleNamespace(index=1, name="iGPU")],
        workers=[_Worker(alive=True), _Worker(alive=True)],
    )

    with pytest.raises(OcrEngineError) as refusal:
        pool._answer()
    assert refusal.value.code == "worker-failed"
    assert "iGPU" in str(refusal.value)
    assert "image-invalid" in str(refusal.value)
    # The failing worker is retired from dispatch (VOCR-0036): left in
    # place, it kept being assigned crops nobody would ever answer.
    assert 1 not in pool._cost
    assert 1 not in pool._requests
    assert [device.index for device in pool._devices] == [0]


def test_a_dead_worker_is_still_a_named_refusal():
    replies = Queue()
    pool = _pool(
        replies=replies,
        devices=[SimpleNamespace(index=0, name="Fast GPU")],
        workers=[_Worker(alive=False, exitcode=-9)],
    )

    with pytest.raises(OcrEngineError) as refusal:
        pool._answer()
    assert refusal.value.code == "worker-died"
    assert "Fast GPU" in str(refusal.value)
    assert "-9" in str(refusal.value)
    assert pool._cost == {}
    assert pool._workers == []


def test_a_stale_reply_from_a_failed_read_cannot_poison_the_next_page():
    """Replies carry their read's generation and stale ones drain (VOCR-0036).

    A read that raised mid-page left workers finishing crops whose "done"
    replies nobody consumed; the next read took them first and paired the
    previous page's crop_index with the new page's regions.
    """

    import numpy as np

    from vulkanocr.detection import TextRegion

    def region(y: float) -> TextRegion:
        return TextRegion(
            center_x=10.0,
            center_y=y,
            width=10.0,
            height=90.0,
            angle=90.0,
            vertical=False,
            score=0.9,
        )

    pairs = [
        (region(10.0), np.zeros((48, 300, 3), dtype=np.uint8)),
        (region(20.0), np.zeros((48, 200, 3), dtype=np.uint8)),
        (region(30.0), np.zeros((48, 100, 3), dtype=np.uint8)),
    ]
    replies = Queue()
    # A leftover from the failed first read: same device, old generation.
    replies.put(("done", 0, 1, 0, ("OLD PAGE", 0.9)))
    replies.put(("done", 0, 2, 0, ("first", 0.9)))
    replies.put(("done", 1, 2, 1, ("second", 0.9)))
    replies.put(("done", 0, 2, 2, ("third", 0.9)))
    pool = _pool(
        replies=replies,
        devices=[SimpleNamespace(index=0, name="Fast GPU"), SimpleNamespace(index=1, name="iGPU")],
        workers=[_Worker(alive=True), _Worker(alive=True)],
    )
    pool._generation = 1
    pool._primary = SimpleNamespace(crops=lambda _rgb: pairs, device_name="Fast GPU")

    result = pool.read(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [line.text for line in result.lines] == ["first", "second", "third"]
    # Every request that left carried the new read's generation.
    sent = [message for channel in pool._requests.values() for message in channel.sent]
    assert {message[0] for message in sent} == {2}
    assert sorted(message[1] for message in sent) == [0, 1, 2]
