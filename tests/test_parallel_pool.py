"""GPU-free tests of the pool's reply handling and dispatch logic."""

from queue import Queue
from types import SimpleNamespace

import pytest

from vulkanocr.engine import OcrEngineError
from vulkanocr.parallel import ParallelOcr


def _pool(*, replies: Queue, devices, workers) -> ParallelOcr:
    pool = ParallelOcr.__new__(ParallelOcr)
    pool._replies = replies
    pool._devices = devices
    pool._workers = workers
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
        workers=[SimpleNamespace(is_alive=lambda: True)] * 2,
    )

    with pytest.raises(OcrEngineError) as refusal:
        pool._answer()
    assert refusal.value.code == "worker-failed"
    assert "iGPU" in str(refusal.value)
    assert "image-invalid" in str(refusal.value)


def test_a_dead_worker_is_still_a_named_refusal():
    replies = Queue()
    pool = _pool(
        replies=replies,
        devices=[SimpleNamespace(index=0, name="Fast GPU")],
        workers=[SimpleNamespace(is_alive=lambda: False, exitcode=-9)],
    )

    with pytest.raises(OcrEngineError) as refusal:
        pool._answer()
    assert refusal.value.code == "worker-died"
    assert "Fast GPU" in str(refusal.value)
    assert "-9" in str(refusal.value)
