"""Pure scheduling and repricing, without queues, processes, or OCR objects."""

import pytest

from vulkanocr.scheduling import CropAssignment, CropScheduler


def test_slow_device_gets_no_crop_the_fast_device_finishes_sooner():
    scheduler = CropScheduler({0: 1.0, 1: 10.0}, [300, 200, 100])

    assert scheduler.assignments() == (CropAssignment(0, 0),)
    scheduler.complete(0, 0, 300.0)
    assert scheduler.assignments() == (CropAssignment(0, 1),)
    scheduler.complete(0, 1, 200.0)
    assert scheduler.assignments() == (CropAssignment(0, 2),)


def test_near_equal_devices_split_expensive_work():
    scheduler = CropScheduler({0: 1.0, 1: 1.2}, [300, 200, 100])

    assert scheduler.assignments() == (CropAssignment(0, 0), CropAssignment(1, 1))


def test_real_completion_refines_probe_price():
    scheduler = CropScheduler({0: 1.0}, [100])
    assert scheduler.assignments() == (CropAssignment(0, 0),)

    scheduler.complete(0, 0, 400.0)

    assert scheduler.costs == {0: pytest.approx(2.5)}


def test_completion_must_match_the_device_assignment():
    scheduler = CropScheduler({0: 1.0}, [100])
    scheduler.assignments()
    with pytest.raises(ValueError, match="expected 0"):
        scheduler.complete(0, 1, 100.0)
