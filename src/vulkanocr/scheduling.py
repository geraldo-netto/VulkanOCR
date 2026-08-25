"""Pure crop scheduling for asymmetric recognition devices."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CropAssignment:
    device_index: int
    crop_index: int


class CropScheduler:
    """Price crop assignments without queues, processes, ncnn, or OCR values."""

    def __init__(self, costs: dict[int, float], widths: list[int]):
        if not costs:
            raise ValueError("at least one device cost is required")
        if any(cost <= 0 for cost in costs.values()):
            raise ValueError("device costs must be positive")
        if any(width < 1 for width in widths):
            raise ValueError("crop widths must be positive")
        self._cost = dict(costs)
        self._widths = tuple(widths)
        self._work = deque(sorted(range(len(widths)), key=lambda index: -widths[index]))
        self._committed = dict.fromkeys(costs, 0.0)
        self._busy: dict[int, int] = {}

    @property
    def costs(self) -> dict[int, float]:
        return dict(self._cost)

    def assignments(self) -> tuple[CropAssignment, ...]:
        """Assign at most one crop to every idle device."""
        fastest = min(self._cost.values())
        assignments = []
        for device_index in sorted(self._cost, key=lambda index: self._cost[index]):
            if device_index in self._busy or not self._work:
                continue
            price = self._cost[device_index]
            if price <= fastest * 1.5:
                crop_index = self._work.popleft()
            else:
                crop_index = self._work[-1]
                remaining = sum(self._widths[index] for index in self._work)
                projected = self._committed[device_index] + price * self._widths[crop_index]
                if projected > fastest * remaining:
                    continue
                self._work.pop()
            self._committed[device_index] += price * self._widths[crop_index]
            self._busy[device_index] = crop_index
            assignments.append(CropAssignment(device_index, crop_index))
        return tuple(assignments)

    def complete(self, device_index: int, crop_index: int, elapsed_ms: float) -> None:
        """Release one device and refine its price from completed real work."""
        expected = self._busy.get(device_index)
        if expected != crop_index:
            raise ValueError(
                f"device {device_index} completed crop {crop_index}; expected {expected}"
            )
        del self._busy[device_index]
        columns = self._widths[crop_index]
        self._cost[device_index] = (self._cost[device_index] + elapsed_ms / columns) / 2


__all__ = ["CropAssignment", "CropScheduler"]
