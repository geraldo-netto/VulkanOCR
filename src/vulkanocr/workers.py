"""Recognition worker fleet: process, queue, handshake, and shutdown adapter."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any, Protocol

import numpy as np

from .device import VulkanDevice, hardware_devices
from .engine import OcrEngine, OcrEngineError, OcrModels
from .options import InferenceOptions

_PROBE_COLUMNS = 96


@dataclass(frozen=True, slots=True)
class WorkerAnswer:
    device_index: int
    generation: int
    crop_index: int
    result: tuple[str, float]
    elapsed_ms: float


class WorkerFleet(Protocol):
    @property
    def count(self) -> int: ...

    @property
    def device_names(self) -> tuple[str, ...]: ...

    @property
    def costs(self) -> dict[int, float]: ...

    def send(
        self,
        device_index: int,
        generation: int,
        crop_index: int,
        patch: np.ndarray,
    ) -> None: ...

    def answer(self) -> WorkerAnswer: ...

    def update_cost(self, device_index: int, cost: float) -> None: ...

    def close(self) -> None: ...


def worker_main(
    device_index: int,
    models: OcrModels,
    options: InferenceOptions,
    requests,
    replies,
) -> None:
    """One process, one device, one recognition engine."""
    try:
        import ncnn  # noqa: PLC0415 - imported in the child on purpose

        device = next(d for d in hardware_devices(ncnn) if d.index == device_index)
        engine = OcrEngine(models, runtime=ncnn, options=options, device=device, nets=("rec",))
    except BaseException as error:  # noqa: BLE001 - the reply is the report
        replies.put(("error", device_index, repr(error)))
        raise
    try:
        probe = np.full((48, _PROBE_COLUMNS, 3), 255, dtype=np.uint8)
        engine.recognise(probe)
        started = time.perf_counter()
        engine.recognise(probe)
        seed = (time.perf_counter() - started) * 1000 / _PROBE_COLUMNS
        replies.put(("ready", device_index, engine.device_name, seed))
        while True:
            message = requests.get()
            if message is None:
                return
            generation, crop_index, patch = message
            started = time.perf_counter()
            answer = engine.recognise(patch)
            elapsed = (time.perf_counter() - started) * 1000
            replies.put(("done", device_index, generation, crop_index, answer, elapsed))
    except BaseException as error:  # noqa: BLE001 - the reply is the report
        replies.put(("error", device_index, repr(error)))
        raise
    finally:
        engine.close()


class MultiprocessingWorkerFleet:
    """One recognition process and request queue per hardware device."""

    def __init__(
        self,
        models: OcrModels,
        devices: tuple[VulkanDevice, ...],
        options: InferenceOptions,
        *,
        context: Any = None,
        target=worker_main,
    ):
        if not devices:
            raise ValueError("a worker fleet requires at least one device")
        self._context = context or mp.get_context("spawn")
        self._replies = self._context.Queue()
        self._devices = {device.index: device for device in devices}
        self._requests: dict[int, Any] = {}
        self._processes: dict[int, Any] = {}
        self._names: dict[int, str] = {}
        self._costs: dict[int, float] = {}
        self._closed = False
        try:
            for device in devices:
                channel = self._context.Queue()
                process = self._context.Process(
                    target=target,
                    args=(device.index, models, options, channel, self._replies),
                    daemon=True,
                )
                try:
                    process.start()
                except BaseException:
                    channel.close()
                    channel.cancel_join_thread()
                    raise
                self._requests[device.index] = channel
                self._processes[device.index] = process
            for _ in devices:
                kind, index, name, seed = self._next_message()
                if kind != "ready":
                    raise OcrEngineError("worker-protocol", f"expected ready, received {kind}")
                self._names[index] = name
                self._costs[index] = seed
        except BaseException:
            self.close()
            raise

    @property
    def count(self) -> int:
        return len(self._names)

    @property
    def device_names(self) -> tuple[str, ...]:
        return tuple(self._names[index] for index in sorted(self._names))

    @property
    def costs(self) -> dict[int, float]:
        return dict(self._costs)

    def send(self, device_index: int, generation: int, crop_index: int, patch: np.ndarray) -> None:
        try:
            channel = self._requests[device_index]
        except KeyError:
            raise OcrEngineError(
                "worker-unavailable", f"worker {device_index} is unavailable"
            ) from None
        channel.put((generation, crop_index, patch))

    def answer(self) -> WorkerAnswer:
        message = self._next_message()
        if message[0] != "done":
            raise OcrEngineError("worker-protocol", f"expected done, received {message[0]}")
        _kind, index, generation, crop_index, result, elapsed = message
        return WorkerAnswer(index, generation, crop_index, result, elapsed)

    def update_cost(self, device_index: int, cost: float) -> None:
        if device_index in self._costs:
            self._costs[device_index] = cost

    def _next_message(self):
        while True:
            try:
                message = self._replies.get(timeout=0.5)
            except Empty:
                for index, process in tuple(self._processes.items()):
                    if not process.is_alive():
                        device = self._devices[index]
                        exitcode = process.exitcode
                        self._retire(index)
                        raise OcrEngineError(
                            "worker-died",
                            f"the GPU worker for {device.name} exited with code {exitcode}",
                        ) from None
                continue
            if message[0] == "error":
                _kind, index, detail = message
                device = self._devices[index]
                self._retire(index)
                raise OcrEngineError(
                    "worker-failed", f"the GPU worker for {device.name} raised {detail}"
                )
            return message

    def _retire(self, index: int) -> None:
        self._names.pop(index, None)
        self._costs.pop(index, None)
        channel = self._requests.pop(index, None)
        if channel is not None:
            channel.close()
            channel.cancel_join_thread()
        process = self._processes.pop(index, None)
        if process is not None:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for channel in self._requests.values():
            channel.put(None)
        for process in self._processes.values():
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        for channel in self._requests.values():
            channel.close()
            channel.cancel_join_thread()
        self._replies.close()
        self._replies.cancel_join_thread()


__all__ = ["MultiprocessingWorkerFleet", "WorkerAnswer", "WorkerFleet", "worker_main"]
