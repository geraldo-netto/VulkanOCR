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
_DEFAULT_READY_TIMEOUT_S = 30.0
_DEFAULT_RESPONSE_TIMEOUT_S = 120.0


def _timeout(error: tuple[str, str] | None) -> None:
    if error is None:
        raise RuntimeError("a message deadline requires a timeout error")
    raise OcrEngineError(*error)


def _message_wait(deadline: float | None, timeout_error: tuple[str, str] | None) -> float:
    if deadline is None:
        return 0.5
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _timeout(timeout_error)
    return min(0.5, remaining)


def _positive_timeout(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class WorkerAnswer:
    device_index: int
    generation: int
    crop_index: int
    result: tuple[str, float]
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class _InFlightJob:
    device_index: int
    deadline: float


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
        ready_timeout_s: float = _DEFAULT_READY_TIMEOUT_S,
        response_timeout_s: float = _DEFAULT_RESPONSE_TIMEOUT_S,
    ):
        if not devices:
            raise ValueError("a worker fleet requires at least one device")
        ready_timeout_s = _positive_timeout("ready_timeout_s", ready_timeout_s)
        response_timeout_s = _positive_timeout("response_timeout_s", response_timeout_s)
        self._context = context or mp.get_context("spawn")
        self._replies = self._context.Queue()
        self._devices = {device.index: device for device in devices}
        self._requests: dict[int, Any] = {}
        self._processes: dict[int, Any] = {}
        self._names: dict[int, str] = {}
        self._costs: dict[int, float] = {}
        self._inflight: dict[tuple[int, int], _InFlightJob] = {}
        self._response_timeout_s = response_timeout_s
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
            ready_deadline = time.monotonic() + ready_timeout_s
            for _ in devices:
                kind, index, name, seed = self._next_message(
                    deadline=ready_deadline,
                    timeout_error=(
                        "worker-start-timeout",
                        f"workers did not become ready within {ready_timeout_s:g} seconds",
                    ),
                )
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
        self._inflight[generation, crop_index] = _InFlightJob(
            device_index,
            time.monotonic() + self._response_timeout_s,
        )

    def answer(self) -> WorkerAnswer:
        deadline = min((job.deadline for job in self._inflight.values()), default=None)
        try:
            message = self._next_message(
                deadline=deadline,
                timeout_error=(
                    "worker-response-timeout",
                    f"recognition did not finish within {self._response_timeout_s:g} seconds",
                ),
            )
        except OcrEngineError as error:
            if error.code == "worker-response-timeout":
                self._retire_expired_jobs()
            raise
        if message[0] != "done":
            raise OcrEngineError("worker-protocol", f"expected done, received {message[0]}")
        _kind, index, generation, crop_index, result, elapsed = message
        self._inflight.pop((generation, crop_index), None)
        return WorkerAnswer(index, generation, crop_index, result, elapsed)

    def update_cost(self, device_index: int, cost: float) -> None:
        if device_index in self._costs:
            self._costs[device_index] = cost

    def _next_message(
        self,
        *,
        deadline: float | None = None,
        timeout_error: tuple[str, str] | None = None,
    ):
        while True:
            try:
                message = self._replies.get(timeout=_message_wait(deadline, timeout_error))
            except Empty:
                self._raise_if_worker_died()
                if deadline is not None and time.monotonic() >= deadline:
                    _timeout(timeout_error)
                continue
            if message[0] == "error":
                self._raise_worker_error(message)
            return message

    def _raise_if_worker_died(self) -> None:
        for index, process in tuple(self._processes.items()):
            if process.is_alive():
                continue
            device = self._devices[index]
            exitcode = process.exitcode
            self._retire(index)
            raise OcrEngineError(
                "worker-died",
                f"the GPU worker for {device.name} exited with code {exitcode}",
            ) from None

    def _raise_worker_error(self, message) -> None:
        _kind, index, detail = message
        device = self._devices[index]
        self._retire(index)
        raise OcrEngineError("worker-failed", f"the GPU worker for {device.name} raised {detail}")

    def _retire_expired_jobs(self) -> None:
        now = time.monotonic()
        expired = {job.device_index for job in self._inflight.values() if job.deadline <= now}
        for index in expired:
            self._retire(index, force=True)

    def _retire(self, index: int, *, force: bool = False) -> None:
        self._names.pop(index, None)
        self._costs.pop(index, None)
        self._inflight = {
            key: job for key, job in self._inflight.items() if job.device_index != index
        }
        channel = self._requests.pop(index, None)
        if channel is not None:
            channel.close()
            channel.cancel_join_thread()
        process = self._processes.pop(index, None)
        if process is not None:
            if force and process.is_alive():
                process.terminate()
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
