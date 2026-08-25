"""Read one page with recognition work distributed across hardware GPUs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .device import hardware_devices
from .engine import OcrEngine, OcrEngineError, OcrModels, OcrResult, assemble_result
from .options import InferenceOptions
from .scheduling import CropScheduler
from .workers import MultiprocessingWorkerFleet, WorkerFleet


class ParallelOcr:
    """Detect once, schedule recognition crops, assemble in reading order."""

    def __init__(
        self,
        models: OcrModels,
        *,
        use_fp16: bool | None = None,
        options: InferenceOptions | None = None,
        runtime: Any = None,
        device_provider: Callable[[Any], tuple] | None = None,
        engine_factory: Callable[..., Any] | None = None,
        fleet_factory: Callable[..., WorkerFleet] | None = None,
    ):
        if runtime is None:
            try:
                import ncnn as runtime  # type: ignore[no-redef]  # noqa: PLC0415
            except ImportError as error:  # pragma: no cover
                raise OcrEngineError("runtime-missing", "ncnn is not installed") from error
        if options is not None and use_fp16 is not None:
            raise OcrEngineError("options-conflict", "options cannot be combined with use_fp16")
        self._options = options or (InferenceOptions.fp16() if use_fp16 else InferenceOptions())
        devices = tuple((device_provider or hardware_devices)(runtime))
        build_engine = engine_factory or OcrEngine
        build_fleet = fleet_factory or MultiprocessingWorkerFleet
        self._primary = build_engine(
            models,
            runtime=runtime,
            options=self._options,
            device=devices[0],
        )
        self._fleet: WorkerFleet | None = None
        self._generation = 0
        self._closed = False
        if len(devices) == 1:
            return
        try:
            self._fleet = build_fleet(models, devices, self._options)
        except BaseException:
            self.close()
            raise

    @property
    def device_names(self) -> tuple[str, ...]:
        if self._fleet is None:
            return (self._primary.device_name,)
        return self._fleet.device_names

    @property
    def device_name(self) -> str:
        return " + ".join(self.device_names)

    def read(self, rgb: np.ndarray) -> OcrResult:
        """Recognise every text line, crops shared across active workers."""
        self._generation += 1
        generation = self._generation
        pairs = self._primary.crops(rgb)
        if self._fleet is None or self._fleet.count <= 1 or len(pairs) < 2:
            return assemble_result(
                self._primary.device_name,
                ((region, self._primary.recognise(patch)) for region, patch in pairs),
            )
        return assemble_result(self.device_name, self._dispatch(generation, pairs))

    def _dispatch(self, generation: int, pairs: list[tuple]) -> list:
        fleet = self._fleet
        if fleet is None:
            raise OcrEngineError("worker-unavailable", "the recognition worker fleet is absent")
        results: list[Any] = [None] * len(pairs)
        scheduler = CropScheduler(fleet.costs, [pair[1].shape[1] for pair in pairs])
        outstanding = 0

        def assign() -> None:
            nonlocal outstanding
            for assignment in scheduler.assignments():
                patch = pairs[assignment.crop_index][1]
                fleet.send(
                    assignment.device_index,
                    generation,
                    assignment.crop_index,
                    patch,
                )
                outstanding += 1

        assign()
        while outstanding:
            reply = fleet.answer()
            if reply.generation != generation:
                continue
            outstanding -= 1
            region, _patch = pairs[reply.crop_index]
            scheduler.complete(reply.device_index, reply.crop_index, reply.elapsed_ms)
            fleet.update_cost(reply.device_index, scheduler.costs[reply.device_index])
            results[reply.crop_index] = (region, reply.result)
            assign()
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fleet is not None:
            self._fleet.close()
        self._primary.close()

    def __enter__(self) -> ParallelOcr:
        return self

    def __exit__(self, *_exception) -> None:
        self.close()


__all__ = ["ParallelOcr"]
