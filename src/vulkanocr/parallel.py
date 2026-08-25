"""Read one page with recognition work distributed across hardware GPUs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .device import hardware_devices
from .engine import OcrEngine, OcrEngineError, OcrModels, OcrResult, assemble_result
from .options import InferenceOptions
from .scheduling import CropScheduler
from .workers import MultiprocessingWorkerFleet, WorkerFleet


class PrimaryEngine(Protocol):
    @property
    def device_name(self) -> str: ...

    def crops(self, rgb: np.ndarray) -> list[tuple]: ...

    def recognise(self, patch: np.ndarray) -> tuple[str, float]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _ParallelComponents:
    options: InferenceOptions
    primary: PrimaryEngine
    fleet: WorkerFleet | None


def _wire_components(
    models: OcrModels,
    *,
    use_fp16: bool | None,
    options: InferenceOptions | None,
    runtime: Any,
    device_provider: Callable[[Any], tuple] | None,
    engine_factory: Callable[..., PrimaryEngine] | None,
    fleet_factory: Callable[..., WorkerFleet] | None,
) -> _ParallelComponents:
    """Concrete construction kept outside OCR orchestration."""
    if runtime is None:
        try:
            import ncnn as runtime  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover
            raise OcrEngineError("runtime-missing", "ncnn is not installed") from error
    if options is not None and use_fp16 is not None:
        raise OcrEngineError("options-conflict", "options cannot be combined with use_fp16")
    resolved = options or (InferenceOptions.fp16() if use_fp16 else InferenceOptions())
    devices = tuple((device_provider or hardware_devices)(runtime))
    build_engine = engine_factory or OcrEngine
    build_fleet = fleet_factory or MultiprocessingWorkerFleet
    primary_nets = ("det", "rec") if len(devices) == 1 else ("det",)
    primary = build_engine(
        models,
        runtime=runtime,
        options=resolved,
        device=devices[0],
        nets=primary_nets,
    )
    if len(devices) == 1:
        return _ParallelComponents(resolved, primary, None)
    try:
        fleet = build_fleet(models, devices, resolved)
    except BaseException:
        primary.close()
        raise
    return _ParallelComponents(resolved, primary, fleet)


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
        engine_factory: Callable[..., PrimaryEngine] | None = None,
        fleet_factory: Callable[..., WorkerFleet] | None = None,
    ):
        components = _wire_components(
            models,
            use_fp16=use_fp16,
            options=options,
            runtime=runtime,
            device_provider=device_provider,
            engine_factory=engine_factory,
            fleet_factory=fleet_factory,
        )
        self._options = components.options
        self._primary = components.primary
        self._fleet = components.fleet
        self._generation = 0
        self._closed = False

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
        if self._fleet is None:
            return assemble_result(
                self._primary.device_name,
                ((region, self._primary.recognise(patch)) for region, patch in pairs),
            )
        if not pairs:
            return assemble_result(self.device_name, ())
        if self._fleet.count == 0:
            raise OcrEngineError("worker-unavailable", "every recognition worker is unavailable")
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


__all__ = ["ParallelOcr", "PrimaryEngine"]
