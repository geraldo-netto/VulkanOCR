"""Reading one page with every hardware GPU the machine has.

Recognition dominates a read — ~600 of ~670 ms on the benchmark page — and is
embarrassingly parallel per crop, while the engine used one device however
many were present.

Two measured facts shaped this design, both recorded in
``benchmarks/multi_gpu.py``'s history rather than assumed:

* The ncnn Python binding **holds the GIL through ``extract``** (a pure-Python
  spin stalls for the full 225 ms of an integrated-GPU extract), so threads
  in one process serialise and a thread-based pool ran four times *slower*
  than one GPU. Each device therefore gets its own worker **process**.
* On an asymmetric pair (this desk's RX 6600 XT against a 610M is ~5x per
  column) a slow device must not take work the fast one would finish sooner.
  The dispatcher prices every assignment: a device receives a crop only while
  its cumulative committed time stays under what the fastest device would
  need for everything still queued. Near-equal devices split the page;
  unequal ones contribute a cheap tail crop or two and cannot hurt.

Correctness is device-count-independent: detection runs once on the preferred
device, every crop keeps its index, and lines return in ``OcrEngine.read``'s
order. Whether a second GPU helps is still a measurement — run
``benchmarks/multi_gpu.py`` on the machine in question.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from queue import Empty
from typing import Any

import numpy as np

from .device import hardware_devices
from .engine import OcrEngine, OcrLine, OcrModels, OcrResult

# One probe strip per worker, recognised twice at start-up: the second pass is
# the seed price (ms per pixel column) the dispatcher plans with before it has
# seen real crops from this device.
_PROBE_COLUMNS = 96


def _worker(device_index: int, models: OcrModels, use_fp16: bool, requests, replies) -> None:
    """One process, one device, one engine; runs until it receives ``None``."""
    import ncnn  # noqa: PLC0415 - imported in the child on purpose

    device = next(d for d in hardware_devices(ncnn) if d.index == device_index)
    engine = OcrEngine(models, runtime=ncnn, use_fp16=use_fp16, device=device)
    probe = np.full((48, _PROBE_COLUMNS, 3), 255, dtype=np.uint8)
    engine.recognise(probe)  # shader warm-up
    started = time.perf_counter()
    engine.recognise(probe)
    seed = (time.perf_counter() - started) * 1000 / _PROBE_COLUMNS
    replies.put(("ready", device_index, engine.device_name, seed))
    while True:
        message = requests.get()
        if message is None:
            engine.close()
            return
        index, patch = message
        replies.put(("done", device_index, index, engine.recognise(patch)))


class ParallelOcr:
    """One engine process per hardware device, work priced per assignment."""

    def __init__(self, models: OcrModels, *, use_fp16: bool = False):
        try:
            import ncnn  # noqa: PLC0415 - only to enumerate devices here
        except ImportError as error:  # pragma: no cover - environment boundary
            from .engine import OcrEngineError  # noqa: PLC0415

            raise OcrEngineError("runtime-missing", "ncnn is not installed") from error
        devices = hardware_devices(ncnn)
        self._devices = devices
        # Detection and single-device fallback stay in this process.
        self._primary = OcrEngine(models, runtime=ncnn, use_fp16=use_fp16, device=devices[0])
        self._context = mp.get_context("spawn")
        self._replies = self._context.Queue()
        self._requests: dict[int, Any] = {}
        self._names: dict[int, str] = {}
        self._cost: dict[int, float] = {}
        self._workers = []
        for device in devices:
            channel = self._context.Queue()
            process = self._context.Process(
                target=_worker,
                args=(device.index, models, use_fp16, channel, self._replies),
                daemon=True,
            )
            process.start()
            self._requests[device.index] = channel
            self._workers.append(process)
        for _ in devices:
            kind, index, name, seed = self._answer()
            assert kind == "ready"
            self._names[index] = name
            self._cost[index] = seed

    def _answer(self):
        """The next worker reply, or a refusal naming the worker that died.

        A bare `Queue.get()` here blocked forever when a worker was killed —
        by the OOM reaper, a driver reset, or a person — turning every later
        read into a hang. The wait now polls, and between polls checks that
        every process is still alive; a dead one is an error with a name, not
        an eternity.
        """
        from .engine import OcrEngineError  # noqa: PLC0415 - avoids a cycle at import

        while True:
            try:
                return self._replies.get(timeout=0.5)
            except Empty:
                for process, device in zip(self._workers, self._devices, strict=True):
                    if not process.is_alive():
                        raise OcrEngineError(
                            "worker-died",
                            f"the GPU worker for {device.name} exited with code {process.exitcode}",
                        ) from None

    @property
    def device_names(self) -> tuple[str, ...]:
        return tuple(self._names[index] for index in sorted(self._names))

    @property
    def device_name(self) -> str:
        """Every pooled device in one string, so callers of either engine
        can print where a read ran without caring which kind they hold."""
        return " + ".join(self.device_names)

    def read(self, rgb: np.ndarray) -> OcrResult:
        """Recognise every text line, crops shared across all devices."""
        pairs = self._primary.crops(rgb)
        if len(self._names) == 1 or len(pairs) < 2:
            return self._primary.read(rgb)

        work = deque(sorted(enumerate(pairs), key=lambda item: -item[1][1].shape[1]))
        results: list = [None] * len(pairs)
        committed = dict.fromkeys(self._cost, 0.0)
        busy: set[int] = set()
        outstanding = 0

        def assign() -> None:
            nonlocal outstanding
            fastest = min(self._cost.values())
            for index in sorted(self._cost, key=lambda i: self._cost[i]):
                if index in busy or not work:
                    continue
                price = self._cost[index]
                if price <= fastest * 1.5:
                    crop_index, (_region, patch) = work[0]
                    side = work.popleft
                else:
                    crop_index, (_region, patch) = work[-1]
                    remaining = sum(pair[1].shape[1] for _i, pair in work)
                    # A slow device is given a cheap crop only while its total
                    # commitment stays under the fast side's projected work.
                    if committed[index] + price * patch.shape[1] > fastest * remaining:
                        continue
                    side = work.pop
                side()
                committed[index] += price * patch.shape[1]
                self._requests[index].put((crop_index, patch))
                busy.add(index)
                outstanding += 1

        assign()
        while outstanding:
            kind, index, crop_index, answer = self._answer()
            assert kind == "done"
            outstanding -= 1
            busy.discard(index)
            region, _patch = pairs[crop_index]
            results[crop_index] = (region, answer)
            assign()

        return _assembled(results, " + ".join(self.device_names))

    def close(self) -> None:
        for channel in self._requests.values():
            # A dead worker's queue still accepts the sentinel; nothing to
            # guard beyond not caring whether anybody reads it.
            channel.put(None)
        for process in self._workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        self._primary.close()

    def __enter__(self) -> ParallelOcr:
        return self

    def __exit__(self, *_exception) -> None:
        self.close()


def _assembled(results: list, device_name: str) -> OcrResult:
    """The same assembly `OcrEngine.read` does, from indexed results."""
    lines = []
    undecoded = 0
    for region, (text, confidence) in results:
        if not text:
            undecoded += 1
            continue
        lines.append(
            OcrLine(
                text=text,
                confidence=confidence,
                box_score=region.score,
                center_x=region.center_x,
                center_y=region.center_y,
                thickness=region.width,
                length=region.height,
                angle=region.angle,
                vertical=region.vertical,
            )
        )
    lines.sort(key=lambda line: (line.center_y, line.center_x))
    return OcrResult(device_name=device_name, lines=tuple(lines), undecoded_regions=undecoded)


__all__ = ["ParallelOcr"]
