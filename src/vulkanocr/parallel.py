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
from .engine import OcrEngine, OcrEngineError, OcrModels, OcrResult, assemble_result

# One probe strip per worker, recognised twice at start-up: the second pass is
# the seed price (ms per pixel column) the dispatcher plans with before it has
# seen real crops from this device.
_PROBE_COLUMNS = 96


def _worker(device_index: int, models: OcrModels, use_fp16: bool, requests, replies) -> None:
    """One process, one device, one engine; runs until it receives ``None``.

    Anything that raises is reported as an ``("error", ...)`` reply before the
    process dies: an unhandled raise killed the child and the parent could
    only say "exited with code 1", with the actual cause stranded on the
    child's inherited stderr (VOCR-0037).
    """
    try:
        import ncnn  # noqa: PLC0415 - imported in the child on purpose

        device = next(d for d in hardware_devices(ncnn) if d.index == device_index)
        # Recognition only: the worker never detects, and the detection
        # net's Vulkan allocations were dead weight on every device (VOCR-0042).
        engine = OcrEngine(models, runtime=ncnn, use_fp16=use_fp16, device=device, nets=("rec",))
    except BaseException as error:  # noqa: BLE001 - the reply is the report
        replies.put(("error", device_index, repr(error)))
        raise
    try:
        probe = np.full((48, _PROBE_COLUMNS, 3), 255, dtype=np.uint8)
        engine.recognise(probe)  # shader warm-up
        started = time.perf_counter()
        engine.recognise(probe)
        seed = (time.perf_counter() - started) * 1000 / _PROBE_COLUMNS
        replies.put(("ready", device_index, engine.device_name, seed))
        while True:
            message = requests.get()
            if message is None:
                return
            generation, index, patch = message
            started = time.perf_counter()
            answer = engine.recognise(patch)
            elapsed = (time.perf_counter() - started) * 1000
            replies.put(("done", device_index, generation, index, answer, elapsed))
    except BaseException as error:  # noqa: BLE001 - the reply is the report
        replies.put(("error", device_index, repr(error)))
        raise
    finally:
        engine.close()


class ParallelOcr:
    """One engine process per hardware device, work priced per assignment."""

    def __init__(self, models: OcrModels, *, use_fp16: bool = False):
        try:
            import ncnn  # noqa: PLC0415 - only to enumerate devices here
        except ImportError as error:  # pragma: no cover - environment boundary
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
        self._generation = 0
        self._closed = False
        try:
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
        except BaseException:
            # A worker that dies before its "ready" — bad driver, OOM — used
            # to raise straight out of a half-built pool: the primary engine
            # (~700 MiB of VRAM, the leak OcrEngine.close exists for) and
            # every already-started worker were never closed (VOCR-0038).
            self.close()
            raise

    def _answer(self):
        """The next worker reply, or a refusal naming the worker that died.

        A bare `Queue.get()` here blocked forever when a worker was killed —
        by the OOM reaper, a driver reset, or a person — turning every later
        read into a hang. The wait now polls, and between polls checks that
        every process is still alive; a dead one is an error with a name, not
        an eternity.
        """
        while True:
            try:
                message = self._replies.get(timeout=0.5)
            except Empty:
                for process, device in zip(self._workers, self._devices, strict=True):
                    if not process.is_alive():
                        self._retire(device.index)
                        raise OcrEngineError(
                            "worker-died",
                            f"the GPU worker for {device.name} exited with code {process.exitcode}",
                        ) from None
                continue
            if message[0] == "error":
                _kind, index, detail = message
                device = next(d for d in self._devices if d.index == index)
                self._retire(index)
                raise OcrEngineError(
                    "worker-failed", f"the GPU worker for {device.name} raised {detail}"
                )
            return message

    def _retire(self, index: int) -> None:
        """Take a dead or failing worker out of dispatch, for good.

        Left in `_cost`/`_requests`, it kept being assigned crops that
        nobody would ever answer, so every later read refused too
        (VOCR-0036). Retired, the survivors carry the next read — and with
        one device left, `read` falls back to the primary engine.
        """
        self._names.pop(index, None)
        self._cost.pop(index, None)
        channel = self._requests.pop(index, None)
        if channel is not None:
            channel.close()
            channel.cancel_join_thread()
        kept = [
            (process, device)
            for process, device in zip(self._workers, self._devices, strict=True)
            if device.index != index
        ]
        retired = [p for p, d in zip(self._workers, self._devices, strict=True) if d.index == index]
        self._workers = [process for process, _device in kept]
        self._devices = [device for _process, device in kept]
        for process in retired:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

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
        # Every message carries the read it belongs to: a read that raised
        # mid-page — a worker refusal, a Ctrl-C — left workers finishing
        # crops whose "done" replies nobody consumed, and the next read
        # took them first, pairing the previous page's crop_index with the
        # new page's regions (VOCR-0036). Stale generations are drained.
        self._generation += 1
        generation = self._generation
        pairs = self._primary.crops(rgb)
        if len(self._names) == 1 or len(pairs) < 2:
            # The crops in hand are the read: `self._primary.read(rgb)` here
            # ran detection a second time from scratch, so a one-GPU machine
            # paid it twice on every --all-gpus read (VOCR-0040).
            return assemble_result(
                self._primary.device_name,
                ((region, self._primary.recognise(patch)) for region, patch in pairs),
            )

        return assemble_result(" + ".join(self.device_names), self._dispatch(generation, pairs))

    def _dispatch(self, generation: int, pairs: list[tuple]) -> list:
        """Price and farm the crops out, and collect this read's replies."""
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
                self._requests[index].put((generation, crop_index, patch))
                busy.add(index)
                outstanding += 1

        assign()
        while outstanding:
            kind, index, reply_generation, crop_index, answer, elapsed = self._answer()
            assert kind == "done"
            if reply_generation != generation:
                # A leftover from a read that raised; its page is gone.
                continue
            outstanding -= 1
            busy.discard(index)
            region, patch = pairs[crop_index]
            self._reprice(index, elapsed, patch.shape[1])
            results[crop_index] = (region, answer)
            assign()

        return results

    def _reprice(self, index: int, elapsed_ms: float, columns: int) -> None:
        """Refine a device's ms-per-column from the crop it just finished.

        The probe seed was written once and never touched (VOCR-0039), so a
        device whose 96-column white strip was unrepresentative — fp16,
        clocks still ramping, an asymmetric detection load — mis-priced
        every assignment for the pool's whole life. An equal-weight blend:
        old enough to smooth one noisy crop, new enough that a wrong seed
        is halved by every real one.
        """
        if columns < 1 or index not in self._cost:
            return
        self._cost[index] = (self._cost[index] + elapsed_ms / columns) / 2

    def close(self) -> None:
        """Release the workers, the queues, and the primary engine.

        Safe to call twice, like `OcrEngine.close`: the second call used to
        put the shutdown sentinel into queues the first call had closed and
        raise "Queue is closed" (VOCR-0049).
        """
        if self._closed:
            return
        self._closed = True
        for channel in self._requests.values():
            # A dead worker's queue still accepts the sentinel; nothing to
            # guard beyond not caring whether anybody reads it.
            channel.put(None)
        for process in self._workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                # Reaped, not just killed: without this join the child stayed
                # a zombie for the parent's whole life (VOCR-0046).
                process.join(timeout=10)
        for channel in self._requests.values():
            # Unsent data to a dead reader is dropped, not awaited: a patch
            # still in the queue buffer left the feeder thread blocked in
            # Connection._send forever, and the interpreter's exit joined
            # that feeder — the process never ended (VOCR-0047).
            channel.close()
            channel.cancel_join_thread()
        self._replies.close()
        self._replies.cancel_join_thread()
        self._primary.close()

    def __enter__(self) -> ParallelOcr:
        return self

    def __exit__(self, *_exception) -> None:
        self.close()


__all__ = ["ParallelOcr"]
