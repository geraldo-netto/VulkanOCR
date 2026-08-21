# VulkanOCR review — 2026-08-21

| id | severity | effort | description |
| --- | --- | --- | --- |
| VOCR-0035 | high | s | **A GPU worker that dies hangs the read forever.** `parallel.py` blocks on `self._replies.get()` with no timeout and no liveness check, both in the constructor (waiting for `ready`) and in `read()`'s dispatch loop — so a worker killed by the OOM reaper or a driver reset turns every subsequent read into an indefinite hang instead of an error naming the dead device. Poll with a short timeout, check `process.is_alive()` between polls, and raise a stable `OcrEngineError` when a worker is gone; `close()` must survive already-dead workers. |
