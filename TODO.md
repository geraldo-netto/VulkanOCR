# VulkanOCR review — 2026-08-21

| id | severity | effort | description |
| --- | --- | --- | --- |
| VOCR-0052 | low | xs | **A recognition-only engine still demands the detection files exist.** `nets=("rec",)` (VOCR-0042) skips loading the det net, but `models.validated()` (called at engine.py:157) still requires `det_param` and its `.bin` on disk — a "half engine" whose contract validates the whole catalog entry. Validate only the halves that were asked for (the dictionary always). |
| VOCR-0053 | low | s | **`_read_and_report` holds three jobs.** src/vulkanocr/cli.py:87-134 owns the sampler thread's lifecycle, the timing loop, and all of the printing in one function; the sysfs busy-sampler (`gpu_busy_paths`/`sample_gpu_busy`, cli.py:24-36) is proof instrumentation, not CLI. Wrap the sampler as a context manager in its own module and let the report function only report — which also makes the sampler reusable and the printing testable. |
