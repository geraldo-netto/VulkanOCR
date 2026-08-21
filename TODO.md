# VulkanOCR review — 2026-08-21

| id | severity | effort | description |
| --- | --- | --- | --- |
| VOCR-0056 | low | xs | **`OcrEngine.close`'s docstring still promises "both nets".** src/vulkanocr/engine.py:198: since the halves seam (VOCR-0042/0052) a recognition-only engine holds one net; the docstring's "Release both nets and their Vulkan allocations" describes the engine that no pool worker builds any more. Say "every loaded net". |
