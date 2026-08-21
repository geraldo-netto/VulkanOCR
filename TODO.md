# VulkanOCR review — 2026-08-21

| id | severity | effort | description |
| --- | --- | --- | --- |
| VOCR-0054 | low | xs | **The empty-results guard in the comparison table is dead code.** `benchmarks/compare_engines.py:24-31`: the aggregate divisions (`cer = sum(...)/sum(...)`, `exact = .../len(rows)`) run before `if not rows:` is reached, so an empty results document crashes the whole table with `ZeroDivisionError` — the exact failure the guard's own comment says it exists to prevent ("naming it beats dividing by zero in the middle of the table"). Confirmed by execution against a `rows: []` document. Move the guard above the arithmetic. |
| VOCR-0056 | low | xs | **`OcrEngine.close`'s docstring still promises "both nets".** src/vulkanocr/engine.py:198: since the halves seam (VOCR-0042/0052) a recognition-only engine holds one net; the docstring's "Release both nets and their Vulkan allocations" describes the engine that no pool worker builds any more. Say "every loaded net". |
