# VulkanOCR review — 2026-08-21

| id | severity | effort | description |
| --- | --- | --- | --- |
| VOCR-0013 | medium | xs | **The README's line counts are stale by ~27 %.** `README.md:41` claims "585 lines of engine, 231 of tests". Actual `wc -l src/vulkanocr/*.py` = **745** (catalog 124, cli 99, detection 179, device 47, engine 177, `__init__` 29, recognition 90) and `wc -l tests/*.py` = **227** (catalog 29, ctc 58, device 55, live_gpu 85). The "21 tests" claim on the same page (`:35`, `:58`) is correct — `pytest -q` reports `21 passed`. Either update the numbers or drop them; a hand-maintained line count in a README goes stale on every commit. |
