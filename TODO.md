# VulkanOCR review — 2026-08-21

| id | severity | effort | description |
| --- | --- | --- | --- |
| VOCR-0057 | low | xs | **README's module map omits proof.py.** README.md:35-43 lists every src module and describes cli.py as carrying the sampling that VOCR-0053 moved out; `proof.py` (sysfs busy sampling as a context manager) is absent. The map says "the claim is the shape, not a census" — and the shape changed. docs/engine-notes.md was updated in the same commit; the README was missed. |
