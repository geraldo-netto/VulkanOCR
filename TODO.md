# VulkanOCR review — 2026-08-21

| id | severity | effort | description |
| --- | --- | --- | --- |
| VOCR-0061 | medium | s | **Consumers restate the port facts the catalog owns.** omnitensor's two OCR adapters each hardcode `blobs=("input","output")` and `dictionary_includes_blank=True` when building `OcrModels` from artifact paths — the facts `catalog.py` records per port, but only reachable through the clone-layout `models_for`. Export a named-port constructor (e.g. `OcrModels.for_port("avafly-v6", det_param, rec_param, dictionary)`) so an external consumer names the port and inherits its facts, and a port drift breaks loudly in one place. |
