# VulkanOCR review — 2026-08-21

| id | severity | effort | description |
| --- | --- | --- | --- |
| VOCR-0034 | high | m | **OCR uses one GPU however many the machine has.** Owner instruction 2026-08-21: run in parallel across GPUs when 2+ are present. `device.py` selects exactly one hardware device and `OcrEngine` builds both nets on it, so this desk's Radeon 610M idles while the RX 6600 XT works through 46 sequential recognition calls (~600 ms of a ~670 ms page). Recognition is embarrassingly parallel per crop. Add a per-device engine pool that shares the work dynamically — a faster card naturally takes more crops — with detection on the preferred device, results in deterministic order, and identical text regardless of device count. Measure before defaulting it on: the second GPU here is 6x weaker (r-score 11 vs 63), and threads through the ncnn binding may hold the GIL; if measurement says it does not pay on asymmetric pairs, land it as an opt-in with the numbers recorded. |
