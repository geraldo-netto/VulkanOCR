# TODO

## Open

| id | status | severity | effort | description |
| --- | --- | --- | --- | --- |
| VOCR-0076 | open | medium | s | Return and print an explicit telemetry-unavailable proof state instead of an empty `gpu_busy_percent` section on drivers without that counter. |
| VOCR-0079 | open | medium | xs | Define and document a geometry-based word-gap heuristic, including punctuation and right-to-left behavior, before adding whitespace postprocessing. |
| VOCR-0080 | open | low | s | Unify benchmark engine lifecycle and no-text precondition handling in a shared harness; several scripts manually close resources only on success, while batching scripts crash on blank images through `min`, division by zero, or a zero range step. |
| VOCR-0116 | open | medium | s | Define a proof-provider result carrying selected-device identity, samples, supported state, and an unavailable reason so ambient work on another GPU cannot count as proof. |
| VOCR-0117 | open | medium | s | Add per-process DRM fdinfo proof where supported and retain AMD `gpu_busy_percent` as an explicitly system-wide fallback. |
| VOCR-0118 | open | medium | s | Add fake AMD, Intel, NVIDIA, missing-counter, and wrong-device proof tests plus CLI documentation for each telemetry state. |
| VOCR-0123 | open | medium | s | Apply the false-positive policy during result assembly and prove it rejects labeled icons while retaining legitimate CJK corpus lines. |
| VOCR-0124 | open | medium | s | Add a whitespace-reconstruction stage over recognised regions using the approved gap heuristic without changing CTC decoding. |
| VOCR-0125 | open | medium | s | Add proportional-font, monospace, punctuation, multiple-space, rotated-line, and right-to-left whitespace reconstruction regressions. |

## Blocked / Deferred

| id | status | severity | effort | description |
| --- | --- | --- | --- | --- |
| VOCR-0083 | blocked | high | xs | Select and approve a compatible Hebrew recognition model, licence, and provenance route; current catalog dictionaries contain no Hebrew glyphs. Hebrew sample generation remains open. |
| VOCR-0098 | blocked | high | s | Validate int8 inference end to end and record accuracy/performance after an owner supplies or approves a quantized OCR model artifact and matching dictionary. |
| VOCR-0126 | blocked | high | s | Convert or acquire and checksum-pin the approved Hebrew recognition graph after its model and provenance route are selected. |
| VOCR-0127 | blocked | high | s | Add the approved Hebrew recognizer and dictionary as a catalog profile after compatible model artifacts exist. |
| VOCR-0128 | blocked | high | s | Run Hebrew corpus acceptance and publish accuracy limitations after the approved Hebrew catalog profile is runnable. |
| VOCR-0129 | blocked | medium | m | Replace whole-upstream model downloads with a checksum-pinned component model store and explicit installer for detector, recognizer, dictionary, and orientation artifacts; blocked on approving selective extraction, upstream split assets, or VulkanOCR-hosted per-profile assets for Avafly's monolithic release archive. |

## Rejected / Won't fix

| id | status | severity | effort | description |
| --- | --- | --- | --- | --- |
