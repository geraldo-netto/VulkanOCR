# TODO

## Open

| id | status | severity | effort | description |
| --- | --- | --- | --- | --- |
| VOCR-0062 | open | high | s | Define a versioned corpus-case schema carrying script, language, exact lines, font and licence, palette, size, background objects, variant, and image path; validate every generated manifest against it. |
| VOCR-0064 | open | medium | s | Add independent immutable detector and recognizer specifications with their own paths, blob names, dictionary convention, and precision requirements. |
| VOCR-0065 | open | high | s | Extract crop assignment and device repricing from `ParallelOcr._dispatch` into a pure scheduling policy with no queues, processes, ncnn objects, or OCR assembly. |
| VOCR-0066 | open | medium | s | Extract detector and recognizer extractor creation, input submission, output extraction, and cleanup into one checked ncnn inference helper. |
| VOCR-0070 | open | high | s | Build the parent engine with only the detection net when a multi-GPU worker fleet is available, eliminating the preferred GPU's idle duplicate recognition net. |
| VOCR-0071 | open | high | s | Add a configurable worker-readiness deadline that closes the partial fleet and reports a stable startup error when a live child never sends `ready`. |
| VOCR-0072 | open | medium | s | Make the configured Pyright check clean and add it to documented development gates; it currently reports 10 errors, including nullable `_replies` access and test doubles incompatible with inferred concrete `ParallelOcr` internals. |
| VOCR-0073 | open | high | xs | Choose and document whether corpus accuracy includes reading order; current concatenated Levenshtein scoring contradicts the documented claim that order is ignored. |
| VOCR-0074 | open | high | s | Validate that every engine result contains exactly the corpus case ids and variants before comparison; a missing variant currently produces an empty subset and is printed as CER 0.000, silently presenting absent work as perfect recognition. |
| VOCR-0075 | open | medium | s | Make `make_corpus.py` importable and its output trustworthy: parse the output path inside `main`, validate configured fonts, check every `cv2.imwrite` result, and centralize manifest/image writes in a tested corpus writer. Import currently raises `IndexError` without a CLI argument and failed image writes are reported as success. |
| VOCR-0076 | open | medium | s | Return and print an explicit telemetry-unavailable proof state instead of an empty `gpu_busy_percent` section on drivers without that counter. |
| VOCR-0077 | open | high | s | Add PP-LCNet orientation-model paths, blob facts, and file validation to model specifications and catalog profiles. |
| VOCR-0078 | open | medium | s | Build labeled positive-CJK and negative glyph-icon/background-object cases that reproduce false readings such as `花` and `回`. |
| VOCR-0079 | open | medium | xs | Define and document a geometry-based word-gap heuristic, including punctuation and right-to-left behavior, before adding whitespace postprocessing. |
| VOCR-0080 | open | low | s | Unify benchmark engine lifecycle and no-text precondition handling in a shared harness; several scripts manually close resources only on success, while batching scripts crash on blank images through `min`, division by zero, or a zero range step. |
| VOCR-0081 | open | low | xs | Reject negative `--repeat` values in the CLI instead of silently treating them as zero extra passes. |
| VOCR-0082 | open | low | xs | Restore documentation consistency: link active gaps to `TODO.md` instead of the historical `docs/findings.md`, and remove or update stale claims of 74 tests now that the suite collects 76. |
| VOCR-0084 | open | high | s | Generate and commit Russian/Cyrillic corpus samples with representative letters, punctuation, exact ground truth, and an appropriate licensed font. |
| VOCR-0085 | open | high | s | Generate and commit Greek corpus samples with tonos and dialytika coverage, punctuation, exact ground truth, and an appropriate licensed font. |
| VOCR-0086 | open | high | s | Generate and commit Japanese corpus samples mixing hiragana, katakana, kanji, punctuation, exact ground truth, and an appropriate licensed font. |
| VOCR-0087 | open | high | s | Generate and commit Mandarin Chinese corpus samples covering simplified and traditional glyphs, punctuation, exact ground truth, and an appropriate licensed font. |
| VOCR-0088 | open | high | s | Generate and commit correctly shaped right-to-left Arabic corpus samples with punctuation, exact ground truth, and an appropriate licensed font. |
| VOCR-0089 | open | high | s | Generate and commit correctly shaped right-to-left Hebrew corpus samples with punctuation and exact ground truth independently of recognition-model availability. |
| VOCR-0090 | open | high | s | Generate and commit Georgian corpus samples covering Mkhedruli text, punctuation, exact ground truth, and an appropriate licensed font. |
| VOCR-0091 | open | high | s | Generate and commit Azerbaijani corpus samples covering `Ə`, `Ğ`, `İ`, `Ö`, `Ş`, `Ü`, `Ç`, exact ground truth, and an appropriate licensed font. |
| VOCR-0092 | open | high | s | Add deterministic visual variants combining multiple foreground/background colour palettes, font sizes and families, and non-text background objects without clipping text. |
| VOCR-0093 | open | medium | s | Add configurable font discovery with script-coverage checks and record each corpus font's source, version, and redistribution licence. |
| VOCR-0094 | open | high | s | Make corpus runners select the declared language/model per case and refuse result sets whose case ids differ from the manifest. |
| VOCR-0097 | open | high | s | Add CLI and benchmark precision flags, result tags, help, and documentation that distinguish fp32, fp16, and quantized/int8 runs. |
| VOCR-0099 | open | medium | s | Compose named model profiles from detector and recognizer specifications so language-specific or quantized recognizers reuse an existing detector. |
| VOCR-0100 | open | medium | s | Migrate `CATALOG`, `PORT_FACTS`, constructors, exports, and catalog tests to composed model profiles without changing existing names or defaults. |
| VOCR-0101 | open | high | s | Define a worker-fleet protocol and isolate multiprocessing queues, processes, handshake parsing, retirement, and shutdown in one adapter. |
| VOCR-0102 | open | high | s | Inject runtime, device enumeration, engine construction, and worker-fleet factories at `ParallelOcr` boundaries instead of importing concrete implementations internally. |
| VOCR-0103 | open | high | s | Reduce `ParallelOcr` to detection, scheduler/fleet coordination, and result assembly over the extracted interfaces. |
| VOCR-0104 | open | medium | s | Replace `ParallelOcr` tests that allocate with `__new__` and overwrite private concrete fields with protocol-backed fakes using normal construction. |
| VOCR-0105 | open | medium | s | Validate detection inference output rank and probability-map dimensions before contour processing, with stable errors for incompatible model ports. |
| VOCR-0106 | open | medium | s | Validate recognition inference output rank and timestep/class dimensions before CTC decoding, with stable errors for incompatible model ports. |
| VOCR-0107 | open | medium | s | Translate ncnn input/extract exceptions and return codes to stage-specific `OcrEngineError` values and make the CLI catch errors raised during reads. |
| VOCR-0109 | open | high | s | Lazily load preferred-device recognition only after every worker is unavailable, preserving single-device fallback without idle multi-GPU allocations. |
| VOCR-0110 | open | high | s | Add lifecycle and fake-net allocation tests proving multi-GPU startup holds one detector and one recognition net per worker, then releases lazy fallback resources. |
| VOCR-0111 | open | high | s | Add an inference-response watchdog based on in-flight jobs so a live worker that stops replying becomes a stable timeout instead of an endless read. |
| VOCR-0112 | open | high | s | Retire non-responsive workers, discard their outstanding generation safely, and add alive-but-silent worker regressions for startup and inference. |
| VOCR-0113 | open | high | s | Preserve ground-truth and observed text as line sequences in corpus result documents instead of joining each page before scoring. |
| VOCR-0114 | open | high | s | Implement the selected reading-order policy, including order-independent line alignment when order is excluded, while preserving aggregate CER and WER arithmetic. |
| VOCR-0115 | open | high | s | Add reversed-line, multi-column, and right-to-left scorer regressions and update benchmark documentation to match measured order semantics. |
| VOCR-0116 | open | medium | s | Define a proof-provider result carrying selected-device identity, samples, supported state, and an unavailable reason so ambient work on another GPU cannot count as proof. |
| VOCR-0117 | open | medium | s | Add per-process DRM fdinfo proof where supported and retain AMD `gpu_busy_percent` as an explicitly system-wide fallback. |
| VOCR-0118 | open | medium | s | Add fake AMD, Intel, NVIDIA, missing-counter, and wrong-device proof tests plus CLI documentation for each telemetry state. |
| VOCR-0119 | open | high | s | Load and close the optional PP-LCNet orientation net with the same checked lifecycle and device/precision policy as detection and recognition nets. |
| VOCR-0120 | open | high | s | Classify each detected patch's orientation and rotate it to the recognizer's expected direction before CTC inference. |
| VOCR-0121 | open | high | s | Add horizontal, 90-degree, 180-degree, and 270-degree unit and live-GPU regressions plus orientation cases in the corpus runner. |
| VOCR-0122 | open | medium | s | Add a configurable script-aware false-positive policy that can use confidence and page/model language context without globally suppressing CJK output. |
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

## Rejected / Won't fix

| id | status | severity | effort | description |
| --- | --- | --- | --- | --- |
