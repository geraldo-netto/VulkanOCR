# VulkanOCR review — 2026-08-21

| id | severity | effort | description |
| --- | --- | --- | --- |
| VOCR-0045 | low | xs | The CLI hides `undecoded_regions`: `OcrResult` carries it precisely so a caller can tell "clean page" from "the detector saw something the recogniser could not read" (src/vulkanocr/engine.py:114-127), but cli.py prints only the lines and their count (cli.py:111-115), so both cases print identically — the invisibility the field's own docstring warns about. One line when the count is non-zero. |
