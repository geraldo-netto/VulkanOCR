# VulkanOCR review — 2026-08-21

| id | severity | effort | description |
| --- | --- | --- | --- |
| VOCR-0060 | medium | xs | **Tag v0.1.0: the package gains its first external consumer.** omnitensor's media-transcription provider is taking `vulkanocr>=0.1,<1` as a dependency, which makes OcrEngine/OcrResult/device=/nets= a contract someone else depends on. Tag the current develop as v0.1.0 so the pin resolves to something named, and treat public-surface changes as versioned from here. |
