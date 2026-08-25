"""Device policy: hardware only, discrete preferred, software refused."""

import numpy as np
import pytest

from vulkanocr.device import HardwareVulkanUnavailableError, select_hardware_device


class FakeInfo:
    def __init__(self, name, kind):
        self._name = name
        self._kind = kind

    def device_name(self):
        return self._name

    def type(self):
        return self._kind


class FakeRuntime:
    # Set per test when the engine's loader is under test; typed loosely
    # because each test assigns its own fake class.
    Net: object = None

    def __init__(self, devices):
        self._devices = devices

    def get_gpu_count(self):
        return len(self._devices)

    def get_gpu_info(self, index):
        return self._devices[index]


def test_discrete_wins_over_integrated_regardless_of_order():
    runtime = FakeRuntime([FakeInfo("iGPU", 1), FakeInfo("dGPU", 0)])
    device = select_hardware_device(runtime)
    assert (device.index, device.name, device.kind) == (1, "dGPU", 0)


def test_the_software_rasteriser_is_never_selected():
    runtime = FakeRuntime([FakeInfo("llvmpipe", 3), FakeInfo("iGPU", 1)])
    assert select_hardware_device(runtime).name == "iGPU"


def test_software_only_is_a_refusal_not_a_fallback():
    runtime = FakeRuntime([FakeInfo("llvmpipe", 3)])
    with pytest.raises(HardwareVulkanUnavailableError):
        select_hardware_device(runtime)


def test_no_devices_is_a_refusal():
    with pytest.raises(HardwareVulkanUnavailableError):
        select_hardware_device(FakeRuntime([]))


def test_first_of_equal_ranks_is_deterministic():
    runtime = FakeRuntime([FakeInfo("dGPU-a", 0), FakeInfo("dGPU-b", 0)])
    assert select_hardware_device(runtime).name == "dGPU-a"


class _FakeNet:
    def __init__(self):
        self.cleared = 0
        self.opt = type("Opt", (), {})()

    def set_vulkan_device(self, _index):
        pass

    def load_param(self, path):
        return 1 if "refuses" in path else 0

    def load_model(self, path):
        return 1 if "half-broken" in path else 0

    def clear(self):
        self.cleared += 1


class TestTheEngineReleasesWhatItHolds:
    """An engine holds ~700 MiB of VRAM; nothing freed it before (VOCR-0005)."""

    def engine(self, tmp_path, stem="model"):
        from vulkanocr.engine import OcrEngine, OcrModels

        runtime = FakeRuntime([FakeInfo("Radeon", 0)])
        runtime.Net = _FakeNet
        for name in (f"{stem}-det", f"{stem}-rec"):
            (tmp_path / f"{name}.param").write_text("7767517\n")
            (tmp_path / f"{name}.bin").write_bytes(b"")
        keys = tmp_path / "keys.txt"
        keys.write_text("a\nb\n")
        models = OcrModels(tmp_path / f"{stem}-det.param", tmp_path / f"{stem}-rec.param", keys)
        return OcrEngine(models, runtime=runtime)

    def test_close_clears_both_nets_and_is_safe_to_repeat(self, tmp_path):
        engine = self.engine(tmp_path)
        first, second = engine._det, engine._rec
        assert first is not None and second is not None

        engine.close()
        engine.close()

        assert (first.cleared, second.cleared) == (1, 1)
        assert engine._det is None and engine._rec is None

    def test_the_context_manager_closes_on_the_way_out(self, tmp_path):
        with self.engine(tmp_path) as engine:
            held = engine._det
        assert held is not None and held.cleared == 1

    def test_a_failed_second_load_frees_the_first_net(self, tmp_path):
        """The detection net must not be stranded on an object nobody gets."""
        from vulkanocr.engine import OcrEngine, OcrEngineError, OcrModels

        runtime = FakeRuntime([FakeInfo("Radeon", 0)])
        built = []

        class Recording(_FakeNet):
            def __init__(self):
                super().__init__()
                built.append(self)

        runtime.Net = Recording
        for name in ("good-det", "refuses-rec"):
            (tmp_path / f"{name}.param").write_text("7767517\n")
            (tmp_path / f"{name}.bin").write_bytes(b"")
        keys = tmp_path / "keys.txt"
        keys.write_text("a\n")
        models = OcrModels(tmp_path / "good-det.param", tmp_path / "refuses-rec.param", keys)

        import pytest

        with pytest.raises(OcrEngineError):
            OcrEngine(models, runtime=runtime)

        assert [net.cleared >= 1 for net in built] == [True] * len(built)


class TestHardwareDeviceEnumeration:
    """The parallel engine takes every hardware device, best first (VOCR-0034)."""

    def test_all_hardware_devices_come_back_ranked(self):
        from vulkanocr.device import hardware_devices

        runtime = FakeRuntime([FakeInfo("llvmpipe", 3), FakeInfo("iGPU", 1), FakeInfo("dGPU", 0)])
        devices = hardware_devices(runtime)

        assert [d.name for d in devices] == ["dGPU", "iGPU"]
        assert devices[0].index == 2  # ranked by capability, not enumeration order

    def test_the_software_rasteriser_never_joins_the_pool(self):
        from vulkanocr.device import hardware_devices

        with pytest.raises(HardwareVulkanUnavailableError):
            hardware_devices(FakeRuntime([FakeInfo("llvmpipe", 3)]))


class TestHalfAnEngineIsAskedFor:
    """A pool worker only recognises; its detection net was dead weight
    on every device (VOCR-0042)."""

    def engine(self, tmp_path, nets):
        from vulkanocr.engine import OcrEngine, OcrModels

        runtime = FakeRuntime([FakeInfo("Radeon", 0)])
        runtime.Net = _FakeNet
        for name in ("model-det", "model-rec"):
            (tmp_path / f"{name}.param").write_text("7767517\n")
            (tmp_path / f"{name}.bin").write_bytes(b"")
        keys = tmp_path / "keys.txt"
        keys.write_text("a\nb\n")
        models = OcrModels(tmp_path / "model-det.param", tmp_path / "model-rec.param", keys)
        return OcrEngine(models, runtime=runtime, nets=nets)

    def test_a_recognition_only_engine_loads_no_detection_net(self, tmp_path):
        import numpy as np
        import pytest

        from vulkanocr.engine import OcrEngineError

        engine = self.engine(tmp_path, ("rec",))
        assert engine._det is None and engine._rec is not None
        with pytest.raises(OcrEngineError) as refusal:
            engine.detect(np.zeros((4, 4, 3), dtype=np.uint8))
        assert refusal.value.code == "net-unloaded"

    def test_asking_for_no_net_at_all_is_refused(self, tmp_path):
        import pytest

        from vulkanocr.engine import OcrEngineError

        with pytest.raises(OcrEngineError) as refusal:
            self.engine(tmp_path, ())
        assert refusal.value.code == "nets-invalid"

    def test_a_recognition_only_engine_ignores_a_missing_detection_graph(self, tmp_path):
        """The halves asked for are the halves that must exist (VOCR-0052)."""
        from vulkanocr.engine import OcrEngine, OcrModels

        runtime = FakeRuntime([FakeInfo("Radeon", 0)])
        runtime.Net = _FakeNet
        for name in ("model-rec",):
            (tmp_path / f"{name}.param").write_text("7767517\n")
            (tmp_path / f"{name}.bin").write_bytes(b"")
        keys = tmp_path / "keys.txt"
        keys.write_text("a\nb\n")
        models = OcrModels(tmp_path / "absent-det.param", tmp_path / "model-rec.param", keys)

        engine = OcrEngine(models, runtime=runtime, nets=("rec",))
        assert engine._det is None and engine._rec is not None


class TestEngineInputValidation:
    def engine(self, tmp_path, **options):
        from vulkanocr.engine import OcrEngine, OcrModels

        runtime = FakeRuntime([FakeInfo("Radeon", 0)])
        runtime.Net = _FakeNet
        for name in ("model-det", "model-rec"):
            (tmp_path / f"{name}.param").write_text("7767517\n")
            (tmp_path / f"{name}.bin").write_bytes(b"")
        keys = tmp_path / "keys.txt"
        keys.write_text("a\nb\n")
        models = OcrModels(tmp_path / "model-det.param", tmp_path / "model-rec.param", keys)
        return OcrEngine(models, runtime=runtime, **options)

    @pytest.mark.parametrize("shape", [(0, 4, 3), (4, 0, 3)])
    def test_empty_image_dimensions_are_refused_before_detection(self, tmp_path, shape):
        from vulkanocr.engine import OcrEngineError

        with self.engine(tmp_path) as engine, pytest.raises(OcrEngineError) as caught:
            engine.detect(np.empty(shape, dtype=np.uint8))
        assert caught.value.code == "image-invalid"

    @pytest.mark.parametrize("target_size", [0, -1, "not-an-integer", None])
    def test_target_size_must_be_a_positive_integer(self, tmp_path, target_size):
        from vulkanocr.engine import OcrEngineError

        with pytest.raises(OcrEngineError) as caught:
            self.engine(tmp_path, target_size=target_size)
        assert caught.value.code == "target-size-invalid"

    @pytest.mark.parametrize(
        "patch",
        [
            pytest.param(np.zeros((48, 0, 3), dtype=np.uint8), id="empty"),
            pytest.param(np.zeros((47, 10, 3), dtype=np.uint8), id="height"),
            pytest.param(np.zeros((48, 10), dtype=np.uint8), id="channels"),
            pytest.param(np.zeros((48, 10, 3), dtype=np.float32), id="dtype"),
        ],
    )
    def test_invalid_recognition_patches_are_refused_before_ncnn(self, tmp_path, patch):
        from vulkanocr.engine import OcrEngineError

        with self.engine(tmp_path) as engine, pytest.raises(OcrEngineError) as caught:
            engine.recognise(patch)
        assert caught.value.code == "patch-invalid"

    def test_model_dictionary_class_mismatch_is_a_stable_refusal(self, tmp_path):
        from vulkanocr.engine import OcrEngineError

        with self.engine(tmp_path) as engine, pytest.raises(OcrEngineError) as caught:
            engine.decode(np.zeros((4, 5), dtype=np.float32))
        assert caught.value.code == "dictionary-mismatch"
        assert "5 classes" in caught.value.detail
