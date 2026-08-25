"""Explicit ncnn precision policy as an immutable value."""

from dataclasses import FrozenInstanceError

import pytest

from vulkanocr import PRECISIONS, InferenceOptions, options_for_precision


def test_defaults_state_existing_fp32_and_int8_runtime_policy():
    options = InferenceOptions()

    assert options.use_vulkan_compute is True
    assert (
        options.use_fp16_packed,
        options.use_fp16_storage,
        options.use_fp16_arithmetic,
    ) == (False, False, False)
    assert (
        options.use_int8_inference,
        options.use_int8_packed,
        options.use_int8_storage,
        options.use_int8_arithmetic,
    ) == (True, True, True, False)


def test_named_fp16_policy_enables_every_fp16_optimization():
    options = InferenceOptions.fp16()

    assert options.use_fp16_packed is True
    assert options.use_fp16_storage is True
    assert options.use_fp16_arithmetic is True


def test_named_int8_policy_enables_quantized_arithmetic():
    assert InferenceOptions.int8().use_int8_arithmetic is True


def test_options_are_immutable():
    options = InferenceOptions()
    attribute = "use_fp16_storage"
    with pytest.raises(FrozenInstanceError):
        setattr(options, attribute, True)


def test_user_facing_precision_names_map_to_explicit_options():
    assert PRECISIONS == ("fp32", "fp16", "int8")
    assert options_for_precision("fp32") == InferenceOptions()
    assert options_for_precision("fp16") == InferenceOptions.fp16()
    assert options_for_precision("int8") == InferenceOptions.int8()


def test_unknown_precision_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown precision"):
        options_for_precision("bf16")
