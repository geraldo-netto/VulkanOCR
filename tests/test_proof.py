"""The busy-sampler, proven on counters it can actually read."""

import time

from vulkanocr.proof import busy_sampler


def test_the_sampler_collects_while_the_block_runs_and_stops_after(tmp_path):
    counter = tmp_path / "gpu_busy_percent"
    counter.write_text("37\n")
    with busy_sampler(paths=[counter]) as samples:
        time.sleep(0.08)
    collected = len(samples[counter])
    assert collected >= 2
    assert set(samples[counter]) == {37}
    # Stopped: nothing more arrives after the block.
    time.sleep(0.05)
    assert len(samples[counter]) == collected


def test_a_counter_that_stops_answering_is_skipped_not_fatal(tmp_path):
    vanishing = tmp_path / "gpu_busy_percent"
    vanishing.write_text("nonsense")
    with busy_sampler(paths=[vanishing]) as samples:
        time.sleep(0.05)
    assert samples[vanishing] == []
