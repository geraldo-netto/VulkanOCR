"""The scorer behind every published CER/WER number, tested as arithmetic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from scoring import levenshtein, load_rgb, normalise, score, score_lines  # noqa: E402


class TestNormalise:
    def test_whitespace_collapses_and_composition_is_canonical(self):
        assert normalise("  a\t b\n c ") == "a b c"
        # NFD "ç" (c + combining cedilla) must equal the NFC form, or an
        # engine would be charged for an accent it read correctly.
        decomposed = "çora̧ção"
        assert normalise("coração") != decomposed  # raw strings differ
        assert normalise("ç") == "ç"


class TestLevenshtein:
    @pytest.mark.parametrize(
        ("left", "right", "distance"),
        [
            ("", "", 0),
            ("abc", "", 3),
            ("", "abc", 3),
            ("abc", "abc", 0),
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("ação", "acao", 2),
        ],
    )
    def test_known_distances(self, left, right, distance):
        assert levenshtein(left, right) == distance
        assert levenshtein(right, left) == distance

    def test_word_sequences_are_first_class(self):
        assert levenshtein(["a", "b", "c"], ["a", "c"]) == 1


class TestScore:
    def test_a_perfect_read_scores_zero_everywhere(self):
        result = score("The quick fox", "The  quick   fox")
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
        assert result["exact"] is True

    def test_rates_divide_by_truth_length_not_by_observation(self):
        result = score("ab", "abcdef")
        assert result["char_length"] == 2
        assert result["cer"] == pytest.approx(2.0)  # 4 insertions / 2 chars

    def test_empty_truth_does_not_divide_by_zero(self):
        result = score("", "ghost text")
        assert result["cer"] >= 0.0
        assert result["exact"] is False

    def test_line_assignment_excludes_traversal_order(self):
        assert score_lines(["first line", "second line"], ["first line", "second line"])["exact"]
        reordered = score_lines(["first line", "second line"], ["second line", "first line"])
        assert reordered["cer"] == 0
        assert reordered["wer"] == 0
        assert reordered["exact"] is True

    def test_line_assignment_preserves_total_truth_denominators(self):
        result = score_lines(["ab", "c"], ["ax"])
        assert result["char_distance"] == 2
        assert result["char_length"] == 3
        assert result["cer"] == pytest.approx(2 / 3)
        assert result["word_distance"] == 2
        assert result["word_length"] == 2


class TestLoadRgb:
    def test_a_path_nothing_can_decode_is_refused_by_name(self, tmp_path):
        broken = tmp_path / "not-an-image.png"
        broken.write_bytes(b"definitely not a png")
        with pytest.raises(FileNotFoundError, match="not-an-image.png"):
            load_rgb(broken)

    def test_a_real_image_comes_back_rgb_and_contiguous(self, tmp_path):
        import cv2
        import numpy as np

        path = tmp_path / "tiny.png"
        bgr = np.zeros((4, 4, 3), np.uint8)
        bgr[:, :, 0] = 255  # blue in BGR
        cv2.imwrite(str(path), bgr)

        rgb = load_rgb(path)

        assert rgb.flags["C_CONTIGUOUS"]
        assert rgb[0, 0].tolist() == [0, 0, 255]  # blue lands in the R-last slot
