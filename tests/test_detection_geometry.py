"""The geometry invariant recognition depends on, checked without a GPU.

`crop_region` computes its output width as `long / short * 48`, so if
`_oriented` ever returns the long side first the crop collapses to a sliver
and the band reads as confident nonsense — which is exactly what happened for
30°–60° skew before the invariant below was enforced (VOCR-0001).
"""

import numpy as np

from vulkanocr.detection import TextRegion, _oriented, unclip_offset
from vulkanocr.recognition import crop_region


def test_the_short_side_always_comes_first():
    """Every angle OpenCV can produce, every aspect a text line can have.

    OpenCV 5's `minAreaRect` reports angles in [-90, 0); older releases used
    other conventions, so the grid runs the full circle to cover them all.
    """
    for angle_int in range(-180, 181, 1):
        for short, long in ((12.0, 40.0), (20.0, 300.0), (54.0, 129.0), (47.0, 1088.0)):
            for sides in ((short, long), (long, short)):
                out_w, out_h, _angle, vertical = _oriented(sides[0], sides[1], float(angle_int))
                if not vertical:
                    assert out_w <= out_h, (sides, angle_int, out_w, out_h)


def test_the_45_degree_band_yields_a_real_crop_not_a_sliver():
    """The exact triple measured from a 45° line before the fix: (129.4, 53.7, -45)."""
    out_w, out_h, _angle, _vertical = _oriented(129.4, 53.7, -45.0)
    region = TextRegion(
        center_x=450.0,
        center_y=150.0,
        width=out_w,
        height=out_h,
        angle=_angle,
        vertical=False,
        score=0.9,
    )
    patch = crop_region(np.full((300, 900, 3), 255, np.uint8), region)

    assert patch.shape[0] == 48
    # 48 * long/short ≈ 115 columns; the bug produced 19.
    assert patch.shape[1] > 80, patch.shape


def test_unclip_grows_a_line_by_roughly_three_quarters_of_its_height():
    """DB's rule, area * ratio / perimeter, for a long thin line."""
    offset = unclip_offset(16.0, 400.0)
    assert 11.0 < offset < 12.0
    assert unclip_offset(0.0, 0.0) == 0.0
