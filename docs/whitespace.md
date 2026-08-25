# Whitespace reconstruction

The recognizer owns characters. Layout postprocessing may only infer separators
between distinct recognised detector regions; it never changes CTC decoding or
normalises whitespace already present in a decoded region.

## Geometry contract

Each region contributes its oriented rectangle's long axis. The axis is made
canonical from left to right for horizontal text and top to bottom for vertical
text, so skewed regions are compared in their local coordinate system rather
than by page `x`/`y` alone.

Two regions can belong to the same visual line only when all of these hold:

- both are horizontal or both are vertical;
- their long axes differ by at most 12 degrees;
- their perpendicular centre offset is at most 0.6 times the greater thickness;
- their nearest long-axis edges overlap by no more than both two times the
  greater thickness and half the shorter region length, and are no more than
  four times the greater thickness apart; and
- their strong Unicode directions agree, or at least one region is neutral.

These conservative limits keep duplicate/strongly overlapping boxes, nearby
columns, and separate rows independent while admitting the modest overlap DB
unclip adds around neighbouring fragments.
Eligible neighbours form a visual-line component. Left-to-right components are
ordered forward on the canonical axis; components containing only strong
right-to-left text are ordered backward. Characters inside each recognition
are never reversed. Mixed strong left-to-right/right-to-left fragments stay
separate because correct bidirectional layout needs more information than the
detector rectangles provide.

## Gap contract

Let `g` be the projected edge-to-edge gap between adjacent regions and `t` the
median region thickness in their visual-line component.

- Existing leading or trailing whitespace is preserved exactly and no extra
  separator is generated at that boundary.
- No separator is generated before closing or terminal punctuation, or after
  opening punctuation.
- When `g < 0.25t`, the fragments are concatenated without a separator.
- Otherwise the separator contains `round(g / 0.5t)` ASCII spaces, rounded
  half up and limited to one through eight spaces.

The same thresholds apply to proportional and monospace text because thickness,
not decoded character count, supplies the scale. The eight-space ceiling follows
from the four-thickness visual-line limit and prevents a layout gap from growing
an unbounded text payload.

## Result contract and limits

Reconstruction runs after empty recognitions and policy-rejected regions have
been counted. A joined result encloses its source rectangles along their local
axes and combines confidence and box score by non-whitespace character weight.
Regions that do not qualify remain separate `OcrLine` values.

This heuristic can infer one or more separators only between distinct regions.
It cannot recover whitespace that CTC omitted inside one region, nor prove how
many spaces existed in the source image. Exact intra-region reconstruction
would require character/timestep geometry or a layout model; tests must not
claim that capability.
