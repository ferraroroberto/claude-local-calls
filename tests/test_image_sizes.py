"""Size parsing, validation and the native/upscale split (#497)."""

from __future__ import annotations

import pytest

from src import image_sizes as sz
from src.image_sizes import ImageSizeError


# --- presets are internally consistent ------------------------------------

def test_every_preset_is_itself_a_valid_size():
    """The presets are the sizes we tell users to ask for; if one of them
    isn't parseable the UI offers something the API rejects."""
    for p in sz.PRESETS:
        assert p.width % sz.DIMENSION_MULTIPLE == 0, p.name
        assert p.height % sz.DIMENSION_MULTIPLE == 0, p.name
        assert sz.MIN_DIMENSION <= p.width <= sz.MAX_DIMENSION, p.name
        assert sz.MIN_DIMENSION <= p.height <= sz.MAX_DIMENSION, p.name
        assert sz.parse_size(p.name) == (p.width, p.height)


def test_preset_names_are_unique():
    names = [p.name for p in sz.PRESETS]
    assert len(names) == len(set(names))


def test_hd_preset_is_1088_not_1080():
    """1080 is not a multiple of 16. We return 1088 and say so, rather than
    accepting 1920x1080 and quietly handing back different dimensions."""
    assert sz.parse_size("hd") == (1920, 1088)


# --- parsing --------------------------------------------------------------

def test_parses_explicit_dimensions():
    assert sz.parse_size("1216x832") == (1216, 832)


def test_parses_unicode_multiplication_sign_and_whitespace():
    assert sz.parse_size(" 1024 × 1024 ") == (1024, 1024)


def test_preset_lookup_is_case_insensitive():
    assert sz.parse_size("4K") == sz.parse_size("4k")


def test_empty_size_falls_back_to_default():
    assert sz.parse_size(None) == sz.parse_size(sz.DEFAULT_SIZE)
    assert sz.parse_size("") == sz.parse_size(sz.DEFAULT_SIZE)


def test_unknown_preset_lists_the_valid_ones():
    with pytest.raises(ImageSizeError) as e:
        sz.parse_size("gigantic")
    assert "square" in str(e.value) and "4k" in str(e.value)


def test_off_grid_size_is_rejected_with_a_usable_suggestion():
    """Rejected, not silently snapped — a caller who asked for 1000x1000 and
    got 992x992 back with no warning would reasonably call that a bug.

    Asserts the *property* the suggestion must have (on-grid and close) rather
    than one exact pair: 1000 is equidistant from 992 and 1008, so pinning the
    tie-break would be testing Python's rounding rule, not our contract.
    """
    with pytest.raises(ImageSizeError) as e:
        sz.parse_size("1000x1000")
    message = str(e.value)
    suggestion = message.rsplit(":", 1)[-1].strip()
    width, height = (int(v) for v in suggestion.split("x"))
    assert width % sz.DIMENSION_MULTIPLE == 0
    assert height % sz.DIMENSION_MULTIPLE == 0
    assert abs(width - 1000) <= sz.DIMENSION_MULTIPLE
    # And the suggestion must itself be accepted, or it isn't a fix.
    assert sz.parse_size(suggestion) == (width, height)


def test_1080_rejection_explains_the_16_multiple():
    with pytest.raises(ImageSizeError) as e:
        sz.parse_size("1920x1080")
    assert "1088" in str(e.value)


@pytest.mark.parametrize("bad", ["16x16", "9000x1024", "1024x9000", "abc", "1024", "1024xx1024"])
def test_out_of_range_and_malformed_are_rejected(bad):
    with pytest.raises(ImageSizeError):
        sz.parse_size(bad)


# --- native vs upscale ----------------------------------------------------

def test_native_sizes_are_not_upscaled():
    assert sz.needs_upscale(1024, 1024) is False
    assert sz.needs_upscale(1920, 1088) is False  # deliberately just inside


def test_4k_needs_upscale():
    assert sz.needs_upscale(3840, 2160) is True


def test_native_source_is_unchanged_below_the_ceiling():
    assert sz.native_source_size(1216, 832) == (1216, 832)


def test_native_source_fits_under_the_ceiling():
    for w, h in [(3840, 2160), (4096, 4096), (2048, 2048), (3840, 1600)]:
        src_w, src_h = sz.native_source_size(w, h)
        assert src_w * src_h <= sz.NATIVE_MAX_PIXELS, (w, h, src_w, src_h)


def test_native_source_preserves_aspect_ratio():
    """Ratio is what carries composition through the upscale; the scale factor
    is free. Allow a little slack for snapping each edge to the 16-grid."""
    for w, h in [(3840, 2160), (4096, 4096), (3840, 1600)]:
        src_w, src_h = sz.native_source_size(w, h)
        assert abs((src_w / src_h) - (w / h)) < 0.02, (w, h, src_w, src_h)


def test_native_source_is_the_largest_that_fits_not_merely_one_that_fits():
    """Resolution thrown away here is resolution the upscaler has to invent.
    4K must sample at 1920x1088 — exactly the `hd` preset, the largest 16:9
    pair under the ceiling — not at some smaller pair that also happens to fit.
    """
    assert sz.native_source_size(3840, 2160) == (1920, 1088)
    # And that really is the largest: one grid step up must not fit.
    assert (1936 * 1088) > sz.NATIVE_MAX_PIXELS


def test_square_targets_get_square_sources():
    """Regression: stepping a single edge down to fit the ceiling skewed the
    ratio — a 1:1 target sampled at 1440x1456, a ~1% stretch applied at the
    final ImageScale. Both edges must come from the same scale factor."""
    for edge in (2048, 3072, 4096):
        src_w, src_h = sz.native_source_size(edge, edge)
        assert src_w == src_h, (edge, src_w, src_h)


def test_native_source_stays_on_the_16_grid():
    for w, h in [(3840, 2160), (4096, 4096), (2560, 1440)]:
        src_w, src_h = sz.native_source_size(w, h)
        assert src_w % sz.DIMENSION_MULTIPLE == 0
        assert src_h % sz.DIMENSION_MULTIPLE == 0


# --- payload for the UI ---------------------------------------------------

def test_preset_payload_flags_upscaled_entries():
    payload = {p["name"]: p for p in sz.preset_payload()}
    assert payload["square"]["upscaled"] is False
    assert payload["4k"]["upscaled"] is True
    assert payload["4k"]["width"] == 3840


def test_preset_payload_carries_everything_the_dropdown_renders():
    for row in sz.preset_payload():
        for key in ("name", "width", "height", "label", "ratio", "megapixels", "upscaled"):
            assert key in row
