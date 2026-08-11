"""Image size presets, parsing and validation for ``/v1/images/*`` (#497).

One source of truth for *what sizes exist and which are sane*, shared by the
API (which validates), the ComfyUI client (which decides whether a request
needs the upscale path) and the admin Playground (which renders the dropdown).
Keeping it here rather than in ``comfyui_client`` is deliberate: this is an
API-contract concern, and a second image backend must not mean a second table.

Three facts drive everything below.

**Dimensions must be multiples of 16.** FLUX's VAE downsamples by 8 and the
transformer patchifies by a further 2. Off-grid dimensions get silently padded
and come back with smeared edges. Note the consequence for "1080p": 1080 is not
a multiple of 16, so the honest 16:9 HD size is **1920x1088**. We return that
and say so rather than quietly handing back something the caller didn't ask for.

**There is a native sampling ceiling.** FLUX.1 [dev] is trained around 1 MP.
Sampling natively at 4K (8.3 MP) is a *quality* cliff before it is a speed one:
attention cost grows with the square of the token count, and composition
degrades into duplicated subjects and incoherent layout. So anything above
:data:`NATIVE_MAX_PIXELS` is generated at the largest native-safe size *of the
same aspect ratio* and then upscaled — see ``comfyui_client``.

**Aspect ratio is preserved, never the pixel count.** When we pick a native
size to upscale from, matching the ratio is what keeps the composition; the
scale factor is free.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# FLUX latent grid: VAE /8, then patchify /2.
DIMENSION_MULTIPLE = 16
# Smallest sensible edge — below this FLUX composition falls apart.
MIN_DIMENSION = 256
# Hard upper bound on any single edge, upscaled included.
MAX_DIMENSION = 4096
# Above this, generate smaller and upscale rather than sampling natively.
# ~2 MP: comfortably past FLUX's ~1 MP training point without falling off the
# quality cliff. 1920x1088 (2.09 MP) is deliberately just inside it.
NATIVE_MAX_PIXELS = 2_100_000


@dataclass(frozen=True)
class SizePreset:
    name: str
    width: int
    height: int
    label: str          # human-facing, shown in the Playground dropdown
    ratio: str


# Ordered for display: native sizes first, then the upscale-backed ones.
PRESETS: Tuple[SizePreset, ...] = (
    SizePreset("square", 1024, 1024, "Square", "1:1"),
    SizePreset("portrait", 832, 1216, "Portrait", "2:3"),
    SizePreset("landscape", 1216, 832, "Landscape", "3:2"),
    SizePreset("widescreen", 1344, 768, "Widescreen", "16:9"),
    SizePreset("tall", 768, 1344, "Tall / phone", "9:16"),
    SizePreset("ultrawide", 1536, 640, "Ultrawide", "21:9"),
    SizePreset("square_hd", 1440, 1440, "Square HD", "1:1"),
    SizePreset("hd", 1920, 1088, "HD", "16:9"),
    SizePreset("square_2k", 2048, 2048, "Square 2K", "1:1"),
    SizePreset("4k", 3840, 2160, "4K UHD", "16:9"),
)

_BY_NAME: Dict[str, SizePreset] = {p.name: p for p in PRESETS}

_WXH_RE = re.compile(r"^\s*(\d{2,5})\s*[x×]\s*(\d{2,5})\s*$", re.IGNORECASE)

DEFAULT_SIZE = "1024x1024"


class ImageSizeError(ValueError):
    """The requested size cannot be served — message is caller-facing."""


def _nearest_valid(value: int) -> int:
    """Nearest in-range multiple of :data:`DIMENSION_MULTIPLE`."""
    snapped = int(round(value / DIMENSION_MULTIPLE)) * DIMENSION_MULTIPLE
    return max(MIN_DIMENSION, min(MAX_DIMENSION, snapped))


def parse_size(size: Optional[str]) -> Tuple[int, int]:
    """Resolve a preset name or ``"WxH"`` string to ``(width, height)``.

    Raises :class:`ImageSizeError` with an actionable message — including the
    nearest valid pair — rather than silently snapping. A caller that asked for
    1920x1080 and got 1920x1088 back with no warning would reasonably call that
    a bug, so we make them ask for what they will actually receive.
    """
    if not size:
        size = DEFAULT_SIZE
    key = size.strip().lower()
    preset = _BY_NAME.get(key)
    if preset is not None:
        return preset.width, preset.height

    m = _WXH_RE.match(size)
    if not m:
        raise ImageSizeError(
            f"unrecognised size {size!r} — use WIDTHxHEIGHT (e.g. '1024x1024') "
            f"or one of: {', '.join(p.name for p in PRESETS)}"
        )
    width, height = int(m.group(1)), int(m.group(2))

    for label, value in (("width", width), ("height", height)):
        if value < MIN_DIMENSION or value > MAX_DIMENSION:
            raise ImageSizeError(
                f"{label} {value} is out of range "
                f"({MIN_DIMENSION}-{MAX_DIMENSION})"
            )
    if width % DIMENSION_MULTIPLE or height % DIMENSION_MULTIPLE:
        raise ImageSizeError(
            f"size {width}x{height} is not a multiple of {DIMENSION_MULTIPLE} — "
            f"FLUX needs that for clean edges. Nearest valid: "
            f"{_nearest_valid(width)}x{_nearest_valid(height)}"
            + (" (note 1080 is not a multiple of 16 — 1088 is the 16:9 HD height)"
               if 1080 in (width, height) else "")
        )
    return width, height


def needs_upscale(width: int, height: int) -> bool:
    """True when this size must be generated smaller and then upscaled."""
    return width * height > NATIVE_MAX_PIXELS


def native_source_size(width: int, height: int) -> Tuple[int, int]:
    """The size to actually sample at, for a requested ``width`` x ``height``.

    Returns the target unchanged when it is already native-safe. Otherwise
    scales it down to fit :data:`NATIVE_MAX_PIXELS` **preserving the aspect
    ratio** — matching the ratio is what preserves composition through the
    upscale; the scale factor itself is free.
    """
    if not needs_upscale(width, height):
        return width, height
    scale = math.sqrt(NATIVE_MAX_PIXELS / (width * height))
    src_w = _nearest_valid(width * scale)
    src_h = _nearest_valid(height * scale)
    # Snapping each edge independently can nudge back over the ceiling; step
    # down one grid unit on the longer edge until it fits.
    while src_w * src_h > NATIVE_MAX_PIXELS:
        if src_w >= src_h:
            src_w -= DIMENSION_MULTIPLE
        else:
            src_h -= DIMENSION_MULTIPLE
    return src_w, src_h


def preset_payload() -> List[Dict[str, object]]:
    """Preset table for the admin UI — the dropdown's single source of truth."""
    return [
        {
            "name": p.name,
            "width": p.width,
            "height": p.height,
            "label": p.label,
            "ratio": p.ratio,
            "megapixels": round(p.width * p.height / 1_000_000, 1),
            "upscaled": needs_upscale(p.width, p.height),
        }
        for p in PRESETS
    ]
