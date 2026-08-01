"""Anthropic request/response header parity for the Anthropic-shape routes (#461).

The hub is a drop-in for the ``anthropic`` SDK, and the SDK speaks a small
header vocabulary the hub used to swallow whole: ``anthropic-version`` and
``anthropic-beta`` went in and nothing came back. Silence is
indistinguishable from "accepted and honoured", which is the failure mode
worth closing — a caller opting into a beta the hub does not implement
deserves an answer, not a shrug.

What this module decides (the middleware in
:mod:`src.trace_id_middleware` applies it):

- **``anthropic-version``** is echoed back. A request without one is still
  served, and the response carries :data:`DEFAULT_ANTHROPIC_VERSION`. The
  hub does not vary its wire shape by version, so the echo is an
  acknowledgement, not a negotiation.

- **``anthropic-beta``** is *accepted and echoed, never honoured* — the
  chosen half of the issue's either/or, because rejecting unknown values
  would 400 an SDK caller who set a beta the hub simply has no opinion
  about. So the echo cannot be misread as support, every requested beta
  that is not in :data:`IMPLEMENTED_BETAS` also produces an RFC 7234
  ``Warning: 299`` advisory on the same response. :data:`IMPLEMENTED_BETAS`
  is empty today and that is the honest value: the hub implements no
  Anthropic beta feature.

``request-id`` is *not* built here — it is an alias of the trace ID the
X-Trace-Id contract already computes, so it lives with that contract in
:mod:`src.trace_id_middleware`.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, List, Tuple

# Answered when the caller sends no ``anthropic-version``. Matches the
# version the anthropic SDK pins by default.
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

# Anthropic beta features this hub actually implements. Empty on purpose —
# add a value here only when the named beta is genuinely wired up, since
# membership is what suppresses the "not implemented" advisory below.
IMPLEMENTED_BETAS: FrozenSet[str] = frozenset()

# Client-supplied text is echoed into a response header, so cap it and
# strip anything that could split a header or unbalance the quoted Warning
# text. Beta names and version strings are short tokens; 200 is generous.
_MAX_ECHO_LEN = 200
_UNSAFE_ECHO_CHARS = '"\\'

_WARNING_AGENT = "local-llm-hub"


def _sanitize(value: str) -> str:
    """Printable-ASCII, quote-free, length-capped echo of a client string."""
    cleaned = "".join(
        ch for ch in value if " " <= ch <= "~" and ch not in _UNSAFE_ECHO_CHARS
    )
    return cleaned.strip()[:_MAX_ECHO_LEN]


def parse_betas(raw: str) -> List[str]:
    """Split one ``anthropic-beta`` header value into its comma-separated names."""
    return [name for name in (part.strip() for part in raw.split(",")) if name]


def _decode(value: bytes) -> str:
    return value.decode("latin-1", errors="replace")


def parity_headers(
    raw_headers: Iterable[Tuple[bytes, bytes]]
) -> List[Tuple[bytes, bytes]]:
    """Build the Anthropic parity response headers for one request.

    ``raw_headers`` is the ASGI scope's header list (lowercase-ish byte
    name/value pairs). Returns lowercase byte pairs ready to append to an
    ``http.response.start`` message.
    """
    version = ""
    betas: List[str] = []
    for name, value in raw_headers:
        lname = name.lower()
        if lname == b"anthropic-version":
            if not version:
                version = _sanitize(_decode(value))
        elif lname == b"anthropic-beta":
            betas.extend(parse_betas(_sanitize(_decode(value))))

    out: List[Tuple[bytes, bytes]] = [
        (
            b"anthropic-version",
            (version or DEFAULT_ANTHROPIC_VERSION).encode("latin-1"),
        )
    ]
    if betas:
        out.append((b"anthropic-beta", ", ".join(betas).encode("latin-1")))
        unimplemented = [b for b in betas if b not in IMPLEMENTED_BETAS]
        if unimplemented:
            out.append(
                (
                    b"warning",
                    (
                        f'299 {_WARNING_AGENT} "anthropic-beta received but not '
                        f'implemented: {", ".join(unimplemented)}"'
                    ).encode("latin-1"),
                )
            )
    return out
