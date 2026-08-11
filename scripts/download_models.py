"""Download GGUF files for each enabled local model into models/.

Reads config/models.yaml via src.model_registry. Only models enabled for
the active host profile are downloaded -- the Mac mini skips GLM's ~60 GB
blob because its host row doesn't enable it.

Usage:
    python scripts/download_models.py            # every enabled openai model
    python scripts/download_models.py --only qwen
    python scripts/download_models.py --list     # show what would be pulled
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)

from huggingface_hub import hf_hub_download, list_repo_files

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_registry import SPAWNABLE_BACKENDS, Model, enabled_models  # noqa: E402


def _pattern_match(name: str, pattern: str) -> bool:
    """Minimal glob: supports `*` and `/` separators."""
    import fnmatch
    return fnmatch.fnmatch(name, pattern)


def _files_for(model: Model) -> List[str]:
    if not model.hf_repo:
        return []
    every = list_repo_files(model.hf_repo)
    if not model.hf_pattern:
        return [f for f in every if f.lower().endswith(".gguf")]
    return [f for f in every if _pattern_match(f, model.hf_pattern)]


def download_one(model_id: str) -> List[Path]:
    """Fetch every file in the model's HF pattern into models/.

    Files are downloaded into `models/<model.id>/` preserving the repo's
    subdirectory structure when relevant (multi-part GGUFs keep their
    shard filenames next to each other so llama-server can follow the
    chain).
    """
    model = next((m for m in enabled_models() if m.id == model_id), None)
    if model is None:
        raise RuntimeError(f"model {model_id!r} is not enabled on this host")
    if model.backend not in SPAWNABLE_BACKENDS or not model.hf_repo:
        raise RuntimeError(f"model {model_id!r} has no hf_repo; nothing to download")

    matches = _files_for(model)
    if not matches:
        raise RuntimeError(
            f"no files matched pattern {model.hf_pattern!r} in {model.hf_repo}"
        )

    target_path = (PROJECT_ROOT / (model.model_path or "")).resolve()
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    results: List[Path] = []
    log.info("downloading %s -> %s (%d file(s))", model.id, target_dir, len(matches))
    for repo_file in matches:
        log.info("  fetching %s", repo_file)
        local_path = hf_hub_download(
            repo_id=model.hf_repo,
            filename=repo_file,
            local_dir=str(target_dir),
        )
        results.append(Path(local_path))
        log.info("    -> %s", local_path)

    # If the repo stored files in a subdir (e.g. Q4_K_M/*.gguf) and the
    # registry's model_path points at <target_dir>/<shard-file>, move
    # shards up to match.
    expected = target_path
    if not expected.exists():
        for r in results:
            if r.name == expected.name:
                if r != expected:
                    r.rename(expected)
                break
        # Move sibling shards up too, so they stay adjacent.
        if expected.exists():
            for r in results:
                flat = target_dir / r.name
                if flat != r and not flat.exists():
                    r.rename(flat)

    results.extend(_download_extra_weights(model))
    return results


def _download_extra_weights(model: Model) -> List[Path]:
    """Fetch a split-loader row's companion weights (#498).

    ``model_path`` is the primary weight; a FLUX.2 row also needs its text
    encoder and VAE, which live in different repos and different subfolders.
    Each entry names its own ``hf_repo``/``hf_pattern`` and target ``path``,
    and files already present are skipped so re-running is cheap.
    """
    out: List[Path] = []
    for spec in getattr(model, "extra_weights", None) or []:
        repo, pattern, rel = spec.get("hf_repo"), spec.get("hf_pattern"), spec.get("path")
        if not (repo and pattern and rel):
            log.warning("  skipping malformed extra_weights entry on %s: %r",
                        model.id, spec)
            continue
        target = (PROJECT_ROOT / rel).resolve()
        if target.exists():
            log.info("  %s already present (%.2f GB)", target.name,
                     target.stat().st_size / 1e9)
            out.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("  fetching %s (%s) -> %s", pattern, repo, target)
        got = Path(hf_hub_download(repo_id=repo, filename=pattern,
                                   local_dir=str(target.parent)))
        # Repos that nest under split_files/ land in a subtree; flatten to the
        # search-path root ComfyUI actually scans, then prune the emptied
        # subtree so it doesn't sit in the models tree looking meaningful.
        if got != target:
            target.unlink(missing_ok=True)
            got.replace(target)
            _prune_empty_dirs(got.parent, stop_at=target.parent)
        out.append(target)
    return out


def _prune_empty_dirs(leaf: Path, *, stop_at: Path) -> None:
    """Remove now-empty directories from ``leaf`` up to (not including)
    ``stop_at``. Best-effort: a non-empty or busy directory just stops it."""
    current = leaf
    while current != stop_at and stop_at in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="model id to download (default: all enabled)")
    p.add_argument("--list", action="store_true", help="print plan and exit")
    args = p.parse_args(argv)

    candidates = [m for m in enabled_models()
                  if m.backend in SPAWNABLE_BACKENDS and m.hf_repo]
    if args.only:
        candidates = [m for m in candidates if m.id == args.only]
        if not candidates:
            log.error("model %r not found / not enabled on this host", args.only)
            return 2

    if not candidates:
        log.info("nothing to download (no local models enabled for this host)")
        return 0

    for m in candidates:
        files = _files_for(m)
        total = len(files)
        log.info("- %s (%s) from %s -- %d file(s)", m.id, m.display_name, m.hf_repo, total)
        for f in files:
            log.info("    %s", f)

    if args.list:
        return 0

    for m in candidates:
        download_one(m.id)
    log.info("done.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
