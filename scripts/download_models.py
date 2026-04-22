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
import sys
from pathlib import Path
from typing import List

from huggingface_hub import hf_hub_download, list_repo_files

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_registry import Model, enabled_models  # noqa: E402


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
    if model.backend not in ("openai", "whisper") or not model.hf_repo:
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
    print(f"downloading {model.id} -> {target_dir} ({len(matches)} file(s))")
    for repo_file in matches:
        print(f"  fetching {repo_file}")
        local_path = hf_hub_download(
            repo_id=model.hf_repo,
            filename=repo_file,
            local_dir=str(target_dir),
        )
        results.append(Path(local_path))
        print(f"    -> {local_path}")

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

    return results


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="model id to download (default: all enabled)")
    p.add_argument("--list", action="store_true", help="print plan and exit")
    args = p.parse_args(argv)

    candidates = [m for m in enabled_models()
                  if m.backend in ("openai", "whisper") and m.hf_repo]
    if args.only:
        candidates = [m for m in candidates if m.id == args.only]
        if not candidates:
            print(f"model {args.only!r} not found / not enabled on this host", file=sys.stderr)
            return 2

    if not candidates:
        print("nothing to download (no local models enabled for this host)")
        return 0

    for m in candidates:
        files = _files_for(m)
        total = len(files)
        print(f"- {m.id} ({m.display_name}) from {m.hf_repo} -- {total} file(s)")
        for f in files:
            print(f"    {f}")

    if args.list:
        return 0

    for m in candidates:
        download_one(m.id)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
