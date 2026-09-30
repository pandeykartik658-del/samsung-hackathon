# scripts/fetch_models.py
"""Download and smoke-load the offline perception models so the image runs with no network.

    python scripts/fetch_models.py                 # base.en into $THEME5_MODEL_DIR (default ./models)
    python scripts/fetch_models.py --model tiny.en
    python scripts/fetch_models.py --check         # load from disk only, fail if weights are missing

Whisper weights come from Hugging Face (Systran/faster-whisper-<model>). RapidOCR's
PP-OCR weights ship inside its wheel, so it only needs a warm-up load.
Exit code 0 when every installed backend loads, 1 otherwise. Skips (exit 0) a backend
whose package is not installed, because the agent core degrades without it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def fetch_whisper(model: str, root: Path, check_only: bool) -> bool:
    try:
        from faster_whisper import WhisperModel, download_model
    except ModuleNotFoundError as exc:
        if exc.name != "faster_whisper":
            raise  # a dependency or shared library is missing: a real failure
        print("faster-whisper not installed: skipping ASR weights")
        return True
    t0 = time.perf_counter()
    try:
        if not check_only:
            download_model(model, cache_dir=str(root))
        # Same arguments theme5.multimodal.FasterWhisperTranscriber uses with THEME5_OFFLINE=1.
        WhisperModel(model, device="cpu", compute_type="int8", download_root=str(root),
                     local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - report any failure as a clear build error
        print(f"whisper {model}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
        return False
    print(f"whisper {model}: ok in {time.perf_counter() - t0:.1f}s -> {root}")
    return True


def warm_ocr() -> bool:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ModuleNotFoundError as exc:
        if exc.name != "rapidocr_onnxruntime":
            raise  # a dependency or shared library is missing: a real failure
        print("rapidocr-onnxruntime not installed: skipping OCR warm-up")
        return True
    t0 = time.perf_counter()
    try:
        RapidOCR()
    except Exception as exc:  # noqa: BLE001
        print(f"rapidocr: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
        return False
    print(f"rapidocr: ok in {time.perf_counter() - t0:.1f}s (weights bundled in wheel)")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.environ.get("THEME5_WHISPER_MODEL", "base.en"))
    ap.add_argument("--dir", type=Path, default=Path(os.environ.get("THEME5_MODEL_DIR", "models")))
    ap.add_argument("--check", action="store_true", help="load from disk only, no download")
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true")
    a = ap.parse_args(argv)
    a.dir.mkdir(parents=True, exist_ok=True)
    ok = True
    steps = ([] if a.skip_ocr else [warm_ocr]) + \
        ([] if a.skip_whisper else [lambda: fetch_whisper(a.model, a.dir, a.check)])
    for step in steps:
        try:
            ok = step() and ok
        except ImportError as exc:
            print(f"import failed: {exc}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
