#!/usr/bin/env python3
"""Regression for exact absolute run paths with mjlab's regex resolver (CPU)."""
from pathlib import Path
from tempfile import TemporaryDirectory

from instinct_mj.scripts.instinct_rl.train import _resolve_resume_checkpoint


def main():
    with TemporaryDirectory(prefix="acp-check-resume-") as directory:
        root = Path(directory)
        for name in ("stage1", "stage1_smoke_v1", "stage1+profile", "stage11profile"):
            run = root / name
            run.mkdir()
            (run / "model_30000.pt").write_bytes(name.encode())
        log_dir = root / "new-run"
        for name in ("stage1", "stage1+profile"):
            expected = root / name / "model_30000.pt"
            actual = _resolve_resume_checkpoint(log_dir, str(expected.parent), "model_30000.pt")
            assert actual == expected, (actual, expected)
        # Relative run expressions retain the CLI's documented regex semantics.
        actual = _resolve_resume_checkpoint(log_dir, "stage1_smoke.*", "model_.*")
        assert actual == root / "stage1_smoke_v1/model_30000.pt"
        # An absent exact directory must not silently fall back to a sibling.
        try:
            _resolve_resume_checkpoint(log_dir, str(root / "stage"), "model_.*")
        except ValueError:
            pass
        else:
            raise AssertionError("missing exact directory accepted a prefix sibling")
    print("RESUME_PATH_REGRESSION_PASS")


if __name__ == "__main__":
    main()
