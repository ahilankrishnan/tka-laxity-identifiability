"""Record the run environment and the exact source code of a pipeline run (results/environment.txt).

Writes the date, Python, platform, and package versions, and a SHA-256 digest of each listed source/input file
(the .py/.sh/.bat files directly in knee/ and scripts/, manuscript/tab_bundles_template.txt, and requirements.txt;
line endings normalized, so Windows and Linux copies of the same files give the same digest), plus one combined
digest.  Matching combined digests show that these listed files were identical; they do not cover the Python
installation itself, so also compare the recorded package versions.
"""
import sys
import hashlib
import platform
import datetime
from pathlib import Path
import numpy
import scipy
import matplotlib

ROOT = Path(__file__).resolve().parents[1]


def source_digests() -> list[tuple[str, str]]:
    """(relative path, SHA-256 of the file with normalized line endings) for every pipeline source file."""
    files = [p for d in ("knee", "scripts") for p in (ROOT / d).iterdir()
             if p.is_file() and p.suffix in (".py", ".sh", ".bat")]
    files = sorted(files + [ROOT / "manuscript" / "tab_bundles_template.txt", ROOT / "requirements.txt"])
    return [(p.relative_to(ROOT).as_posix(), hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest())
            for p in files]


def main() -> None:
    """Write results/environment.txt."""
    dig = source_digests()
    total = hashlib.sha256("".join(f"{n} {h}\n" for n, h in dig).encode()).hexdigest()
    lines = [f"date {datetime.datetime.now().isoformat(timespec='seconds')}", sys.version, platform.platform(),
             f"numpy {numpy.__version__} scipy {scipy.__version__} matplotlib {matplotlib.__version__}",
             f"source digest {total}"] + [f"  {h}  {n}" for n, h in dig]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()
