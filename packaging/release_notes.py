"""Write the GitHub release notes of a version from CHANGELOG.md.

    python packaging/release_notes.py 2.1.0 notes.md
"""
import sys
from pathlib import Path

version = sys.argv[1].lstrip("v")
text = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
sections = text.split("\n## ")
body = next((s.split("\n", 1)[1] for s in sections[1:] if s.startswith(version)), "")
if not body:
    sys.exit(f"version {version} not found in CHANGELOG.md")
out = f"""**PAGANI Toolkit {version}**

**Download:** `PAGANI-Toolkit-{version}-win64.zip` below — unzip and run `PAGANI.exe`
(64-bit Windows 10/11, NVIDIA GPU GTX 10xx or newer with a current driver; no CUDA
installation needed). User guide: `docs/USER_GUIDE.md` in the package; 中文说明见 `README_zh.md`.
Validation and comparison with PAGANI 2.0: `docs/VALIDATION_REPORT.md`.

"""
out += body.strip().replace("](docs/", "](https://github.com/mingruixia/PAGANI-toolkit/blob/master/docs/") + "\n"
Path(sys.argv[2]).write_text(out, encoding="utf-8")
