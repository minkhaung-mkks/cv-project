#!/usr/bin/env python3
"""Create the local Python environment needed by the final workflow."""

from pathlib import Path
import hashlib
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".venv"
CHECKPOINT = ROOT / "models" / "sam2.1_hiera_small.pt"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
CHECKPOINT_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
BRUSH_ROOT = ROOT / "vendor" / "brush_release"
BRUSH_ARCHIVE = BRUSH_ROOT / "brush.tar.xz"
BRUSH_APP = BRUSH_ROOT / "brush-app-aarch64-apple-darwin" / "brush_app"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_large_tools() -> None:
    if not BRUSH_APP.exists():
        if not BRUSH_ARCHIVE.exists():
            raise FileNotFoundError(f"Brush archive is missing: {BRUSH_ARCHIVE}")
        with tarfile.open(BRUSH_ARCHIVE) as archive:
            archive.extractall(BRUSH_ROOT, filter="data")
        BRUSH_APP.chmod(0o755)

    if not CHECKPOINT.exists():
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        partial = CHECKPOINT.with_suffix(".pt.part")
        print("Downloading the SAM 2.1 checkpoint...")
        urllib.request.urlretrieve(CHECKPOINT_URL, partial)
        if sha256(partial) != CHECKPOINT_SHA256:
            partial.unlink(missing_ok=True)
            raise RuntimeError("The downloaded SAM checkpoint did not match the expected checksum")
        partial.rename(CHECKPOINT)


def main() -> None:
    prepare_large_tools()
    subprocess.run([sys.executable, "-m", "venv", str(ENV)], check=True)
    python = ENV / "bin" / "python"
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements-macos.txt")], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "git+https://github.com/facebookresearch/sam2.git"], check=True)
    print(f"Setup complete. Run {ROOT / 'run_workflow.py'} with {python}")


if __name__ == "__main__":
    main()
