import zipfile
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent.parent

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def rebuild():
    # 1. colab_backend_only.zip
    backend_zip = ROOT / "colab_backend_only.zip"
    backend_files = [
        "README.md",
        "requirements.txt",
    ]
    for p in sorted((ROOT / "bodyscan_pipeline").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
            backend_files.append(p.relative_to(ROOT).as_posix())

    with zipfile.ZipFile(backend_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(backend_files):
            zf.write(ROOT / rel, rel)
    print(f"Created {backend_zip.name}: {len(backend_files)} files, {backend_zip.stat().st_size:,} bytes")
    print(f"  SHA256: {sha256_of(backend_zip)}")

    # 2. bodyscan_pipeline_bundle.zip
    bundle_zip = ROOT / "bodyscan_pipeline_bundle.zip"
    bundle_files = [
        "README.md",
        "requirements.txt",
        "AlphaScan_Plan2A_Colab.ipynb",
    ]
    for folder in ["bodyscan_pipeline", "frontend_lab", "tests", "sample_ai_images"]:
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                bundle_files.append(p.relative_to(ROOT).as_posix())

    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(bundle_files):
            zf.write(ROOT / rel, rel)
    print(f"Created {bundle_zip.name}: {len(bundle_files)} files, {bundle_zip.stat().st_size:,} bytes")
    print(f"  SHA256: {sha256_of(bundle_zip)}")

if __name__ == "__main__":
    rebuild()
