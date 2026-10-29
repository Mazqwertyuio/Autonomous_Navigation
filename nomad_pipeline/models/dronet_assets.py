"""Pinned official checkpoints. Importing this module needs no ML packages."""
import hashlib
from pathlib import Path

UPSTREAM_REVISION = "98f15cfe2e9c34c3a546358e22e45b3b09f11a79"
UPSTREAM_BASE = (
    "https://raw.githubusercontent.com/pulp-platform/pulp-dronet/"
    f"{UPSTREAM_REVISION}/tiny-pulp-dronet-v3/model"
)
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
CHECKPOINTS = {
    "v3": {
        "filename": "pulp-dronet-v3-resblock-1.0.pth",
        "sha256": "943ab4cbbc28cd711e874559dcb3ba0419a0ae1250dc8c94665f4a46d878ff6e",
    },
    "tiny-v3": {
        "filename": "tiny-pulp-dronet-v3-dw-pw-0.125.pth",
        "sha256": "9775bb595cbc8dba4043b95a31dbec2674789522d339913e67c9655cf5d84729",
    },
}


def verified_checkpoint(variant="v3", weights_path=None):
    """Require the complete official file, including when a path is supplied."""
    if variant not in CHECKPOINTS:
        raise ValueError(f"Unknown DroNet variant {variant!r}; choose {tuple(CHECKPOINTS)}")
    asset = CHECKPOINTS[variant]
    path = Path(weights_path) if weights_path is not None else WEIGHTS_DIR / asset["filename"]
    if not path.is_file():
        raise FileNotFoundError(
            f"DroNet {variant} checkpoint missing: {path}. "
            f"Run: python download_dronet_v3.py --variant {variant}"
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != asset["sha256"]:
        raise ValueError(
            f"DroNet {variant} SHA-256 mismatch: {path}. "
            "The file is corrupt or is not the pinned official checkpoint. "
            "No untrained fallback is allowed."
        )
    return path
