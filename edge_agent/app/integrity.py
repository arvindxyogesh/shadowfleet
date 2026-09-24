import hashlib
import secrets

_CHUNK_SIZE = 1024 * 1024


class ChecksumMismatch(ValueError):
    pass


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_checksum(path: str, expected_sha256: str) -> None:
    """Raises ChecksumMismatch unless the artifact at `path` hashes to
    `expected_sha256` (NFR-9), so a corrupted or tampered model file is
    rejected before ONNX Runtime ever parses it. Raises OSError if the
    file can't be read at all.
    """
    actual = sha256_file(path)
    if not secrets.compare_digest(actual, expected_sha256.lower()):
        raise ChecksumMismatch(f"sha256 mismatch for {path}: expected {expected_sha256.lower()}, got {actual}")
