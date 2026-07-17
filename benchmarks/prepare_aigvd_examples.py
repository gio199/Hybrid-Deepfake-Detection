#!/usr/bin/env python
"""Prepare examples 4-6 from the official AIGVDBench release.

The examples use the same prompt-aligned test item in three benchmark tracks:

  example4/aigvd_real_kitchen.mp4
  example5/aigvd_hunyuanvideo_t2v_kitchen.mp4
  example6/aigvd_svd_i2v_kitchen.mp4

AIGVDBench is large, so this script reads ZIP central directories and the
selected members with HTTP range requests. It does not download the complete
378 GB dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional


DATASET_NAME = "AIGVDBench"
DATASET_PAGE = "https://huggingface.co/datasets/AIGVDBench/AIGVDBench"
PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2026/html/"
    "Ma_Your_One-Stop_Solution_for_AI-Generated_Video_Detection_CVPR_2026_paper.html"
)
REPOSITORY_URL = "https://github.com/LongMa-2025/AIGVDBench"
DATASET_LICENSE = "CC BY 4.0"
DATASET_REVISION = "bf48acaa4920990af3dd2a511ea88e63354df305"

SAMPLE_ID = "IjW3jibCCmw_16_574to774.mp4"
SAMPLE_PROMPT = (
    "A woman and a young girl are in a kitchen, preparing a chocolate cake. "
    "The scene is casual and homey, with both people engaged in baking."
)

EXAMPLES = (
    {
        "id": "example4",
        "label": "real",
        "task": "Real",
        "generator": "camera-captured source",
        "group": "aigvd_real",
        "archive_path": "AIGVDBench/Real/Real.zip",
        "filename": "aigvd_real_kitchen.mp4",
    },
    {
        "id": "example5",
        "label": "generated",
        "task": "T2V",
        "generator": "HunyuanVideo",
        "group": "aigvd_t2v",
        "archive_path": "AIGVDBench/OpenSource/T2V/HunyuanVideo.zip",
        "filename": "aigvd_hunyuanvideo_t2v_kitchen.mp4",
    },
    {
        "id": "example6",
        "label": "generated",
        "task": "I2V",
        "generator": "SVD",
        "group": "aigvd_i2v",
        "archive_path": "AIGVDBench/OpenSource/I2V/SVD.zip",
        "filename": "aigvd_svd_i2v_kitchen.mp4",
    },
)

_USER_AGENT = "AIGVDBenchSelectiveExample/1.0"
_RANGE_BLOCK_BYTES = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_url(archive_path: str) -> str:
    quoted_path = urllib.parse.quote(archive_path, safe="/")
    return (
        f"{DATASET_PAGE}/resolve/{DATASET_REVISION}/{quoted_path}"
        "?download=true"
    )


class HTTPRangeReader(io.BufferedIOBase):
    """Seekable read-only stream backed by HTTP byte-range requests."""

    def __init__(self, url: str, block_size: int = _RANGE_BLOCK_BYTES):
        self.url = url
        self.block_size = block_size
        self._position = 0
        self._cache_start = 0
        self._cache = b""

        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
        if content_length is None:
            raise RuntimeError(f"Server did not provide Content-Length for {url}")
        self.length = int(content_length)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self._position = min(position, self.length)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self._position >= self.length or size == 0:
            return b""
        if size is None or size < 0:
            size = self.length - self._position
        requested_end = min(self._position + size, self.length)

        cache_end = self._cache_start + len(self._cache)
        if self._cache_start <= self._position and requested_end <= cache_end:
            relative_start = self._position - self._cache_start
            relative_end = requested_end - self._cache_start
            result = self._cache[relative_start:relative_end]
            self._position = requested_end
            return result

        fetch_start = self._position
        fetch_end = min(
            self.length - 1,
            max(requested_end - 1, fetch_start + self.block_size - 1),
        )
        request = urllib.request.Request(
            self.url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept-Encoding": "identity",
                "Range": f"bytes={fetch_start}-{fetch_end}",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            status = getattr(response, "status", response.getcode())
            content_range = response.headers.get("Content-Range", "")
            payload = response.read()
        if status != 206:
            raise RuntimeError(f"Server ignored byte-range request for {self.url}")
        expected_prefix = f"bytes {fetch_start}-"
        if not content_range.startswith(expected_prefix):
            raise RuntimeError(
                f"Unexpected Content-Range {content_range!r}; expected {expected_prefix!r}"
            )

        self._cache_start = fetch_start
        self._cache = payload
        result_length = requested_end - fetch_start
        result = payload[:result_length]
        if len(result) != result_length:
            raise EOFError(
                f"Short range response for {self.url}: expected {result_length}, "
                f"received {len(result)}"
            )
        self._position = requested_end
        return result


def _find_member(archive: zipfile.ZipFile, sample_id: str) -> zipfile.ZipInfo:
    matches = [
        info
        for info in archive.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).name == sample_id
    ]
    if len(matches) != 1:
        rendered = ", ".join(info.filename for info in matches) or "none"
        raise RuntimeError(
            f"Expected exactly one {sample_id!r} member, found {len(matches)}: {rendered}"
        )
    return matches[0]


def _verified_local_output(
    video_path: Path,
    metadata_path: Path,
    example: Dict[str, str],
) -> bool:
    if not video_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        metadata.get("dataset_revision") == DATASET_REVISION
        and metadata.get("sample_id") == SAMPLE_ID
        and metadata.get("archive_path") == example["archive_path"]
        and metadata.get("video_sha256") == _sha256(video_path)
    )


def _extract_example(
    project_root: Path,
    example: Dict[str, str],
    force: bool,
    offline: bool,
) -> Path:
    example_dir = project_root / example["id"]
    video_path = example_dir / example["filename"]
    metadata_path = example_dir / "source_metadata.json"
    if not force and _verified_local_output(video_path, metadata_path, example):
        return video_path
    if offline:
        raise FileNotFoundError(
            f"Offline mode requested but no verified local output exists: {video_path}"
        )

    example_dir.mkdir(parents=True, exist_ok=True)
    archive_url = _archive_url(example["archive_path"])
    stream = HTTPRangeReader(archive_url)
    temporary = video_path.with_suffix(video_path.suffix + ".part")
    member: Optional[zipfile.ZipInfo] = None
    try:
        with zipfile.ZipFile(stream) as archive:
            member = _find_member(archive, SAMPLE_ID)
            with archive.open(member) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        os.replace(temporary, video_path)
    finally:
        temporary.unlink(missing_ok=True)

    if member is None:
        raise RuntimeError(f"Could not extract {SAMPLE_ID} from {archive_url}")

    metadata = {
        "example_id": example["id"],
        "label": example["label"],
        "group": example["group"],
        "task": example["task"],
        "generator": example["generator"],
        "source_dataset": DATASET_NAME,
        "dataset_page": DATASET_PAGE,
        "paper_url": PAPER_URL,
        "repository_url": REPOSITORY_URL,
        "dataset_license": DATASET_LICENSE,
        "dataset_revision": DATASET_REVISION,
        "archive_path": example["archive_path"],
        "archive_url": archive_url,
        "archive_member": member.filename,
        "archive_member_crc32": f"{member.CRC:08x}",
        "archive_member_bytes": member.file_size,
        "sample_id": SAMPLE_ID,
        "sample_prompt": SAMPLE_PROMPT,
        "video_path": video_path.name,
        "video_sha256": _sha256(video_path),
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "usage_notice": (
            "AIGVDBench is licensed CC BY 4.0. Preserve attribution and review "
            "the dataset card before redistributing benchmark media."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return video_path


def prepare_examples(
    project_root: Path,
    force: bool = False,
    offline: bool = False,
) -> Iterable[Path]:
    for example in EXAMPLES:
        yield _extract_example(project_root, example, force=force, offline=offline)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rewrite every output.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require already verified outputs; do not access the network.",
    )
    args = parser.parse_args()

    for output in prepare_examples(project_root, force=args.force, offline=args.offline):
        print(output.relative_to(project_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
