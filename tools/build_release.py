#!/usr/bin/env python3
"""Build byte-stable wheel and source archives from the committed Git tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import zipfile


@dataclass(frozen=True)
class Artifact:
    name: str
    sha256: str
    size: int


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _export_commit(root: Path, commit: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as source:
        members = source.getmembers()
        if any(not _safe_name(member.name) or member.issym() or member.islnk() for member in members):
            raise RuntimeError("committed tree contains an unsafe archive path or link")
        source.extractall(destination, members=members, filter="data")


def _normalize_wheel(source: Path, destination: Path, epoch: int) -> None:
    moment = datetime.fromtimestamp(max(epoch, 315532800), tz=UTC)
    timestamp = (moment.year, moment.month, moment.day, moment.hour, moment.minute, moment.second)
    with zipfile.ZipFile(source, "r") as incoming, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as outgoing:
        for original in sorted(incoming.infolist(), key=lambda item: item.filename):
            if not _safe_name(original.filename) or original.is_dir():
                if not _safe_name(original.filename):
                    raise RuntimeError("wheel contains an unsafe path")
            info = zipfile.ZipInfo(original.filename, timestamp)
            info.compress_type = original.compress_type
            info.external_attr = original.external_attr
            info.create_system = 3
            outgoing.writestr(info, incoming.read(original.filename))


def _normalize_sdist(source: Path, destination: Path, epoch: int) -> None:
    with tarfile.open(source, "r:gz") as incoming:
        members = sorted(incoming.getmembers(), key=lambda item: item.name)
        if any(not _safe_name(member.name) or member.issym() or member.islnk() for member in members):
            raise RuntimeError("source archive contains an unsafe path or link")
        payloads = {
            member.name: incoming.extractfile(member).read()
            for member in members
            if member.isfile()
        }

    with destination.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as outgoing:
                for original in members:
                    info = tarfile.TarInfo(original.name)
                    info.mode = original.mode
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = epoch
                    info.type = original.type
                    info.size = original.size if original.isfile() else 0
                    outgoing.addfile(
                        info,
                        BytesIO(payloads[original.name]) if original.isfile() else None,
                    )


def _digest(path: Path) -> Artifact:
    return Artifact(
        name=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size=path.stat().st_size,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dist")
    args = parser.parse_args()

    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked working tree changes must be committed before release build")
    commit = _git(root, "rev-parse", "HEAD")
    epoch = int(_git(root, "show", "-s", "--format=%ct", commit))
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="lightphone-gpt-release-") as temporary:
        workspace = Path(temporary)
        exported = workspace / "source"
        raw = workspace / "raw"
        exported.mkdir()
        raw.mkdir()
        _export_commit(root, commit, exported)
        environment = dict(os.environ)
        environment["SOURCE_DATE_EPOCH"] = str(epoch)
        subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(raw)],
            cwd=exported,
            env=environment,
            check=True,
        )
        raw_wheels = list(raw.glob("*.whl"))
        raw_sdists = list(raw.glob("*.tar.gz"))
        if len(raw_wheels) != 1 or len(raw_sdists) != 1:
            raise RuntimeError("build did not produce exactly one wheel and one source archive")
        wheel = output / raw_wheels[0].name
        sdist = output / raw_sdists[0].name
        _normalize_wheel(raw_wheels[0], wheel, epoch)
        _normalize_sdist(raw_sdists[0], sdist, epoch)

    artifacts = [_digest(sdist), _digest(wheel)]
    print(
        json.dumps(
            {
                "commit": commit,
                "source_date_epoch": epoch,
                "artifacts": [artifact.__dict__ for artifact in artifacts],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
