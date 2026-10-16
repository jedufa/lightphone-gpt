"""Exclusive process ownership for mutating bridge runtimes."""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
import stat


class RuntimeLockError(RuntimeError):
    """The state directory is already owned or its lock file is unsafe."""


class RuntimeLock:
    """A nonblocking advisory lock held for the lifetime of a runtime."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._descriptor: int | None = None

    def acquire(self) -> "RuntimeLock":
        if self._descriptor is not None:
            return self
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as error:
            raise RuntimeLockError("runtime lock file is unavailable or unsafe") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeLockError("runtime lock path is not a regular file")
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    raise RuntimeLockError(
                        "another lightphone-gpt runtime owns this state directory"
                    ) from error
                raise
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def close(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "RuntimeLock":
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.close()
