"""Single-controller lease, extracted from robotskills storage.ControllerLease."""

from pathlib import Path


class ControllerLease:
    def __init__(self, path):
        self.path, self.stream = Path(path).expanduser(), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(self.path, "a+b")
        try:
            import os
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                if not self.stream.read(1):
                    self.stream.write(b"\0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            self.stream = None
            raise RuntimeError(f"another controller owns {self.path}") from None
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
