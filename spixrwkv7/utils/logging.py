import sys
import contextlib

class TeeStream:
    """A stream that writes to multiple streams (e.g. stdout and a log file)."""
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name):
        if self.streams:
            return getattr(self.streams[0], name)
        raise AttributeError(name)

@contextlib.contextmanager
def redirect_stdout_tee(file_path, mode="w"):
    """Context manager to redirect stdout to both sys.stdout and a file."""
    with open(file_path, mode, encoding="utf-8") as f:
        tee = TeeStream(sys.stdout, f)
        with contextlib.redirect_stdout(tee):
            yield
