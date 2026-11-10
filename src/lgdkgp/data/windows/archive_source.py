"""Seekable read-only views of split tar archives and bounded HDF5 members."""
import bisect
import io
import json
from pathlib import Path


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        super().__init__()
        self.paths = [Path(p) for p in paths]
        self.ends = []
        total = 0
        for p in self.paths:
            total += p.stat().st_size
            self.ends.append(total)
        self.size, self.position = total, 0
        self.streams = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset if whence == 2 else None
        if value is None or value < 0: raise ValueError('Invalid archive seek')
        self.position = value
        return value

    def read(self, size=-1):
        remaining = max(0, self.size-self.position)
        remaining = remaining if size is None or size < 0 else min(size, remaining)
        chunks = []
        while remaining:
            i = bisect.bisect_right(self.ends, self.position)
            start = self.ends[i-1] if i else 0
            stream = self.streams.setdefault(i, None)
            if stream is None:
                stream = self.streams[i] = self.paths[i].open('rb')
            stream.seek(self.position-start)
            amount = min(remaining, self.ends[i]-self.position)
            data = stream.read(amount)
            if len(data) != amount: raise EOFError('Archive part changed or truncated')
            chunks.append(data)
            self.position += amount
            remaining -= amount
        return b''.join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data
        return len(data)

    def close(self):
        for stream in self.streams.values():
            if stream: stream.close()
        super().close()


class MemberReader(io.RawIOBase):
    def __init__(self, source, offset, size):
        super().__init__()
        self.source, self.offset, self.size, self.position = source, offset, size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset if whence == 2 else None
        if value is None or value < 0: raise ValueError('Invalid member seek')
        self.position = value
        return value

    def read(self, size=-1):
        size = max(0, self.size-self.position) if size is None or size < 0 else min(size, max(0, self.size-self.position))
        self.source.seek(self.offset+self.position)
        data = self.source.read(size)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data
        return len(data)


def archive_parts(root, environment, split):
    root = Path(root).resolve()
    if split not in ('train', 'validation'): raise ValueError('Training source only')
    if split == 'validation': return [root/'training'/f'{environment}_validation.tar']
    manifest = json.loads((root/'training'/f'{environment}_train.tar.parts.json').read_text())
    paths = []
    for part in manifest['parts']:
        path = (root/part['path']).resolve()
        if not path.is_relative_to(root): raise ValueError('Archive path escapes data root')
        if path.stat().st_size != part['bytes']: raise ValueError(f'Archive size mismatch: {path}')
        paths.append(path)
    if sum(p.stat().st_size for p in paths) != manifest['bytes']: raise ValueError('Archive total size mismatch')
    return paths
