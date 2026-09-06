"""Shards d'octets pour le pré-entraînement (spec §6.1) : documents concaténés,
chaque document précédé de SEG + en-tête minimal + HDR."""
from relis.tape import codes as C


def encode_document(content: bytes, header: str) -> bytes:
    # l'en-tête vient de la source : il peut contenir des octets de contrôle du ruban
    head = C.sanitize(header.encode("utf-8", errors="replace"))
    return bytes([C.SEG]) + head + bytes([C.HDR]) + C.sanitize(content)


class ShardWriter:
    def __init__(self, path: str):
        self._f = open(path, "wb")
        self.total_bytes = 0

    def add(self, content: bytes, header: str) -> int:
        data = encode_document(content, header)
        self._f.write(data)
        self.total_bytes += len(data)
        return len(data)

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
