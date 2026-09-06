from relis.data.shards import encode_document, ShardWriter
from relis.tape import codes as C


def test_encode_document_layout():
    doc = encode_document(b"bonjour\x01monde", "src=test")
    assert doc[0] == C.SEG
    assert doc[1:9] == b"src=test"
    assert doc[9] == C.HDR
    assert doc[10:] == b"bonjour\xef\xbf\xbdmonde"


def test_encode_document_sanitizes_header():
    # un en-tête venu de la source peut contenir un code de contrôle du ruban
    doc = encode_document(b"texte", "src=a\x01b")
    assert doc[0] == C.SEG
    assert doc[1:doc.index(bytes([C.HDR]))] == b"src=a" + C.REPLACEMENT + b"b"
    assert doc.count(bytes([C.HDR])) == 1
    assert doc.count(bytes([C.SEG])) == 1


def test_shard_writer_concatenates(tmp_path):
    p = str(tmp_path / "train.bin")
    w = ShardWriter(p)
    n1 = w.add(b"aaa", "src=a")
    n2 = w.add(b"bb", "src=b")
    w.close()
    with open(p, "rb") as f:
        data = f.read()
    assert len(data) == n1 + n2 == w.total_bytes
    assert data.count(bytes([C.SEG])) == 2
