from relis.tape.headers import format_header, parse_header


def test_roundtrip_simple():
    h = format_header({"role": "user", "t": "2026-09-07T10:00"})
    assert h == "role=user;t=2026-09-07T10:00"
    assert parse_header(h) == {"role": "user", "t": "2026-09-07T10:00"}


def test_values_are_escaped_and_sanitized():
    h = format_header({"type": "doc", "name": "a=b;c\x01d.txt"})
    assert ";" not in h.split("name=")[1] and "=" not in h.split("name=")[1]
    assert "\x01" not in h
    assert parse_header(h)["type"] == "doc"


def test_header_is_truncated_to_valid_utf8():
    h = format_header({"name": "é" * 200})
    assert len(h.encode("utf-8")) <= 120
    h.encode("utf-8").decode("utf-8")   # ne lève pas


def test_parse_ignores_malformed_parts():
    assert parse_header("a=1;;b;c=2=3") == {"a": "1", "c": "2=3"}
