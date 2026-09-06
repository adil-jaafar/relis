import pytest
from relis.tape import codes as C


def test_control_table_matches_spec():
    assert (C.ENC, C.SEG, C.END, C.STOP, C.REFRESH, C.NEXT, C.CONT, C.SKIP) == (
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08)
    assert (C.SCAN, C.GEN, C.DEC, C.READ, C.PART, C.RECALL, C.CHAN, C.HDR) == (
        0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x1C, 0x1F)


def test_control_codes_exclude_text_controls():
    assert 0x09 not in C.CONTROL_CODES
    assert 0x0A not in C.CONTROL_CODES
    assert 0x0D not in C.CONTROL_CODES
    assert C.ENC in C.CONTROL_CODES
    assert len(C.CONTROL_CODES) == 32 - 3


def test_sanitize_replaces_control_bytes_keeps_text_controls():
    raw = b"a\x01b\tc\nd\re\x1f"
    out = C.sanitize(raw)
    assert out == b"a\xef\xbf\xbdb\tc\nd\re\xef\xbf\xbd"
    assert all(b not in C.CONTROL_CODES for b in out)


def test_mode_values():
    assert int(C.Mode.ENCODE) == 0
    assert int(C.Mode.SCAN) == 1
    assert int(C.Mode.GENERATE) == 2


@pytest.mark.parametrize("index", [0, 1, 223, 224, 4095])
def test_extrait_address_roundtrip(index):
    addr = C.encode_extrait_address(index)
    assert len(addr) == 2
    assert C.ADDR_BASE <= addr[0] <= C.EXTRAIT_FIRST_MAX
    assert C.ADDR_BASE <= addr[1] <= 0xFF
    assert C.address_length(addr[0]) == 2
    assert C.decode_address(addr) == ("extrait", index)


def test_extrait_address_out_of_range():
    with pytest.raises(ValueError):
        C.encode_extrait_address(C.MAX_EXTRAITS)
    with pytest.raises(ValueError):
        C.encode_extrait_address(-1)


def test_address_length_lexique_reserved():
    assert C.address_length(0x90) == 3
    assert C.address_length(0xFF) == 3
