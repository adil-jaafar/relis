import io
from relis.infer.controller import Event
from relis.infer.render import TerminalRenderer, supports_color


def _events():
    return [
        Event("turn_start", text="Quelle est l'adresse ?"),
        Event("channel", channel="history", size=3),
        Event("segment", channel="history", header="role=user;i=0", size=12, action="read"),
        Event("decision", channel="history", text="CONT"),
        Event("segment", channel="history", header="role=user;i=1", size=40, action="skip"),
        Event("decision", channel="history", text="STOP"),
        Event("scan_end", text="stop"),
        *[Event("gen_byte", byte=b) for b in "ok.".encode()],
        Event("end"),
        Event("turn_end", text="ok.", stats={"read": 12, "written": 3, "refresh": 0,
                                             "segments": 2, "available": 52,
                                             "seconds": 0.4, "saved": 0.769}),
    ]


def test_render_shows_segments_decisions_and_counts():
    buf = io.StringIO()
    out = TerminalRenderer(buf, color=False).run(_events())
    s = buf.getvalue()
    assert "role=user;i=0" in s and "role=user;i=1" in s
    assert "lu" in s and "sauté" in s and "STOP" in s
    assert "12" in s and "77" in s              # octets lus et % économisé
    assert out == "ok."


def test_render_without_color_emits_no_escape():
    buf = io.StringIO()
    TerminalRenderer(buf, color=False).run(_events())
    assert "\x1b[" not in buf.getvalue()


def test_render_with_color_emits_escape():
    buf = io.StringIO()
    TerminalRenderer(buf, color=True).run(_events())
    assert "\x1b[" in buf.getvalue()


def test_quiet_prints_only_the_answer():
    buf = io.StringIO()
    TerminalRenderer(buf, color=False, quiet=True).run(_events())
    assert buf.getvalue().strip() == "ok."


def test_note_and_refresh_are_shown():
    buf = io.StringIO()
    ev = [Event("note", text="il manque la ville"), Event("refresh", index=1),
          Event("turn_end", text="", stats={"read": 0, "written": 0, "refresh": 1,
                                            "segments": 0, "available": 0,
                                            "seconds": 0.1, "saved": 0.0})]
    TerminalRenderer(buf, color=False).run(ev)
    assert "il manque la ville" in buf.getvalue()
    assert "relecture" in buf.getvalue()


def test_supports_color_on_plain_stringio():
    assert supports_color(io.StringIO()) is False


def test_streamed_accented_characters_are_not_mojibake():
    buf = io.StringIO()
    events = [Event("gen_byte", byte=b) for b in "café".encode("utf-8")]
    out = TerminalRenderer(buf, color=False).run(events)
    s = buf.getvalue()
    assert "café" in s
    assert "�" not in s
    assert out == "café"


def test_note_after_streaming_starts_on_its_own_line():
    buf = io.StringIO()
    events = [Event("gen_byte", byte=b) for b in "partiel".encode()]
    events.append(Event("note", text="il manque la ville"))
    events.append(Event("turn_end", text="partiel", stats={"read": 0, "written": 7, "refresh": 1,
                                                            "segments": 0, "available": 0,
                                                            "seconds": 0.1, "saved": 0.0}))
    TerminalRenderer(buf, color=False).run(events)
    lines = buf.getvalue().splitlines()
    assert not any("partiel" in line and "il manque" in line for line in lines)
