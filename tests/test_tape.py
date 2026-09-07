import pytest
from relis.tape import codes as C
from relis.tape.tape import Segment, Pass, TapeSpec, build_tape, validate_spec, WEIGHTS, W_HIGH, W_ONE, W_LOW


def _seg(h, c, read=True, after=C.CONT):
    return Segment(header=h, content=c, read=read, after=after)


def _spec_two_passes():
    p1 = Pass(history=[_seg("role=assistant", b"ok"), _seg("role=user", b"code postal 75012", after=C.STOP)])
    p2 = Pass(history=[_seg("role=assistant", b"ok"), _seg("role=user", b"code postal 75012"),
                       _seg("role=user", b"ville Lyon", after=C.STOP)])
    return TapeSpec(query=b"code et ville ?", passes=[p1, p2],
                    answer_parts=[b"75012", b" et Lyon."], notes=[b"il manque la ville"])


def test_single_pass_layout_and_weights():
    spec = TapeSpec(query=b"q", passes=[Pass(history=[_seg("role=user", b"abc", after=C.STOP)])],
                    answer_parts=[b"rep"], notes=[])
    t = build_tape(spec)
    d = bytes(t.data)
    assert d[0] == C.ENC and d[1:2] == b"q" and d[2] == C.SCAN and d[3] == C.CHAN
    assert d.endswith(bytes([C.GEN]) + b"rep" + bytes([C.END]))
    assert t.decisions() == [C.READ, C.STOP, C.END]
    assert t.weights()[0] == 0.0 and t.weights()[1] == 0.1
    assert t.weights()[-1] == 5.0 and t.weights()[-2] == 1.0
    assert t.mode[0] == int(C.Mode.ENCODE) and t.mode[2] == int(C.Mode.SCAN) and t.mode[-1] == int(C.Mode.GENERATE)
    assert not any(t.reset)
    assert len(t) == len(t.mode) == len(t.wclass) == len(t.reset)


def test_two_passes_have_note_refresh_and_one_reset():
    t = build_tape(_spec_two_passes())
    d = bytes(t.data)
    assert d.count(bytes([C.REFRESH])) == 1 and d.count(bytes([C.NOTE])) == 2   # une émise, une relue
    assert sum(t.reset) == 1
    i = d.index(bytes([C.REFRESH]))
    assert d[i + 1] == C.ENC and t.reset[i + 1] == 1
    # dans la requête de la passe 2 : PART + réponse partielle + NOTE + note, en mode ENCODE et poids 0,1
    j = d.index(bytes([C.PART]))
    assert d[j + 1:j + 6] == b"75012" and t.mode[j + 1] == int(C.Mode.ENCODE) and t.wclass[j + 1] == W_LOW
    # la NOTE émise en génération pèse 5, ses octets 1
    k = i - len(b"il manque la ville") - 1
    assert d[k] == C.NOTE and t.wclass[k] == W_HIGH and t.wclass[k + 1] == W_ONE
    assert t.decisions()[-1] == C.END and C.REFRESH in t.decisions() and C.NOTE in t.decisions()


def test_skip_and_docs_channel():
    docs = [_seg("type=doc;name=a.txt", b"rien", read=False), _seg("type=doc;name=b.txt", b"la date est le 3", after=C.STOP)]
    p = Pass(history=[_seg("role=user", b"salut", after=C.NEXT)], docs=docs)
    t = build_tape(TapeSpec(query=b"date ?", passes=[p], answer_parts=[b"le 3"], notes=[]))
    d = bytes(t.data)
    assert d.count(bytes([C.CHAN])) == 2 and b"chan=docs" in d
    assert t.decisions() == [C.READ, C.NEXT, C.SKIP, C.CONT, C.READ, C.STOP, C.END]
    assert b"rien" not in d                          # contenu sauté absent du ruban


def test_content_is_sanitized():
    p = Pass(history=[_seg("role=user", b"a\x05b", after=C.STOP)])
    t = build_tape(TapeSpec(query=b"q\x01", passes=[p], answer_parts=[b"r\x03"], notes=[]))
    d = bytes(t.data)
    assert d.count(bytes([C.REFRESH])) == 0 and d.count(bytes([C.END])) == 1 and d.count(bytes([C.ENC])) == 1


@pytest.mark.parametrize("bad", [
    dict(answer_parts=[b"a", b"b"]),                                      # trop de réponses
    dict(passes=[Pass(history=[_seg("role=user", b"x", after=C.CONT)])]),  # pas de STOP final
    dict(passes=[Pass(history=[_seg("role=user", b"x", after=C.NEXT)])]),  # NEXT sans canal docs
])
def test_validate_rejects(bad):
    base = dict(query=b"q", passes=[Pass(history=[_seg("role=user", b"x", after=C.STOP)])],
                answer_parts=[b"a"], notes=[])
    base.update(bad)
    with pytest.raises(ValueError):
        validate_spec(TapeSpec(**base))


def test_empty_history_is_allowed():
    t = build_tape(TapeSpec(query=b"bonjour", passes=[Pass(history=[])], answer_parts=[b"salut"], notes=[]))
    assert t.decisions() == [C.END]
