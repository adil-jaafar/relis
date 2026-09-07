"""Épisodes synthétiques de lecture et oracle des décisions (spec §6.2).

Chaque épisode est une conversation construite pour que l'on sache exactement
où se trouve l'information nécessaire ; l'oracle en déduit READ/SKIP,
CONT/STOP/NEXT, et la position des REFRESH avec leur NOTE.

Garde-fou de taille : `generate_episode` construit le ruban (via `build_tape`)
et redemande un tirage (même genre, même rng) tant que sa taille dépasse
`MAX_TAPE_BYTES`, jusqu'à `MAX_REDRAWS` fois ; au-delà, il lève `RuntimeError`.
Les plages de remplissage sont choisies pour que ce cas soit rarissime — le
garde-fou est une ceinture de sécurité, pas le mécanisme principal.
"""
import random

from relis.tape import codes as C
from relis.tape.headers import format_header
from relis.tape.tape import Segment, Pass, TapeSpec, build_tape

KINDS = ("fact_recall", "variable_tracking", "what_did_i_say", "doc_lookup", "absent", "two_facts", "long_answer")
KIND_WEIGHTS = (22, 12, 14, 22, 8, 12, 10)
MAX_TAPE_BYTES = 15_000
MAX_REDRAWS = 8

PRENOMS = ["Adil", "Camille", "Nadia", "Julien", "Inès", "Marc", "Sofia", "Karim", "Léa", "Youssef"]
VILLES = ["Lyon", "Paris", "Rabat", "Lille", "Casablanca", "Bordeaux", "Nantes", "Tunis", "Genève", "Montréal"]
OBJETS = ["le rapport", "la facture", "le contrat", "le devis", "la présentation", "le planning"]
SUJETS = ["la réunion", "le budget", "les vacances", "le projet RELIS", "la voiture", "le déménagement",
          "le stage", "la formation", "le serveur", "la thèse"]
FILLERS_USER = ["Merci pour ton aide.", "D'accord, je note.", "Peux-tu me rappeler l'heure {de_sujet} ?",
                "Parlons {de_sujet}.", "Je reviens vers toi demain.", "Qu'en penses-tu ?",
                "J'ai avancé sur {sujet} ce matin.", "On en reparle plus tard."]
FILLERS_ASSISTANT = ["Bien sûr.", "Entendu, je m'en occupe.", "Voici ce que je propose pour {sujet}.",
                     "Je reste disponible.", "C'est noté.", "Très bien, continuons."]

# faits : (phrase de l'utilisateur, question, réponse, libellé pour la NOTE, générateur de valeur)
FACTS = [
    ("Mon code postal est {v}.", "Quel est mon code postal ?", "Votre code postal est {v}.", "votre code postal",
     lambda r: f"{r.randint(10000, 95999)}"),
    ("Je m'appelle {v}.", "Comment je m'appelle ?", "Vous vous appelez {v}.", "votre prénom",
     lambda r: r.choice(PRENOMS)),
    ("J'habite à {v}.", "Où est-ce que j'habite ?", "Vous habitez à {v}.", "votre ville",
     lambda r: r.choice(VILLES)),
    ("Mon rendez-vous est le {v}.", "Quand est mon rendez-vous ?", "Votre rendez-vous est le {v}.",
     "la date de votre rendez-vous", lambda r: f"{r.randint(1, 28)} {r.choice(['mars', 'avril', 'mai', 'juin', 'octobre'])}"),
    ("Le mot de passe du wifi est {v}.", "Quel est le mot de passe du wifi ?", "Le mot de passe du wifi est {v}.",
     "le mot de passe du wifi", lambda r: "".join(r.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(8))),
    ("Mon numéro de dossier est {v}.", "Quel est mon numéro de dossier ?", "Votre numéro de dossier est {v}.",
     "votre numéro de dossier", lambda r: f"D-{r.randint(1000, 9999)}"),
]

DOC_TOPICS = [
    ("contrat", "Le contrat expire le {v}.", "Quand expire le contrat ?", "Le contrat expire le {v}.",
     lambda r: f"{r.randint(1, 28)}/{r.randint(1, 12):02d}/{r.randint(2026, 2029)}"),
    ("facture", "Le montant total de la facture est de {v} euros.", "Quel est le montant de la facture ?",
     "Le montant de la facture est de {v} euros.", lambda r: f"{r.randint(50, 9000)}"),
    ("reunion", "La réunion aura lieu en salle {v}.", "Dans quelle salle a lieu la réunion ?",
     "La réunion a lieu en salle {v}.", lambda r: f"{r.choice('ABCDE')}{r.randint(1, 40)}"),
    ("serveur", "L'adresse du serveur est {v}.", "Quelle est l'adresse du serveur ?", "L'adresse du serveur est {v}.",
     lambda r: f"10.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"),
    ("recette", "Il faut {v} grammes de farine.", "Combien de farine faut-il ?", "Il faut {v} grammes de farine.",
     lambda r: f"{r.randint(100, 900)}"),
]
FILLER_DOC_NAMES = ["notes", "brouillon", "todo", "lecture", "journal", "annexe", "divers", "memo"]


def _stamp(rng, i):
    return f"2026-09-{rng.randint(1, 28):02d}T{rng.randint(8, 19):02d}:{i % 60:02d}"


def _de(sujet: str) -> str:
    """Élision de « de » devant un sujet qui porte déjà son article : « le »/« les »
    -> « du »/« des », « l' » -> « de l' » ; « la » et le reste restent inchangés
    (« de la réunion » ne s'élide pas)."""
    if sujet.startswith("l'"):
        return "de l'" + sujet[2:]
    if sujet.startswith("le "):
        return "du " + sujet[3:]
    if sujet.startswith("les "):
        return "des " + sujet[4:]
    return "de " + sujet


def _fill(rng, tpl):
    """Formate un gabarit de remplissage : tire un sujet et fournit à la fois
    {sujet} (tel quel) et {de_sujet} (élidé) — un gabarit n'utilise que l'un des deux."""
    sujet = rng.choice(SUJETS)
    return tpl.format(sujet=sujet, de_sujet=_de(sujet))


def _filler_turn(rng, role, i):
    tpl = rng.choice(FILLERS_USER if role == "user" else FILLERS_ASSISTANT)
    text = _fill(rng, tpl)
    return Segment(header=format_header({"role": role, "t": _stamp(rng, i)}), content=text.encode("utf-8"))


def _history(rng, n_pairs, injections):
    """Construit n_pairs paires (user, assistant), du plus ANCIEN au plus récent, en injectant
    `injections` = {index_de_tour_user: texte}. Renvoie la liste du plus RÉCENT au plus ancien."""
    turns = []
    for i in range(n_pairs):
        u = _filler_turn(rng, "user", 2 * i)
        if i in injections:
            u = Segment(header=u.header, content=injections[i].encode("utf-8"))
        turns.append(u)
        turns.append(_filler_turn(rng, "assistant", 2 * i + 1))
    return turns[::-1]


def _filler_doc(rng, i):
    name = f"{rng.choice(FILLER_DOC_NAMES)}_{i}.txt"
    body = " ".join(_fill(rng, rng.choice(FILLERS_USER)) for _ in range(rng.randint(2, 25)))
    return Segment(header=format_header({"type": "doc", "name": name, "mime": "text/plain",
                                         "bytes": len(body.encode()), "t": _stamp(rng, i)}),
                   content=body.encode("utf-8"))


def oracle_pass(history_segments, needed_hist, docs, needed_doc, doc_relevant) -> Pass:
    hist = []
    collected = set()
    n = len(history_segments)
    for k, s in enumerate(history_segments):
        collected.add(k)
        if needed_hist and needed_hist <= collected and needed_doc is None:
            after = C.STOP                      # tout ce qu'il faut est lu
        elif k == n - 1:
            after = C.NEXT if docs else C.STOP  # fin de l'historique
        elif not needed_hist and needed_doc is not None:
            after = C.NEXT                      # rien dans l'historique, un document est requis
        else:
            after = C.CONT
        hist.append(Segment(header=s.header, content=s.content, read=True, after=after))
        if after in (C.STOP, C.NEXT):
            break
    if not docs:
        return Pass(history=hist, docs=None)
    if hist and hist[-1].after != C.NEXT:
        return Pass(history=hist, docs=None)
    out_docs = []
    got = False
    m = len(docs)
    for j, d in enumerate(docs):
        read = bool(doc_relevant[j])
        if read and needed_doc is not None and j == needed_doc:
            got = True
        after = C.STOP if (got or j == m - 1) else C.CONT
        out_docs.append(Segment(header=d.header, content=d.content, read=read, after=after))
        if after == C.STOP:
            break
    return Pass(history=hist, docs=out_docs)


def _fact_recall(rng):
    f = rng.choice(FACTS); v = f[4](rng)
    n = rng.randint(3, 14); at = rng.randint(0, n - 1)
    hist = _history(rng, n, {at: f[0].format(v=v)})
    idx = 2 * (n - 1 - at) + 1                        # position dans l'ordre récent -> ancien
    p = oracle_pass(hist, {idx}, [], None, [])
    return TapeSpec(query=f[1].encode(), passes=[p], answer_parts=[f[2].format(v=v).encode()], notes=[])


def _variable_tracking(rng):
    n = rng.randint(3, 10); x = rng.randint(1, 9); ops = {}
    at = sorted(rng.sample(range(n), k=min(n, rng.randint(1, 3))))
    val = x
    ops[at[0]] = f"Note que x vaut {x}."
    for a in at[1:]:
        d = rng.randint(1, 5); val += d
        ops[a] = f"x augmente de {d}."
    hist = _history(rng, n, ops)
    needed = {2 * (n - 1 - a) + 1 for a in at}
    p = oracle_pass(hist, needed, [], None, [])
    return TapeSpec(query=b"Combien vaut x maintenant ?", passes=[p],
                    answer_parts=[f"x vaut {val}.".encode()], notes=[])


def _what_did_i_say(rng):
    sujet = rng.choice(SUJETS); v = rng.choice(["c'est urgent", "c'est reporté", "c'est terminé", "il faut un budget"])
    n = rng.randint(3, 12); at = rng.randint(0, n - 1)
    hist = _history(rng, n, {at: f"À propos {_de(sujet)} : {v}."})
    p = oracle_pass(hist, {2 * (n - 1 - at) + 1}, [], None, [])
    return TapeSpec(query=f"Qu'ai-je dit sur {sujet} ?".encode(), passes=[p],
                    answer_parts=[f"Vous avez dit : « {v} ».".encode()], notes=[])


def _docs_for(rng, topic, sentence, n_docs, noise):
    docs, relevant = [], []
    target = rng.randint(0, n_docs - 1)
    for j in range(n_docs):
        if j == target:
            body = " ".join([_fill(rng, rng.choice(FILLERS_ASSISTANT))
                             for _ in range(rng.randint(2, 40))] + [sentence] +
                            [_fill(rng, rng.choice(FILLERS_USER)) for _ in range(rng.randint(0, 40))])
            name = f"{topic}_{rng.randint(1, 99)}.txt"
            docs.append(Segment(header=format_header({"type": "doc", "name": name, "mime": "text/plain",
                                                      "bytes": len(body.encode()), "t": _stamp(rng, j)}),
                                content=body.encode("utf-8")))
            relevant.append(True)
        else:
            d = _filler_doc(rng, j)
            if noise and rng.random() < 0.10:      # en-tête trompeur : nom pertinent, contenu non
                d = Segment(header=format_header({"type": "doc", "name": f"{topic}_ancien.txt", "mime": "text/plain",
                                                  "bytes": len(d.content), "t": _stamp(rng, j)}), content=d.content)
                relevant.append(True)
            else:
                relevant.append(False)
            docs.append(d)
    return docs, relevant, target


def _doc_lookup(rng):
    topic, tpl, q, a, gen = rng.choice(DOC_TOPICS); v = gen(rng)
    hist = _history(rng, rng.randint(1, 4), {})
    docs, relevant, target = _docs_for(rng, topic, tpl.format(v=v), rng.randint(1, 8), noise=True)
    p = oracle_pass(hist, set(), docs, target, relevant)
    return TapeSpec(query=q.encode(), passes=[p], answer_parts=[a.format(v=v).encode()], notes=[])


def _absent(rng):
    topic, tpl, q, a, gen = rng.choice(DOC_TOPICS)
    hist = _history(rng, rng.randint(1, 5), {})
    n_docs = rng.randint(0, 6)
    docs = [_filler_doc(rng, j) for j in range(n_docs)]
    p = oracle_pass(hist, set(), docs, None, [False] * n_docs)
    return TapeSpec(query=q.encode(), passes=[p],
                    answer_parts=[b"Je ne trouve pas cette information dans notre \xc3\xa9change ni dans les documents."],
                    notes=[])


def _two_facts(rng):
    fa, fb = rng.sample(FACTS, 2); va, vb = fa[4](rng), fb[4](rng)
    n = rng.randint(4, 14)
    at_a, at_b = sorted(rng.sample(range(n), 2), reverse=True)     # A plus récent que B
    hist = _history(rng, n, {at_a: fa[0].format(v=va), at_b: fb[0].format(v=vb)})
    ia, ib = 2 * (n - 1 - at_a) + 1, 2 * (n - 1 - at_b) + 1
    p1 = oracle_pass(hist, {ia}, [], None, [])
    p2 = oracle_pass(hist, {ia, ib}, [], None, [])
    q = f"{fa[1][:-2]} et {fb[3]} ?".encode()
    part1 = fa[2].format(v=va).encode()
    part2 = (" " + fb[2].format(v=vb)).encode()
    return TapeSpec(query=q, passes=[p1, p2], answer_parts=[part1, part2],
                    notes=[f"il manque {fb[3]}".encode()])


def _long_answer(rng):
    f = rng.choice(FACTS); v = f[4](rng)
    n = rng.randint(2, 8); at = rng.randint(0, n - 1)
    hist = _history(rng, n, {at: f[0].format(v=v)})
    p = oracle_pass(hist, {2 * (n - 1 - at) + 1}, [], None, [])
    prefix = f[2].format(v=v) + " Voici le détail :\n"
    items = [f"{i + 1}. Point {i + 1} concernant {rng.choice(SUJETS)} : {rng.choice(FILLERS_ASSISTANT).format(sujet=rng.choice(SUJETS))}"
             for i in range(rng.randint(16, 24))]
    # Garantit au moins 600 octets de réponse : on ajoute des points tant qu'il en manque,
    # pour que L (borné à len(answer) // 2 ci-dessous) puisse toujours tenir dans [256, 1024].
    while len((prefix + "\n".join(items)).encode("utf-8")) < 600:
        i = len(items)
        items.append(f"{i + 1}. Point {i + 1} concernant {rng.choice(SUJETS)} : "
                     f"{rng.choice(FILLERS_ASSISTANT).format(sujet=rng.choice(SUJETS))}")
    answer = (prefix + "\n".join(items)).encode("utf-8")
    L = rng.randint(256, min(1024, len(answer) // 2))
    parts = [answer[i:i + L] for i in range(0, len(answer), L)]
    # `passes = [p] * len(parts)` alias intentionnellement le même objet Pass pour chaque
    # passe : toutes relisent exactement le même historique (seul le préfixe PART/NOTE de
    # la requête change entre les passes, dans build_tape) et Pass n'est jamais modifié en
    # place — l'aliasing est donc sûr et évite de reconstruire un Pass identique n fois.
    passes = [p] * len(parts)
    notes = [b"r\xc3\xa9ponse longue, suite"] * (len(parts) - 1)
    return TapeSpec(query=(f[1] + " Donne-moi tous les détails.").encode("utf-8"), passes=passes,
                    answer_parts=parts, notes=notes)


_GEN = {"fact_recall": _fact_recall, "variable_tracking": _variable_tracking, "what_did_i_say": _what_did_i_say,
        "doc_lookup": _doc_lookup, "absent": _absent, "two_facts": _two_facts, "long_answer": _long_answer}


def generate_episode(rng: random.Random, kind: str | None = None) -> TapeSpec:
    if kind is None:
        kind = rng.choices(KINDS, weights=KIND_WEIGHTS, k=1)[0]
    spec = _GEN[kind](rng)
    if len(build_tape(spec)) <= MAX_TAPE_BYTES:
        return spec
    for _ in range(MAX_REDRAWS):                    # garde-fou dur : redemande un tirage
        spec = _GEN[kind](rng)
        if len(build_tape(spec)) <= MAX_TAPE_BYTES:
            return spec
    raise RuntimeError(f"impossible de générer un épisode '{kind}' sous {MAX_TAPE_BYTES} octets "
                       f"après {1 + MAX_REDRAWS} tirages")


def iter_episodes(seed: int, n: int):
    rng = random.Random(seed)
    for _ in range(n):
        yield generate_episode(rng)
