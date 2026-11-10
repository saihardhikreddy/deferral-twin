"""
Telugu and Hindi questions, and follow-up questions in the same chat.

Memory and retrieval are in English, so a question is first turned into one standalone English question
("prepare"), the twin answers that through the usual gate, and the answer is turned back into the
question's language ("reply_in"). With Online mode, Gemini does both (one call each way, or none when the
answer is already written in the right language); offline, Hindi uses the local Marian translators
(opus-mt-hi-en, opus-mt-en-hi) and Telugu stays in English with a note.
"""
import re
from pathlib import Path

NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu"}
SCRIPT = {"hi": "Hindi in Devanagari script (not Latin letters)", "te": "Telugu in Telugu script (not Latin letters)"}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
TELUGU = re.compile(r"[ఀ-౿]")
# a short question that leans on the one before it: "and on weekends?", "why?", "what about her?"
FOLLOW_RE = re.compile(r"^\s*(?:and|but|so|also|then|why|how come|what about|how about|and you|really|same)\b|"
                       r"\b(?:it|that|this|those|these|they|them|there|then|he|she|him|her|one)\b", re.I)


LATIN = re.compile(r"[A-Za-z]")


def detect(text):
    """The language whose script holds most of the letters. A single word in another script used
    to decide it: an English answer that mentions "दाल" counted as Hindi, so reply_in left it
    untranslated, and a Hindi question quoting a Telugu place name was taken for Telugu."""
    t = text or ""
    counts = {"te": len(TELUGU.findall(t)), "hi": len(DEVANAGARI.findall(t)), "en": len(LATIN.findall(t))}
    best = max(counts, key=lambda k: (counts[k], k != "en"))   # on a tie, prefer the Indian script
    return best if counts[best] else "en"


# "there" and "that" are often not pointing back at anything: "Are there foods you avoid?" (existential),
# "Do you think that cities are better?" (a conjunction). Those uses are removed before looking for a reference.
NOT_A_REFERENCE_RE = re.compile(r"\b(?:is|are|was|were|isn'?t|aren'?t)\s+there\b(?!\s*[?.!]*$)|"
                                r"\b(?:think|thinks|say|says|said|believe|feel|know|hope|agree|mean|guess)\s+that\b"
                                r"(?!\s*[?.!]*$|\s+(?:is|was|'s|would|will|sounds|seems|way|one|kind|much)\b)",   # "think that is fair": a pronoun
                                re.I)


def is_follow_up(q, history):
    return bool(history) and len(q.split()) <= 9 and bool(FOLLOW_RE.search(NOT_A_REFERENCE_RE.sub(" ", q)))


def history_text(history, n=6):
    rows = [h for h in (history or []) if (h.get("q") or "").strip()][-n:]
    return "\n".join(f"Q: {h['q'].strip()}\nA: {(h.get('a') or '').strip()[:400]}" for h in rows)


class Translator:
    """Local Marian models, loaded on first use."""

    def __init__(self, models_dir):
        self.dirs = {("hi", "en"): Path(models_dir) / "mt" / "opus-mt-hi-en", ("en", "hi"): Path(models_dir) / "mt" / "opus-mt-en-hi"}
        self._m = {}

    def can(self, src, dst):
        d = self.dirs.get((src, dst))
        return bool(d and (d / "config.json").exists())

    def __call__(self, text, src, dst):
        if not self.can(src, dst):
            return None
        if (src, dst) not in self._m:
            from transformers import MarianMTModel, MarianTokenizer
            d = str(self.dirs[(src, dst)])
            self._m[(src, dst)] = (MarianTokenizer.from_pretrained(d), MarianMTModel.from_pretrained(d))
        tok, model = self._m[(src, dst)]
        out = []
        for part in re.split(r"(?<=[.!?।])\s+", text.strip()):           # sentence by sentence keeps it faithful
            if part:
                ids = model.generate(**tok([part], return_tensors="pt", truncation=True, max_length=400), max_new_tokens=400)
                out.append(tok.decode(ids[0], skip_special_tokens=True))
        return " ".join(out)


def prepare(q, history, lang, gemini=None, translator=None):
    """-> (standalone English question, note or None). gemini(system, prompt) -> text, or None offline."""
    follow = is_follow_up(q, history) if lang == "en" else bool(history)
    if lang == "en" and not follow:
        return q, None
    if gemini:
        try:
            ask = (f"Chat so far:\n{history_text(history)}\n\n" if history else "") + f"Last question: {q}"
            out = gemini("Rewrite the last question as one standalone question in English, resolving words like "
                         "'it' or 'that' from the chat. Keep its meaning and person (you/I) exactly. Output only the question.",
                         ask).strip().strip('"')
            if out:
                return out.splitlines()[0], None
        except Exception:
            pass
    text = q
    if lang != "en":
        t = translator(q, lang, "en") if translator else None
        if not t:
            return q, f"{NAMES[lang]} questions need Online mode on this Mac; answering as best it can."
        text = t
    if history and is_follow_up(text, history):                   # offline: lean on the previous question
        prev = (history[-1].get('q') or '').strip()
        prev_lang = detect(prev)
        if prev_lang != "en":                                     # retrieval is English: don't glue Hindi onto it
            prev = (translator(prev, prev_lang, "en") if translator else None) or ""
        text = f"{prev} {text}".strip()
    return text, None


def reply_in(answer, lang, gemini=None, translator=None):
    """-> (answer in lang, note or None). Leaves an answer already in lang alone."""
    if lang == "en" or not answer or detect(answer) == lang:
        return answer, None
    if gemini:
        try:
            out = gemini(f"Translate into natural, everyday {SCRIPT[lang]}, as the speaker would say it. Keep the meaning, "
                         "tone and first person. Output only the translation.", answer).strip()
            if out and detect(out) == lang:
                return out, None
        except Exception:
            pass
    t = translator(answer, "en", lang) if translator else None
    if t:
        return t, None
    return answer, f"Replying in English: {NAMES[lang]} needs Online mode on this Mac."
