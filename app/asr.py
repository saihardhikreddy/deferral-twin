"""
Talk mode: speech to text on this Mac with OpenAI's Whisper (small, multilingual: English, Hindi, Telugu
and more), through transformers on Apple's GPU (MPS) when available. Nothing leaves the Mac.

    text, lang = Recognizer(model_dir).transcribe("speech.wav")
"""
import subprocess
import threading
import wave
from pathlib import Path

LANG_TOKENS = {"<|en|>": "en", "<|hi|>": "hi", "<|te|>": "te"}


def deva_to_telugu(text):
    """Whisper-small often writes Telugu speech in Devanagari letters. The two Unicode blocks share one layout
    (both come from ISCII), so shifting each letter by 0x300 gives the Telugu spelling.

    Only the shared part is shifted. Devanagari's precomposed nukta letters (U+0958-095F) sit where Telugu has
    other letters or none: क़ became ౘ (TSA), ख़ became ౙ (DZA), and ज़ stayed in Devanagari in the middle of
    a Telugu word. They are now split into letter + nukta first, and the nukta, which Telugu spelling doesn't
    mark (its sign shows as a dotted circle in most fonts), is dropped. U+0970-097F were added to Devanagari
    later and land on Telugu fraction digits, so they are left alone."""
    import unicodedata
    text = "".join(unicodedata.normalize("NFD", ch) if 0x0958 <= ord(ch) <= 0x095F else ch for ch in text)
    out = []
    for ch in text:
        o = ord(ch)
        if o == 0x093C:                                                   # nukta
            continue
        if 0x0900 <= o <= 0x096F and o not in (0x0964, 0x0965):          # keep the danda punctuation
            t = chr(o + 0x300)
            try:
                unicodedata.name(t)
                out.append(t)
                continue
            except ValueError:
                pass
        out.append(ch)
    return "".join(out)


class Recognizer:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self._m = None
        self._lock = threading.Lock()

    def available(self):
        return (self.dir / "config.json").exists() and (self.dir / "model.safetensors").exists()

    def _load(self):
        if self._m is None:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            dev, dtype = "cpu", torch.float32             # half precision on MPS gives garbage; the CPU is as fast here
                                                          # (about 1 s) and leaves the GPU to the language model
            proc = WhisperProcessor.from_pretrained(str(self.dir))
            model = WhisperForConditionalGeneration.from_pretrained(str(self.dir), dtype=dtype).to(dev).eval()
            self._m = (proc, model, dev, dtype)
        return self._m

    @staticmethod
    def to_wav16k(src, dst):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
                       check=True, stdin=subprocess.DEVNULL, capture_output=True)
        return dst

    def transcribe(self, wav16k, language=None, hint=None):
        """-> (text, language code such as "en", "hi", "te"; or None). Up to 30 s of speech."""
        import numpy as np
        import torch
        with wave.open(str(wav16k), "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size < 16000 * 0.3:
            return "", None
        with self._lock:
            proc, model, dev, dtype = self._load()
            feats = proc(audio[:16000 * 30], sampling_rate=16000, return_tensors="pt").input_features.to(dev, dtype)
            kw = {"task": "transcribe", "max_new_tokens": 200}
            if language:
                kw["language"] = language
            with torch.no_grad():
                lang = language or LANG_TOKENS.get(proc.tokenizer.decode(model.detect_language(feats)[0]))
                ids = model.generate(feats, **kw)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if hint in ("en", "hi", "te"):
            lang = hint                                   # the person said which language they speak
        if lang == "te":
            text = deva_to_telugu(text)
        return text, lang
