"""Deferral honesty and the voice/avatar job queue, without any real model.
Run from the project root:  python tests/test_gate_and_media.py   (or pytest)"""
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import fake_st  # noqa: E402

fake_st.install()

import numpy as np  # noqa: E402


def test_boost_never_reaches_the_gate():
    """HANDOFF.md section 2, item 3: person-stated facts rank higher, but the
    confidence that drives the three tiers stays the raw cosine similarity."""
    import rag_pipeline
    d = Path(tempfile.mkdtemp())
    facts = d / "facts.jsonl"
    facts.write_text("\n".join(json.dumps({"text": t, "source": "interview"}) for t in
                               ["Q: Which city is home?\nA: Hyderabad", "Q: Favourite food?\nA: Biryani"]))
    cfg = rag_pipeline.load_config()
    cfg["paths"] = dict(cfg["paths"], qa_facts_file=str(facts))
    cfg["rag"] = dict(cfg["rag"], index_path=str(d / "i.bin"), correction_boost=0.05)
    st = rag_pipeline.RagStore(cfg)
    st.build()
    st.load()
    q = "robotics lab september switched"
    _, before = st.retrieve(q)
    st.add_fact("I switched to a robotics lab in September and love it.", source="user_note")
    res, conf = st.retrieve(q)
    emb = st.embedder.encode([q, res[0]["text"]], normalize_embeddings=True)
    raw = float(np.dot(emb[0], emb[1]))
    assert res[0]["source"] == "user_note"
    assert abs(conf - raw) < 1e-5 and abs(res[0]["score"] - raw) < 1e-5, (conf, raw)
    assert before < 0.15                                  # it was a refusal before the note


def _wav(path, secs=2.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.zeros(int(16000 * secs), "<i2").tobytes())


class FakeMM:
    def __init__(self, video_secs=1.0):
        self.video_secs, self.spoken, self.rendered = video_secs, 0, 0

    def speak(self, text, lang="en", out_path=None):
        _wav(out_path, 2.0)
        self.spoken += 1
        return out_path

    def animate(self, wav, out_path=None, sharpen=False, on_stage=None):
        on_stage and on_stage("lipsync")
        time.sleep(self.video_secs)
        Path(out_path).write_bytes(b"\0" * 2000)
        self.rendered += 1
        return out_path


def _wait(svc, jid, states=("done", "error"), timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = svc.view(jid)
        if v["state"] in states:
            return v
        time.sleep(0.02)
    raise AssertionError(f"job {jid} stuck in {svc.view(jid)['state']}")


def test_video_never_blocks_voice_and_clips_are_cached():
    from app.media import MediaService
    from app.paths import Paths
    p = Paths(tempfile.mkdtemp()).ensure()
    p.benchmark.write_text(json.dumps({"source": "benchmark", "voice": {"rtf": 1.5},
                                       "lipsync": {"per_audio_second": 12.0}}))
    mm = FakeMM(video_secs=1.5)
    svc = MediaService(p, lambda: mm)
    a = svc.submit("First answer, with a video.", "en", video=True)
    _wait(svc, a["id"], states=("video",))
    v = svc.view(a["id"])
    assert v["estimate_s"] == 24 and "remaining_s" in v          # 12 s per second of speech x 2 s, from the benchmark
    b = svc.submit("Second answer, voice only.", "en", video=False)
    vb = _wait(svc, b["id"])
    assert vb["state"] == "done" and vb["audio"]
    assert svc.view(a["id"])["state"] == "video"                 # the render is still going; voice did not wait
    assert _wait(svc, a["id"])["video"]
    again = svc.submit("First answer, with a video.", "en", video=True)
    assert again["state"] == "done" and again["cached"] and mm.rendered == 1 and mm.spoken == 2
    sharp = svc.submit("First answer, with a video.", "en", video=True, sharpen=True)
    assert sharp["state"] != "done"                              # sharpened clips are cached separately
    assert svc.est.summary()["video_seconds"] == round(12.0 * 40 / 2.6)


def test_latest_correction_wins():
    """Correcting the same question again replaces the earlier correction, also when the log is replayed
    on start-up; the gate still sees only raw similarity."""
    import rag_pipeline
    d = Path(tempfile.mkdtemp())
    facts = d / "facts.jsonl"
    facts.write_text(json.dumps({"text": "Q: Favourite food?\nA: Biryani", "source": "interview"}))
    cfg = rag_pipeline.load_config()
    cfg["paths"] = dict(cfg["paths"], qa_facts_file=str(facts))
    cfg["rag"] = dict(cfg["rag"], index_path=str(d / "i.bin"))
    st = rag_pipeline.RagStore(cfg)
    st.build()
    st.load()
    st.add_fact("Q: What can you do?\nA: i am not an ai", source="user_correction", persist=False)
    st.add_fact("Q: what can you do\nA: I build AI projects", source="user_correction", persist=False)
    res, conf = st.retrieve("What can you do?")
    texts = [r["text"] for r in res]
    assert any("I build AI projects" in t for t in texts) and not any("not an ai" in t for t in texts), texts
    emb = st.embedder.encode(["What can you do?", res[0]["text"]], normalize_embeddings=True)
    assert abs(conf - float(np.dot(emb[0], emb[1]))) < 1e-5


def test_many_recorrections_do_not_crowd_out_other_facts():
    """Each re-correction leaves the old one in the index, marked superseded. A question corrected
    20 times still gets top_k live facts back, including the latest correction."""
    import rag_pipeline
    d = Path(tempfile.mkdtemp())
    facts = d / "facts.jsonl"
    hobbies = ["cricket", "chess", "drawing", "cycling", "trekking", "cooking", "gaming", "reading"]
    facts.write_text("\n".join(json.dumps({"text": f"Q: Childhood hobbies?\nA: {h}", "source": "interview"})
                               for h in hobbies))
    cfg = rag_pipeline.load_config()
    cfg["paths"] = dict(cfg["paths"], qa_facts_file=str(facts))
    cfg["rag"] = dict(cfg["rag"], index_path=str(d / "i.bin"), top_k=8)
    st = rag_pipeline.RagStore(cfg)
    st.build()
    st.load()
    for i in range(20):
        st.add_fact(f"Q: Which hobbies do you enjoy?\nA: hobbies enjoy version {i}",
                    source="user_correction", persist=False)
    res, _ = st.retrieve("Which hobbies do you enjoy?")
    texts = [r["text"] for r in res]
    assert len(res) == 8, texts
    assert any("version 19" in t for t in texts), texts
    assert not any(f"version {i}\n" in t + "\n" for i in range(19) for t in texts), texts


def test_retrieve_from_an_empty_store():
    import rag_pipeline
    d = Path(tempfile.mkdtemp())
    facts = d / "facts.jsonl"
    facts.write_text(json.dumps({"text": "Q: Home?\nA: Hyderabad", "source": "interview"}))
    cfg = rag_pipeline.load_config()
    cfg["paths"] = dict(cfg["paths"], qa_facts_file=str(facts))
    cfg["rag"] = dict(cfg["rag"], index_path=str(d / "i.bin"))
    st = rag_pipeline.RagStore(cfg)
    st.build()
    st.load()
    st.index.reset()
    st.facts = []
    assert st.retrieve("anything") == ([], 0.0)


def test_unmeasured_mac_gives_no_estimate():
    from app.media import Estimates
    assert Estimates(Path(tempfile.mkdtemp()) / "none.json").video_seconds(10, False) is None


def test_phase2_summary_counts_deferrals_but_does_not_score_them():
    """Hard defers lower coverage, never the twin's mean; % of ceiling is twin mean / human retest."""
    from eval_phase2 import summarize
    rows = [{"tier": "confident_answer", "score": 0.8}, {"tier": "confident_answer", "score": 0.6},
            {"tier": "soft_answer", "score": 0.4}, {"tier": "hard_defer", "score": None}]
    s = summarize(rows, [0.9, 0.7], pid="p", n_taught=2)
    assert s["n_test"] == 4 and s["n_corrections_in_memory"] == 2
    assert s["twin_mean"] == 0.6 and s["human_ceiling"] == 0.8 and s["pct_of_ceiling"] == 75.0
    assert s["confident_only_mean"] == 0.7 and s["confident_coverage"] == 0.5
    assert s["tiers"]["hard_defer"] == {"n": 1, "share": 0.25, "mean": None}
    assert s["tiers"]["soft_answer"]["mean"] == 0.4
    # no baseline file, and a twin that deferred everything: no numbers made up
    s = summarize([{"tier": "hard_defer", "score": None}], [])
    assert s["twin_mean"] is None and s["human_ceiling"] is None and s["pct_of_ceiling"] is None
    assert s["confident_coverage"] == 0.0 and summarize([], [])["n_test"] == 0


if __name__ == "__main__":
    test_boost_never_reaches_the_gate()
    test_video_never_blocks_voice_and_clips_are_cached()
    test_latest_correction_wins()
    test_many_recorrections_do_not_crowd_out_other_facts()
    test_retrieve_from_an_empty_store()
    test_unmeasured_mac_gives_no_estimate()
    test_phase2_summary_counts_deferrals_but_does_not_score_them()
    print("gate and media checks passed")
