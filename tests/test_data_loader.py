"""Twin-2K-500 conversion (src/load_twin2k500.py) on hand-made survey JSON, no download, and the
Phase 2 interview conversion (src/load_phase2.py) on a hand-made answers file.
Run from the project root:  python tests/test_data_loader.py   (or pytest)"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from load_twin2k500 import parse_persona_json  # noqa: E402
import load_phase2  # noqa: E402


def _survey(*questions):
    return json.dumps([{"Questions": list(questions)}])


def test_skipped_questions_are_left_out():
    qa = parse_persona_json(_survey(
        {"QuestionType": "MC", "QuestionText": "Favourite season?", "Answers": {"SelectedText": "Winter"}},
        {"QuestionType": "TE", "QuestionText": "Anything else?", "Answers": {}},
        {"QuestionType": "TE", "QuestionText": "Your city?", "Answers": {"Text": "  "}},
        {"QuestionType": "MC", "QuestionText": "Pets?", "Answers": {"SelectedText": []}},
    ))
    assert qa == [{"question": "Favourite season?", "answer": "Winter"}], qa


def test_unanswered_matrix_rows_and_slider_values_are_left_out():
    qa = parse_persona_json(_survey(
        {"QuestionType": "Matrix", "QuestionText": "Agree?", "Rows": ["I plan ahead", "I like crowds"],
         "Answers": {"SelectedText": ["Agree", ""]}},
        {"QuestionType": "Slider", "QuestionText": "How likely", "Statements": ["to vote", "to move"],
         "Answers": {"Values": [80, None]}},
    ))
    assert qa == [{"question": "Agree? -- I plan ahead", "answer": "Agree"},
                  {"question": "How likely -- to vote", "answer": "80"}], qa


def test_malformed_survey_json_gives_nothing_instead_of_crashing():
    for raw in ("not json", None, "{}", '{"Questions": []}', "[1, 2]", '[{"Questions": [null, 3]}]',
                '[{"Questions": [{"QuestionText": null, "Answers": {"Text": "x"}}]}]'):
        assert parse_persona_json(raw) == [], raw


def _phase2(tmp, answers):
    src = Path(tmp) / "phase2_answers.json"
    src.write_text(json.dumps({"consent": {"name": "Test Person"}, "answers": answers}), encoding="utf-8")
    load_phase2.convert(src, out_root=Path(tmp) / "out", pid="p")
    out = Path(tmp) / "out" / "p"
    rows = lambda name: [json.loads(x) for x in (out / name).read_text(encoding="utf-8").splitlines()]  # noqa: E731
    return rows("train.jsonl") + rows("test.jsonl"), rows("baseline.jsonl")


def test_phase2_keeps_a_repeat_whose_original_was_skipped():
    import tempfile
    answers = [{"id": f"q{i}", "section": "What you believe", "question": f"Question {i}?", "answer": f"answer {i}"}
               for i in range(24)]
    answers[3]["answer"] = "  "                                  # the original, skipped
    answers += [{"id": "r3", "repeat_of": "q3", "section": "What you believe",
                 "question": "Question 3, reworded?", "answer": "the only answer to 3"},
                {"id": "r5", "repeat_of": "q5", "section": "What you believe",
                 "question": "Question 5, reworded?", "answer": "answer 5 again"}]
    with tempfile.TemporaryDirectory() as tmp:
        used, baseline = _phase2(tmp, answers)
    texts = [r["answer"] for r in used]
    assert "the only answer to 3" in texts                     # used to be dropped
    assert "answer 5 again" not in texts                       # a real repeat stays out of train/test
    assert [b["reworded_answer"] for b in baseline] == ["answer 5 again"]
    assert len(used) == 24


if __name__ == "__main__":
    test_skipped_questions_are_left_out()
    test_unanswered_matrix_rows_and_slider_values_are_left_out()
    test_malformed_survey_json_gives_nothing_instead_of_crashing()
    test_phase2_keeps_a_repeat_whose_original_was_skipped()
    print("data loader checks passed")
