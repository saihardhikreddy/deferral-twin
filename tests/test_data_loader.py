"""Twin-2K-500 conversion (src/load_twin2k500.py) on hand-made survey JSON, no download.
Run from the project root:  python tests/test_data_loader.py   (or pytest)"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from load_twin2k500 import parse_persona_json  # noqa: E402


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


if __name__ == "__main__":
    test_skipped_questions_are_left_out()
    test_unanswered_matrix_rows_and_slider_values_are_left_out()
    test_malformed_survey_json_gives_nothing_instead_of_crashing()
    print("data loader checks passed")
