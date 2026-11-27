"""A student's statement, especially a wrong one in a question, must
not appear in the summary as fact. Synthetic lectures only (tests/attribution_cases.py); a real recording
is a metrics-only sanity check. Same LLM requests as before: the rules are prompt text plus a
deterministic [question] tag on the transcript lines."""
import re

import pytest

import lecture_recorder  # noqa: F401  (lockdown)
from attribution_cases import CASES, doc, judge
from lecture_recorder.config import load_config
from lecture_recorder.summary.llm import LocalLLM
from lecture_recorder.summary.summarize import (
    ATTRIBUTION, CONDENSE_TASK, GLANCE_TASK, MAP_TASK, SYSTEM, _budget, mark_questions, split_questions, summarize_doc,
    transcript_lines,
)


def test_questions_are_tagged_whoever_speaks_and_numbers_stay_whole():
    assert mark_questions("So heavier objects fall faster, right?") == "[question] So heavier objects fall faster, right?"
    assert mark_questions("g is 9.8 m/s. Is that clear? Good.") == "g is 9.8 m/s. [question] Is that clear? Good."
    assert mark_questions("No question here, 3.5 stays.") == "No question here, 3.5 stays."
    lines = transcript_lines(doc("biology", mislabel=True))
    assert "Lecturer: [question] And the midterm is next Friday, right?" in lines[2].text


def test_questions_move_out_of_the_lecture_text_with_markers():
    text = "\n".join(l.text for l in transcript_lines(doc("biology", mislabel=True)))
    out, block = split_questions(text)
    assert "midterm" not in out and "[Q1]" in out and "we'll talk about the schedule later" in out
    assert block.splitlines()[0] == "Q1 [00:00:20]: And the midterm is next Friday, right?"
    assert len(block.splitlines()) == 2 and "[question]" not in out  # both questions moved, nothing else
    assert split_questions("[00:00:01] Lecturer: No questions, 3.5 stays.") == (
        "[00:00:01] Lecturer: No questions, 3.5 stays.", "None")


def test_question_echoes_are_dropped_but_answered_and_lecture_facts_stay():
    from lecture_recorder.summary.summarize import doc_evidence, drop_question_echoes, is_question_echo

    ev = doc_evidence(doc("economics", mislabel=True))
    assert is_question_echo("- The quiz only covers chapter five.", ev)  # unanswered ("We'll see next week")
    assert not is_question_echo("- Rent control is a price ceiling set below equilibrium.", ev)
    answered = {"segments": [
        {"id": 0, "start": 0, "end": 5, "role": "Audience", "text": "Is the exam on Friday?", "words": []},
        {"id": 1, "start": 5, "end": 9, "role": "Main", "text": "Yes, the exam is on Friday in room four.", "words": []}]}
    assert not is_question_echo("- The exam is on Friday in room four.", doc_evidence(answered))  # the reply says so
    notes = "#### Key points\n- The quiz only covers chapter five.\n- Rent control is a ceiling.\n" \
            "#### Questions from students\n- Asked: The quiz only covers chapter five? -> not answered"
    kept, removed = drop_question_echoes(notes, ev)
    assert removed == 1 and "Asked: The quiz" in kept and "Rent control" in kept  # the Q&A line itself stays


def test_timestamps_never_count_as_support_and_a_plain_yes_confirms():
    from lecture_recorder.summary.summarize import doc_evidence, is_question_echo

    seg = lambda i, role, text: {"id": i, "start": 60.0 * i + 10, "end": 60.0 * i + 15, "role": role, "text": text, "words": []}
    unanswered = {"segments": [seg(0, "Main", "Today: enzymes."), seg(1, "Audience", "Is the exam on March 15 at 10?"),
                               seg(2, "Main", "We'll see.")]}
    assert is_question_echo("- Exam on March 15 at 10.", doc_evidence(unanswered))  # stamps' digits are not support
    yes = {"segments": [seg(0, "Audience", "Is the midterm on October 12?"), seg(1, "Main", "Yes.")]}
    assert not is_question_echo("- Midterm on October 12.", doc_evidence(yes))  # a plain yes confirms it


def test_the_at_a_glance_list_is_filtered_too():
    from lecture_recorder.summary.summarize import doc_evidence, reduce_parts

    class StubLLM:  # returns a glance that restates the unanswered question as a fact
        max_prompt_tokens, num_predict = 5000, 2000

        def chat(self, system, user, **kw):
            return "Price floors.\n- The quiz only covers chapter five.\n- Rent control is a price ceiling."

    r = reduce_parts([{"start": 0, "end": 60, "notes": "#### Topics\nfloors"}], StubLLM(), "Minimum wage",
                     doc_evidence(doc("economics", mislabel=True)))
    assert "quiz" not in r["glance"] and "price ceiling" in r["glance"] and r["echoes_removed"] == 1


def test_prompts_carry_the_attribution_rules_within_budget():
    from lecture_recorder.summary.summarize import QUESTIONS_RULE

    assert ATTRIBUTION in MAP_TASK and QUESTIONS_RULE in MAP_TASK and "-> not answered" in QUESTIONS_RULE
    assert "A student asked whether" not in MAP_TASK  # one Q&A format only
    assert "never move them into Key points" in CONDENSE_TASK and "never a student's" in GLANCE_TASK
    llm = LocalLLM(host="http://127.0.0.1:11434", model="qwen3:8b")
    longest = {"part": 999, "parts": 999, "start": "00:00:00", "end": "00:00:00", "questions": ""}
    assert _budget(llm, SYSTEM.format(tag="transcript"), MAP_TASK, **longest) > 3500  # room for the transcript


@pytest.mark.ollama
@pytest.mark.gpu
@pytest.mark.parametrize("case,mislabel", [
    pytest.param(c, m, id=f"{c}-{'all-labeled-lecturer' if m else 'labeled'}")
    for c in sorted(CASES) for m in (False, True)])  # the deflected-date-question case now passes
def test_student_claims_never_appear_as_facts(case, mislabel):
    from lecture_recorder.pipeline.gpu import gpu_job

    cfg = load_config()
    llm = LocalLLM.from_config(cfg.llm)
    try:
        with gpu_job(cfg.home):
            r = summarize_doc(doc(case, mislabel), llm, title=CASES[case]["title"])
            text = r["glance"] + "\n" + "\n".join(p["notes"] for p in r["parts"])
            stated = {k: judge(llm, text, c) for k, c in CASES[case]["claims"].items()}
            llm.unload()
    finally:
        llm.close()
    assert not any(stated.values()), stated
    for fact in CASES[case]["kept"]:  # the lecturer's corrected facts survive
        assert re.search(fact, text, re.I), fact
    assert len(r["parts"]) == 1 and r["reduce_levels"] == 0  # same requests as before: 1 map + 1 glance
