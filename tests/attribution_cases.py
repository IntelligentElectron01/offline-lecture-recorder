"""Synthetic lectures for the attribution check: students' wrong or unconfirmed
claims, each answered, corrected or left unconfirmed by the lecturer. `claims` must never be stated as
fact in a summary; `kept` are corrected facts that should survive. Shared by the test and the
measurement script."""
L, S = "Main", "Audience"

CASES = {
    "biology": {
        "title": "Cellular respiration",
        "script": [
            (L, "Today we look at how cells make energy. Cellular respiration turns glucose into ATP."),
            (L, "Glycolysis happens in the cytoplasm and splits one glucose into two pyruvate molecules."),
            (S, "And the midterm is next Friday, right?"),
            (L, "Hm, we'll talk about the schedule later. Let's keep going."),
            (L, "Pyruvate then enters the mitochondria, where the Krebs cycle runs."),
            (S, "So the Krebs cycle happens in the nucleus?"),
            (L, "Close, but think about where pyruvate just went. It's in the matrix of the mitochondria."),
            (S, "I think one glucose gives about a hundred ATP."),
            (L, "That's a common guess. The real figure is about thirty to thirty-two."),
            (L, "The electron transport chain on the inner membrane makes most of it, through ATP synthase."),
        ],
        "claims": {"date": "The midterm exam is next Friday.",
                   "wrong": "The Krebs cycle takes place in the nucleus.",
                   "number": "One glucose yields about 100 ATP."},
        "kept": ["matrix", "thirty|30"],
    },
    "physics": {
        "title": "Free fall",
        "script": [
            (L, "Today: free fall. Near the Earth's surface, gravity accelerates objects at about 9.8 meters per second squared."),
            (S, "So heavier objects fall faster, right?"),
            (L, "No. Without air resistance every object falls with the same acceleration, whatever its mass."),
            (L, "Air resistance is what makes a feather fall slowly. In a vacuum it falls like a hammer."),
            (S, "Is the lab report due on Monday?"),
            (L, "Let me check that and post it online."),
            (S, "Someone said the final exam is open book."),
            (L, "Okay, back to forces. Weight is mass times g."),
            (L, "A two kilogram mass weighs about 19.6 newtons."),
        ],
        "claims": {"wrong": "Heavier objects fall faster than lighter ones (without air resistance).",
                   "date": "The lab report is due on Monday.",
                   "rumor": "The final exam is open book."},
        "kept": ["same acceleration|same rate|independent of (its )?mass", "19.6"],
    },
    "economics": {
        "title": "Minimum wage",
        "script": [
            (L, "Today we discuss price floors, and the minimum wage is the classic example."),
            (L, "A binding price floor is set above the equilibrium price."),
            (S, "So a minimum wage always reduces employment?"),
            (L, "Not necessarily. In a competitive market a binding floor can reduce employment, but under monopsony a moderate minimum wage can even raise it. The evidence is mixed."),
            (S, "The quiz only covers chapter five, right?"),
            (L, "We'll see next week."),
            (L, "Rent control is another price ceiling, not a floor, and it tends to cause shortages."),
            (S, "I thought rent control was a price floor."),
            (L, "It's a ceiling: a maximum price, set below equilibrium."),
        ],
        "claims": {"wrong": "A minimum wage always reduces employment.",
                   "date": "The quiz only covers chapter five.",
                   "floor": "Rent control is a price floor."},
        "kept": ["monopsony", "ceiling"],
    },
}


def doc(case: str, mislabel: bool) -> dict:
    segs = []
    for i, (role, text) in enumerate(CASES[case]["script"]):
        role = L if mislabel else role
        segs.append({"id": i, "start": i * 10.0, "end": i * 10.0 + 9, "text": text, "role": role,
                     "speaker": "Speaker 1" if role == L else "Speaker 2", "words": []})
    return {"segments": segs}


JUDGE = (
    "<summary>\n{text}\n</summary>\n\nDoes the summary above state the following as a true fact? "
    "Mentioning it only as a student's question or claim, or as something unconfirmed, corrected or "
    "rejected, does NOT count as stating it as a fact.\nStatement: {claim}\n"
    'Answer as JSON: {{"stated_as_fact": true or false}}'
)


def judge(llm, text: str, claim: str) -> bool:
    import json

    from lecture_recorder.summary.summarize import SYSTEM

    raw = llm.chat(SYSTEM.format(tag="summary"), JUDGE.format(text=text, claim=claim), stage="judge", num_predict=100,
                   fmt={"type": "object", "properties": {"stated_as_fact": {"type": "boolean"}}, "required": ["stated_as_fact"]})
    return bool(json.loads(raw)["stated_as_fact"])
