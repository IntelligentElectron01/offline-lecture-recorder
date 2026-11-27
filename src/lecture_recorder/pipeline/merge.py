"""Merge transcript words with diarization turns by time overlap, and name the speakers.

Speaker 1 = the voice with the most diarized speaking time, tagged "Main"; all others "Audience".
The heuristic is provisional until it is checked on real recordings.
"""
MAIN, AUDIENCE = "Main", "Audience"


def rank_speakers(turns: list[dict]) -> list[dict]:
    """[{'id': 'Speaker 1', 'role': 'Main', 'label': 'SPEAKER_03', 'speech_s': 812.4}, ...], most speech first."""
    totals: dict[str, float] = {}
    for t in turns:
        totals[t["label"]] = totals.get(t["label"], 0.0) + max(0.0, t["end"] - t["start"])
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"id": f"Speaker {i}", "role": MAIN if i == 1 else AUDIENCE, "label": label, "speech_s": round(s, 3)}
            for i, (label, s) in enumerate(ranked, start=1)]


def label_for(start: float, end: float, turns: list[dict]) -> str | None:
    """Turn label with the largest overlap with [start, end]; if none overlaps, the nearest turn."""
    if not turns:
        return None
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(end, t["end"]) - max(start, t["start"])
        if overlap > best_overlap:
            best, best_overlap = t["label"], overlap
    if best is not None:
        return best
    mid = (start + end) / 2
    return min(turns, key=lambda t: max(t["start"] - mid, mid - t["end"], 0.0))["label"]


def merge(stt_segments: list[dict], turns: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split each Whisper segment wherever the speaker changes. Returns (segments, speakers)."""
    speakers = rank_speakers(turns)
    by_label = {s["label"]: s for s in speakers}
    default = speakers[0] if speakers else {"id": "Speaker 1", "role": MAIN, "label": None, "speech_s": 0.0}
    if not speakers:
        speakers = [default]

    out: list[dict] = []
    for seg in stt_segments:
        words = seg.get("words") or [{"w": " " + seg["text"], "start": seg["start"], "end": seg["end"], "p": None}]
        current, run = None, []
        for w in words:
            spk = by_label.get(label_for(w["start"], w["end"], turns), default)
            if current is not None and spk["id"] != current["id"]:
                out.append(_segment(run, current))
                run = []
            current = spk
            run.append(w)
        if run:
            out.append(_segment(run, current))
    for i, s in enumerate(out):
        s["id"] = i
    return out, speakers


def _segment(words: list[dict], spk: dict) -> dict:
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "speaker": spk["id"],
        "role": spk["role"],
        "text": "".join(w["w"] for w in words).strip(),
        "words": words,
    }


def role_time(segments: list[dict], start: float, end: float, role: str, step: float = 0.01) -> float:
    """Seconds of [start, end] covered by at least one segment with `role` (overlaps counted once)."""
    n = max(0, int(round((end - start) / step)))
    covered = [False] * n
    for s in segments:
        if s["role"] != role or s["end"] <= start or s["start"] >= end:
            continue
        for i in range(max(0, int((s["start"] - start) / step)), min(n, int(round((s["end"] - start) / step)))):
            covered[i] = True
    return sum(covered) * step


def role_accuracy(segments: list[dict], truth_turns: list[dict]) -> float:
    """Share of true speech time whose transcript label has the right role (main vs audience).

    Time where no transcript segment covers the truth turn counts as wrong; time covered by
    several segments counts once.
    """
    total = correct = 0.0
    for t in truth_turns:
        total += t["end"] - t["start"]
        correct += role_time(segments, t["start"], t["end"], MAIN if t["role"] == "main" else AUDIENCE)
    return correct / total if total else 0.0
