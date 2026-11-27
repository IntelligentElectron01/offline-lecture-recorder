"""Text normalization for word error rate: case, punctuation, and simple numbers.

"50%" and "fifty percent" must count as the same words, or WER measures formatting, not hearing.
"""
import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
        "sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_words(rest))
    if n < 1_000_000:
        rest = n % 1000
        return number_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + number_words(rest))
    return str(n)


def normalize(text: str) -> str:
    t = text.lower().replace("%", " percent")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)  # 1,000 -> 1000
    t = re.sub(r"\d+", lambda m: f" {number_words(int(m.group()))} ", t)
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = t.replace("'", "")
    return " ".join(t.split())
