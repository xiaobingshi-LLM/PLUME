from __future__ import annotations

import re
import string
from collections import Counter


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = normalize(prediction).split(), normalize(reference).split()
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not predicted or not expected:
        return float(predicted == expected)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-L precision/recall/F1 using the paper's stemmed implementation."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as error:
        raise RuntimeError("ROUGE-L evaluation requires the rouge-score package") from error
    score = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(
        reference, prediction)["rougeL"]
    return {"rouge_l_precision": score.precision, "rouge_l_recall": score.recall,
            "rouge_l_f1": score.fmeasure}
