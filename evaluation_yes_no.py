#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import Counter
import argparse

"""
Modified evaluation_yes_no.py

- Produces only summary information in the output report (no line-by-line items).
- Accepts one or more JSONL result files (positional args).
- If no files provided, uses DEFAULT_PATH.
"""

# configurable defaults
DEFAULT_PATH = Path("Experiment_Results/DTC_Quantized_7B/YesNo_results.jsonl")
OUT_REPORT = Path("yesno_evaluation_report.json")

YES_TERMS = {"yes", "y"}
NO_TERMS = {"no", "n"}

WORD_YES_RE = re.compile(r"\byes\b", re.I)
WORD_NO_RE = re.compile(r"\bno\b", re.I)


def normalize_to_yesno(raw):
    """
    Convert various label/response strings to 'yes' / 'no' or None if ambiguous.
    """
    if raw is None or not isinstance(raw, str):
        return None
    s = str(raw).strip()
    if not s:
        return None
    low = s.lower().strip().strip(".,:;\"'()[]{}")
    if low in YES_TERMS:
        return "yes"
    if low in NO_TERMS:
        return "no"
    if WORD_YES_RE.search(s):
        if WORD_NO_RE.search(s):
            return None #ambiguous if both present
        return "yes"
    if WORD_NO_RE.search(s):
        return "no"
    return None


def evaluate_file(path: Path):
    """
    Evaluate a single JSONL file.
    Returns a dict: {
      "file": str,
      "summary": {...},
      "titles": { title: {counts...} },
      "ambiguous_gold_values": [...],
      "ambiguous_pred_values": [...],
    }

    Note: does NOT collect or return line-by-line items.
    """
    if not path.exists():
        return {"file": str(path), "error": f"File not found: {path}"}

    counts = Counter()
    total_lines = 0

    # collect raw ambiguous values
    ambiguous_gold_values = set()
    ambiguous_pred_values = set()

    # candidate keys to look for gold/label and response/prediction
   

    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                counts["parse_error"] += 1
                continue

            title = path.stem
            label_raw = obj.get("label")
            response_raw = obj.get("response")

            gold = normalize_to_yesno(label_raw)
            pred = normalize_to_yesno(response_raw)

            # collect ambiguous raw values (as strings) for reporting
            if gold is None:
                ambiguous_gold_values.add("" if label_raw is None else str(label_raw))
                counts["gold_missing_or_ambiguous"] += 1
            if pred is None:
                ambiguous_pred_values.add("" if response_raw is None else str(response_raw))
                # Only increment pred_missing_or_ambiguous_total once per row; the flow below also increments it.
                counts["pred_missing_or_ambiguous_total"] += 1
                if gold == "yes":
                    counts["pred_missing_or_ambiguous_label_yes"] += 1
                if gold == "no":
                    counts["pred_missing_or_ambiguous_label_no"] += 1

            # determine correctness only if both present
            if gold is not None and pred is not None:
                if pred == "yes" and gold == "yes":
                    counts["TP"] += 1
                elif pred == "no" and gold == "no":
                    counts["TN"] += 1
                elif pred == "no" and gold == "yes":
                    counts["FN"] += 1
                elif pred == "yes" and gold == "no":
                    counts["FP"] += 1


    precision = counts.get("TP", 0) / (counts.get("TP", 0) + counts.get("FP", 0)) if (counts.get("TP", 0) + counts.get("FP", 0)) > 0 else None
    recall = counts.get("TP", 0) / (counts.get("TP", 0) + counts.get("FN", 0)) if (counts.get("TP", 0) + counts.get("FN", 0)) > 0 else None
    f1 = (2 * precision * recall) / (precision + recall) if precision is not None and recall is not None and (precision + recall) > 0 else None 
    correct = counts.get("TP", 0) + counts.get("TN", 0)
    incorrect = counts.get("FP", 0) + counts.get("FN", 0)
    pred_missing = counts.get("pred_missing_or_ambiguous_total", 0)
    pred_missing_yes = counts.get("pred_missing_or_ambiguous_label_yes", 0)
    pred_missing_no = counts.get("pred_missing_or_ambiguous_label_no", 0)
    gold_missing = counts.get("gold_missing_or_ambiguous", 0)
    parse_error = counts.get("parse_error", 0)
    evaluated = correct + incorrect

    accuracy_over_evaluated = (correct / evaluated) if evaluated > 0 else None
    hallucination_rate_evaluated = 1 - accuracy_over_evaluated if accuracy_over_evaluated is not None else None
    accuracy_over_all = (correct / total_lines) if total_lines > 0 else None
    hallucination_rate_all = 1 - accuracy_over_all if accuracy_over_all is not None else None

    summary = {
        "total_lines": total_lines,
        "evaluated_pairs": evaluated,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "incorrect": incorrect,
        "pred_missing_or_ambiguous_total": pred_missing,
        "pred_missing_or_ambiguous_label_yes": pred_missing_yes,
        "pred_missing_or_ambiguous_label_no": pred_missing_no,
        "gold_missing_or_ambiguous": gold_missing,
        "parse_error": parse_error,
        "accuracy_over_evaluated": accuracy_over_evaluated,
        "accuracy_over_all_lines": accuracy_over_all,
        "hallucination_rate_over_evaluated": hallucination_rate_evaluated,
        "hallucination_rate_over_all_lines": hallucination_rate_all,
    }
    # simple console output: print summary as JSON (no fancy formatting)
    out = {
        "file": str(path),
        "title": path.stem,
        "summary": summary,
        "ambiguous_gold_examples": sorted(ambiguous_gold_values)[:10],
        "ambiguous_pred_examples": sorted(ambiguous_pred_values)[:10],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    
    return {
        "file": str(path),
        "summary": summary,
        "title": path.stem,
        "ambiguous_gold_values": sorted(ambiguous_gold_values),
        "ambiguous_pred_values": sorted(ambiguous_pred_values),
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Yes/No results JSONL (one or more files)")
    p.add_argument("paths", nargs="*", help="path(s) to JSONL results file(s)")
    p.add_argument("--out", "-o", default=str(OUT_REPORT), help="output report JSON path")
    args = p.parse_args()

    paths = args.paths or [str(DEFAULT_PATH)]
    paths = [Path(p) for p in paths]

    per_file_reports = []

    for path in paths:
        result = evaluate_file(path)
        per_file_reports.append(result)
        
       

    combined_report = {
        "files": per_file_reports
    }

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as out:
        json.dump(combined_report, out, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
