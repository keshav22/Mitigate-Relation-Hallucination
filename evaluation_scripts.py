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

RELATION_TYPE_LOOKUP = {}

def create_relation_type_lookup(questions_path: dict):
    """
    Create a lookup dict mapping question text to relation type from a JSONL file.
    Each line in the file should be a JSON object with 'question' and 'relation_type' fields.
    """
    lookup = {}
    for key, path in questions_path.items():
        assert key in ["mcq", "yesno", "vqa"], f"Invalid key in questions_path: {key}"
        path = Path(path)
        if not path.exists():
            print(f"Questions file not found: {questions_path}")
            continue
        lookup[key] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    question = obj.get("query_prompt")
                    relation_type = obj.get("relation_type")
                    if question and relation_type:
                        assert relation_type in ["cognitive", "perception"], f"Invalid relation_type: {relation_type}"
                        lookup[key][question] = relation_type
                except Exception as e:
                    print(f"Error parsing line in questions file: {e}")
                    continue
    return lookup

def validate_mcq_choice(response, label):
    """
    Validate that both response and label are one of A/B/C/D and that the response
    selects the same option letter as the label. Returns True only when both are valid
    and match; otherwise False.
    """

    def extract_choice(s):
        if not isinstance(s, str):
            return None
        s = s.strip()
        # match a standalone letter A-D (handles "A", "A.", "A) ", "A. holding", etc.)
        m = re.search(r"\b([A-Da-d])\b", s)
        return m.group(1).lower() if m else None

    resp_choice = extract_choice(response)
    label_choice = extract_choice(label)

    return resp_choice, label_choice
    

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


def evaluate_mcq_choice(path: Path, detailed_metrics: bool = False):
    if not path.exists():
        return {"file": str(path), "error": f"File not found: {path}"}

    counts = Counter()
    total_lines = 0
    ambiguous_responses = set()

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

            # validate MCQ choices: response first, label second
            resp_choice, label_choice = validate_mcq_choice(response_raw, label_raw)

            # If either side looks like an MCQ choice, prefer MCQ handling
            if resp_choice is None or label_choice is None:

                if label_choice is None:
                    counts["mcq_parse_error_label"] += 1
                    
                if resp_choice is None:
                    ambiguous_responses.add("" if response_raw is None else str(response_raw))
                    counts["mcq_parse_error_response"] += 1
            else:
                counts["evaluated_mcq"] += 1

                # both parsed as MCQ choices; count per-class TP/FP
                # true positive when predicted == label
                if resp_choice == label_choice:
                    counts[f"mcq_TP_{resp_choice}"] += 1
                else:
                    counts[f"mcq_FP_{resp_choice}"] += 1
                    counts[f"mcq_FN_{label_choice}"] += 1


    # per-class MCQ precision (for choices a,b,c,d)
    per_class_mcq = {}
    prec_vals = []
    rec_vals = []
    f1_vals = []
    for ch in ["a", "b", "c", "d"]:
        tp_ch = counts.get(f"mcq_TP_{ch}", 0)
        fp_ch = counts.get(f"mcq_FP_{ch}", 0)
        fn_ch = counts.get(f"mcq_FN_{ch}", 0)
        
        precision = tp_ch / (tp_ch + fp_ch) if (tp_ch + fp_ch) > 0 else None
        recall = tp_ch / (tp_ch + fn_ch) if (tp_ch + fn_ch) > 0 else None
        f1_ch = (2 * precision * recall) / (precision + recall) if (precision is not None and recall is not None and (precision + recall) > 0) else None

        if precision is not None:
            prec_vals.append(precision)
        if recall is not None:
            rec_vals.append(recall)
        if f1_ch is not None:
            f1_vals.append(f1_ch)

        per_class_mcq[ch] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_ch
        }
    
    macro_precision = sum(prec_vals) / len(prec_vals) if prec_vals else None
    macro_recall = sum(rec_vals) / len(rec_vals) if rec_vals else None
    macro_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else None

    accuracy_total = sum(counts[f"mcq_TP_{ch}"] for ch in ["a", "b", "c", "d"]) / total_lines if total_lines > 0 else None
    accuracy_evaluated = sum(counts[f"mcq_TP_{ch}"] for ch in ["a", "b", "c", "d"]) / counts.get("evaluated_mcq", 0) if counts.get("evaluated_mcq", 0) > 0 else None
    hallucination_rate_total = 1 - accuracy_total if accuracy_total is not None else None
    hallucination_rate_evaluated = 1 - accuracy_evaluated if accuracy_evaluated is not None else None
    summary = {
        "total_lines": total_lines,
        "evaluated_lines": counts.get("evaluated_mcq", 0),
        "accuracy_over_all_lines": accuracy_total,
        "accuracy_over_evaluated_lines": accuracy_evaluated,
        "hallucination_rate_over_all_lines": hallucination_rate_total,
        "hallucination_rate_over_evaluated_lines": hallucination_rate_evaluated,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_mcq": per_class_mcq,        
        "parse_errors_label": counts.get("mcq_parse_error_label", 0),
        "parse_errors_response": counts.get("mcq_parse_error_response", 0),
        "ambiguous_responses": sorted(ambiguous_responses),
    } if detailed_metrics else {
        "total_lines": total_lines,
        "evaluated_lines": counts.get("evaluated_mcq", 0),
        "accuracy_over_all_lines": accuracy_total,
        "accuracy_over_evaluated_lines": accuracy_evaluated,
        "hallucination_rate_over_all_lines": hallucination_rate_total,
        "hallucination_rate_over_evaluated_lines": hallucination_rate_evaluated,
        "ambiguous_responses": sorted(ambiguous_responses)
    }

    out = {
        "file": str(path),
        "title": title,
        "summary": summary
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out




def evaluate_yesno(path: Path, detailed_metrics: bool = False):
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
            question = obj.get("query_prompt")

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
                counts[f"evaluated_{RELATION_TYPE_LOOKUP['yesno'][question]}"] += 1
                if pred == "yes" and gold == "yes":
                    counts["TP"] += 1
                    counts["TP_" + RELATION_TYPE_LOOKUP['yesno'][question]] += 1
                elif pred == "no" and gold == "no":
                    counts["TN"] += 1
                    counts["TN_" + RELATION_TYPE_LOOKUP['yesno'][question]] += 1
                elif pred == "no" and gold == "yes":
                    counts["FN"] += 1
                    counts["FN_" + RELATION_TYPE_LOOKUP['yesno'][question]] += 1
                elif pred == "yes" and gold == "no":
                    counts["FP"] += 1
                    counts["FP_" + RELATION_TYPE_LOOKUP['yesno'][question]] += 1


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

    accuracy_by_type = {}
    hallucination_rate_by_type = {}
    for relation_type in ["cognitive", "perception"]:
        accuracy_by_type[relation_type] = (counts.get(f"TP_{relation_type}", 0) + counts.get(f"TN_{relation_type}", 0)) / counts.get(f"evaluated_{relation_type}", 0) if counts.get(f"evaluated_{relation_type}", 0) > 0 else None
        hallucination_rate_by_type[relation_type] = 1 - accuracy_by_type[relation_type] if accuracy_by_type[relation_type] is not None else None
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
        "accuracy_by_type": accuracy_by_type,
        "hallucination_rate_by_type": hallucination_rate_by_type,
        "hallucination_rate_over_evaluated": hallucination_rate_evaluated,
        "hallucination_rate_over_all_lines": hallucination_rate_all,
        "ambiguous_gold_examples": sorted(ambiguous_gold_values),
        "ambiguous_pred_examples": sorted(ambiguous_pred_values),
    } if detailed_metrics else {
        "total_lines": total_lines,
        "evaluated_pairs": evaluated,
        "accuracy_over_evaluated": accuracy_over_evaluated,
        "accuracy_over_all_lines": accuracy_over_all,
        "hallucination_rate_over_evaluated": hallucination_rate_evaluated,
        "hallucination_rate_over_all_lines": hallucination_rate_all,
        "ambiguous_pred_examples": sorted(ambiguous_pred_values),
    }
    # simple console output: print summary as JSON (no fancy formatting)
    out = {
        "file": str(path),
        "title": title,
        "summary": summary,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate Yes/No results JSONL (files or root dirs)")
    p.add_argument("paths", nargs="*", help="path(s) to JSONL results file(s) or root directories")
    p.add_argument("--out", "-o", default=str(OUT_REPORT), help="output report JSON path")
    p.add_argument("--detailed_metrics", action="store_true", help="include detailed metrics in the output report")
    args = p.parse_args()

    paths = []

    questions_path = {
        "mcq": "Reefknot/Dataset/Multichoice.jsonl",
        "yesno": "Reefknot/Dataset/YESNO.jsonl",
        "vqa": "Reefknot/Dataset/VQA.jsonl"
    }

    global RELATION_TYPE_LOOKUP
    RELATION_TYPE_LOOKUP = create_relation_type_lookup(questions_path)

    for rp in args.paths:
        pth = Path(rp)
        if pth.is_dir():
            # collect all .jsonl files under the directory (recursive)
            paths.extend(sorted(pth.rglob("*.jsonl")))
        else:
            paths.append(pth)

    # deduplicate 
    seen = set()
    deduped = []
    for p in paths:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            deduped.append(p)
    paths = deduped

    per_file_reports = []

    for path in paths:
        if "Multichoice" in path.stem:
            #ToDo : handle MCQ lines where response is not a letter A-D and instead whole words.
            result = evaluate_mcq_choice(path, detailed_metrics=args.detailed_metrics)
        elif "YesNo" in path.stem:
            result = evaluate_yesno(path, detailed_metrics=args.detailed_metrics)
        else:
            #ToDo: VQA files?
            print(f"Skipping unrecognized file (not YesNo or Multichoice): {path}")
            continue
        per_file_reports.append(result)
        
       

    combined_report = {
        "files": per_file_reports
    }

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as out:
        json.dump(combined_report, out, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
