#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
ROOT = Path(_file_).resolve().parents[1]
CATEGORIES = ('breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors')
EXPECTED = {'breakfast_box': 76.9976, 'juice_bottle': 87.8787, 'pushpins': 57.7646, 'screw_bag': 65.2481, 'splicing_connectors': 59.1109}
EXPECTED_MACRO = 69.4

def auc(rows):
    ordered = sorted(((float(r['score']), int(r['label'])) for r in rows))
    if any((not math.isfinite(s) or y not in (0, 1) for (s, y) in ordered)):
        raise ValueError('Non-finite score or non-binary label')
    positives = sum((y for (_, y) in ordered))
    negatives = len(ordered) - positives
    if not positives or not negatives:
        raise ValueError('AUROC requires both classes')
    (rank_sum, start) = (0.0, 0)
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank_sum += (start + 1 + end) / 2 * sum((y for (_, y) in ordered[start:end]))
        start = end
    return 100 * (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

def read_rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    print(f"{'category':22s} {'n':>5s} {'stored':>8s} {'recomputed':>11s} {'expected':>9s}")
    ok = True
    values = []
    for cat in CATEGORIES:
        folder = ROOT / 'results' / 'stage5_full' / cat
        result = json.loads((folder / 'result.json').read_text())
        rows = read_rows(folder / 'test_scores.csv')
        score = auc(rows)
        stored = result['image_auroc']
        if abs(score - stored) > 1e-08:
            ok = False
        same = round(score, 4) == EXPECTED[cat]
        ok = ok and same
        values.append(score)
        print(f"{cat:22s} {len(rows):5d} {stored:8.4f} {score:11.4f} {EXPECTED[cat]:9.4f} {('OK' if same else 'MISMATCH')}")
        for kind in ('logical_anomalies', 'structural_anomalies'):
            sub = [r for r in rows if r['type'] in ('good', kind)]
            sub_score = auc(sub)
            stored_sub = result[kind + '_auroc']
            if abs(sub_score - stored_sub) > 1e-08:
                ok = False
    total = sum((len(read_rows(ROOT / 'results' / 'stage5_full' / cat / 'test_scores.csv')) for cat in CATEGORIES))
    macro = sum(values) / len(values)
    same = round(macro, 4) == EXPECTED_MACRO
    ok = ok and same
    print(f"{'macro (mean of 5)':22s} {total:5d} {'':8s} {macro:11.4f} {EXPECTED_MACRO:9.4f} {('OK' if same else 'MISMATCH')}")
    print()
    print('Result:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1
if _name_ == '_main_':
    raise SystemExit(main())
