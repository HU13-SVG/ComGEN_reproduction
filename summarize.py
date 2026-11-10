#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(_file_).resolve().parent
CATEGORIES = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
PAPER = [87.2, 99.1, 92.9, 79.8, 95.2]

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

def audit(folder, rows):
    protocol = json.loads((folder / 'protocol.json').read_text())
    split = json.loads((folder / 'split.json').read_text())
    pair_file = Path(protocol['args']['masks']) / folder.name / 'pairs_1000.json'
    if not pair_file.exists():
        pair_file = ROOT / 'results' / 'stage2_pairs' / folder.name / 'pairs_1000.json'
    pairs = json.loads(pair_file.read_text())
    (train, val) = (set(split['train_indices']), set(split['validation_indices']))
    ns = len(pairs)
    groups = lambda indices: {pairs[i]['closest_normal'] for i in indices if i < ns}
    test_paths = {str(Path(r['path']).resolve()) for r in rows}
    used_paths = {str(Path(p).resolve()) for p in split['paths']}
    paths_for = lambda indices: {str(Path(split['paths'][i]).resolve()) for i in indices}
    test_root = Path(protocol['args']['data_root']) / folder.name / 'test'
    expected_test_paths = {str(p.resolve()) for p in test_root.glob('*/*.png')}
    own_refs = [i for i in range(len(protocol['train_normals'])) if split['references'][ns + i] == i]
    return dict(split_index_overlap=sorted(train & val), synthetic_source_group_overlap=sorted(groups(train) & groups(val)), train_validation_test_path_overlap=sorted(used_paths & test_paths), training_validation_path_overlap=sorted(paths_for(train) & paths_for(val)), missing_test_paths=sorted(expected_test_paths - test_paths), extra_test_paths=sorted(test_paths - expected_test_paths), normal_self_reference_indices=own_refs, train_count=len(train), validation_count=len(val), synthetic_train_count=sum((i < ns for i in train)), synthetic_validation_count=sum((i < ns for i in val)), distinct_synthetic_image_paths=len({r['image'] for r in pairs}), pair_manifest_hash_matches=hashlib.sha256(pair_file.read_bytes()).hexdigest() == protocol['pair_manifest_sha256'], test_used_for_selection=protocol['test_used_for_selection'])

def main():
    parser = argparse.ArgumentParser(description=_doc_)
    parser.add_argument('--root', type=Path, default=ROOT / 'results' / 'stage5_full')
    args = parser.parse_args()
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), root=str(args.root.resolve()), method='independent Mann-Whitney average-rank AUROC; equal-weighted category macro average', categories=[], missing_categories=[])
    for (category, paper) in zip(CATEGORIES, PAPER):
        folder = args.root / category
        if not (folder / 'result.json').exists() or not (folder / 'test_scores.csv').exists():
            report['missing_categories'].append(category)
            continue
        result = json.loads((folder / 'result.json').read_text())
        with (folder / 'test_scores.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len({r['path'] for r in rows}) != len(rows):
            raise ValueError(f'{category}: duplicated test paths')
        if any((int(r['label']) != int(r['type'] != 'good') for r in rows)):
            raise ValueError(f'{category}: label/type mismatch')
        score = auc(rows)
        if abs(score - result['image_auroc']) > 1e-08 or len(rows) != result['test_count']:
            raise ValueError(f'{category}: stored metric/count mismatch')
        row = dict(category=category, image_auroc=score, paper_image_auroc=paper, gap_percentage_points=score - paper, test_count=len(rows), best_epoch=result['best_epoch'])
        for kind in ('logical_anomalies', 'structural_anomalies'):
            row[kind + '_auroc'] = auc([r for r in rows if r['type'] in ('good', kind)])
            if abs(row[kind + '_auroc'] - result[kind + '_auroc']) > 1e-08:
                raise ValueError(f'{category}: stored subtype metric mismatch')
        row['audit'] = audit(folder, rows)
        report['categories'].append(row)
    report['complete'] = not report['missing_categories']
    report['five_category_mean_image_auroc'] = sum((r['image_auroc'] for r in report['categories'])) / 5 if report['complete'] else None
    report['paper_target_mean'] = 91.2
    report['paper_reported_mean_arithmetic_check'] = dict(rounded_category_values=PAPER, arithmetic_mean=sum(PAPER) / len(PAPER), paper_table_reported_mean=91.2, note='The rounded per-category values average to 90.84, not 91.2; this discrepancy is reported rather than adjusted.')
    output = args.root / ('audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    fields = ['category', 'image_auroc', 'paper_image_auroc', 'gap_percentage_points', 'test_count', 'best_epoch', 'logical_anomalies_auroc', 'structural_anomalies_auroc']
    with (output / 'summary.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['categories'])
    print(json.dumps(dict(output=str(output), complete=report['complete'], missing=report['missing_categories'], five_category_mean_image_auroc=report['five_category_mean_image_auroc']), indent=2))
if _name_ == '_main_':
    main()
