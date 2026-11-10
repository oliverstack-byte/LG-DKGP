"""Aggregate completed full-method seed summaries into the main result row."""
import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVS = ['pusht', 'cube', 'libero_caddy', 'libero_scene2', 'robotwin_a2b']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outputs', type=Path, default=ROOT / 'outputs')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/main_summary.json')
    a = p.parse_args()
    rows = []
    for env in ENVS:
        path = a.outputs / env / 'evaluation/summary.json'
        report = json.loads(path.read_text())
        if report['state'] != 'complete' or len(report['seeds']) != 5:
            raise ValueError('Complete five-seed result required: ' + env)
        rows.append({'environment': env, 'mean_success_percent': report['mean_success_percent'],
                     'sample_sd_percent': report['sample_sd_percent'],
                     'eligible': sum(s['eligible'] for s in report['seeds'].values())})
    result = {'method': 'LG-DKGP', 'tasks': rows,
              'macro_success_percent': statistics.mean(r['mean_success_percent'] for r in rows)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    with a.output.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
