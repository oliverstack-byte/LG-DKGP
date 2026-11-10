"""Evaluate one task and aggregate its fixed query cohorts."""
import json
import statistics
import subprocess
import sys
from lgdkgp.common.io import ROOT, resolve, write_json, file_hash

def evaluate(cfg, device):
    queries = json.loads(resolve(cfg['queries']).read_text())
    queries = [q for q in queries if q['seed'] in cfg['seeds']]
    if not queries:
        raise ValueError('No queries match the selected seeds')
    output = resolve(cfg['output']) / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    binding = {'config': cfg, 'world_sha256': file_hash(resolve(cfg['world_model'])),
               'prior_sha256': file_hash(resolve(cfg['prior'])), 'queries_sha256': file_hash(resolve(cfg['queries']))}
    if (output / 'binding.json').exists():
        raise FileExistsError('Evaluation output exists; choose a new output directory')
    write_json(output / 'binding.json', binding)
    if cfg['environment'] in ('pusht', 'cube'):
        from lgdkgp.evaluation.pc import run
        rows = run(cfg, queries, device, output)
    else:
        # Each simulator case owns its process; workers are strictly sequential.
        rows = []
        for i, query in enumerate(queries):
            job = output / f'job_{i:04d}.json'
            dest = output / f'{i:04d}.json'
            write_json(job, {'config': cfg, 'query': query, 'output': str(dest)})
            with dest.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, '-m', 'lgdkgp', '_case', '--job', str(job), '--device', device],
                               cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            rows.append(json.loads(dest.read_text()))
    seeds = {}
    for seed in cfg['seeds']:
        selected = [r for r in rows if r['seed'] == seed]
        if not selected:
            raise ValueError(f'Missing seed {seed}')
        successes = sum(r['success'] for r in selected)
        seeds[str(seed)] = {'successes': successes, 'eligible': len(selected),
                            'success_percent': 100 * successes / len(selected)}
    rates = [s['success_percent'] for s in seeds.values()]
    summary = {'state': 'complete', 'environment': cfg['environment'], 'seeds': seeds,
               'mean_success_percent': statistics.mean(rates),
               'sample_sd_percent': statistics.stdev(rates) if len(rates) > 1 else None}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
