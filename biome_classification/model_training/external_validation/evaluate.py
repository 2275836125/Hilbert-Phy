"""External inference with existing formal checkpoints; no fitting or new splits."""
import argparse
from collections import Counter
import csv
import fcntl
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy import sparse
import torch

from foundation import (DATA, HERE, ROOT, TRAIN, ExternalData, classification_metrics,
                        read_json, require, sha, write_json)
from prepare import inventory

MODELS = [f'{encoding}_resnet{depth}' for encoding in ('hilbert', 'rowwise') for depth in (18, 34, 50)]
MODELS += ['deepphylo_taxonomy', 'popphy_cnn']
MODELS += [f'{model}_{order}' for model in ('random_forest', 'logistic_regression') for order in ('rowwise', 'random')]
CHOICES = MODELS + ['1d_resnet18']


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_predictor(record, classes, data, device, threads):
    """Return the exact trained architecture and its batch probability function."""
    name = record['model']
    run = ROOT / record['run']
    cfg = read_json(run / 'config.json')
    checkpoint = ROOT / record['checkpoint']
    incidence = order = None
    if name.startswith(('random_forest_', 'logistic_regression_')):
        import joblib
        request = cfg['request']
        require(request['model'] + '_' + request['order'] == name, 'Classical model/order mismatch')
        import sklearn
        require(cfg['request']['versions']['sklearn'] == sklearn.__version__, 'sklearn version differs from saved estimator')
        model = joblib.load(checkpoint)
        require(np.array_equal(model.classes_, np.arange(len(classes))), 'Estimator class order mismatch')
        require(model.n_features_in_ == len(data.tax_ids), 'Estimator feature dimension mismatch')
        if hasattr(model, 'n_jobs'):
            model.n_jobs = threads
        if name.endswith('_random'):
            order = np.load(TRAIN / 'classical_order_control/permutation_seed0.npy')
            require(np.array_equal(order, np.random.default_rng(0).permutation(len(data.tax_ids))), 'Random permutation mismatch')
        return model.predict_proba, incidence, order
    if 'resnet' in name:
        encoding, depth = name.split('_resnet')
        require(cfg['encoding'] == encoding and cfg['depth'] == int(depth), 'ResNet architecture mismatch')
        definitions = module_from(TRAIN / 'models.py', 'external_resnet_definitions')
        model = definitions.build_model(encoding, int(depth), len(classes))
        h = torch.as_tensor(np.load(ROOT / 'data_processing/hilbert_encoding/hilbert.npy'), device=device)
        require(np.array_equal(np.sort(h.cpu().numpy()), np.arange(len(data.tax_ids))), 'Invalid Hilbert permutation')
        def forward(x, present):
            return model(definitions.encode(x, encoding, h))
    else:
        require(cfg['method_key'] == name, 'Phylogeny architecture mismatch')
        base = TRAIN / f'phylogeny_baselines/{name}'
        cache = base / 'cache'
        manifest = read_json(cache / 'manifest.json')
        require(cfg['signature'] == manifest['signature'] and cfg['tree_signature'] == manifest['tree_signature'], 'Tree signature mismatch')
        require(np.array_equal(np.load(cache / 'tax_ids.npy'), data.tax_ids), 'Tree taxa mismatch')
        definitions = module_from(TRAIN / 'phylogeny_baselines/networks.py', 'external_' + name)
        if name == 'deepphylo_taxonomy':
            embeddings = np.load(cache / 'embeddings.npy')
            model = definitions.DeepPhylo(embeddings, len(classes), hidden=cfg['architecture']['hidden'], kernel=cfg['architecture']['kernel'])
            def forward(x, present):
                return model(x, present)
        else:
            incidence = sparse.load_npz(cache / 'incidence.npz')
            shape = read_json(cache / 'tree_audit.json')['matrix_shape']
            require(shape == cfg['architecture']['matrix_shape'], 'Tree matrix shape mismatch')
            positions = torch.as_tensor(np.load(cache / 'positions.npy'), device=device)
            require(positions.shape == (incidence.shape[1], 2), 'Tree position dimensions mismatch')
            require(len(torch.unique(positions, dim=0)) == len(positions), 'Duplicate tree positions')
            require(bool(((positions >= 0) & (positions < torch.tensor(shape, device=device))).all()), 'Invalid tree positions')
            model = definitions.PopPhyCNN(shape, len(classes))
            def forward(x, present):
                matrix = x.new_zeros((len(x), 1, *shape))
                matrix[:, 0, positions[:, 0], positions[:, 1]] = x
                return model(matrix)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if name == 'deepphylo_taxonomy':
        require(torch.equal(state['embeddings'], torch.as_tensor(embeddings)), 'Checkpoint PCA embeddings differ from cache')
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    def predict(x, present):
        with torch.inference_mode():
            logits = forward(torch.from_numpy(x).to(device), torch.from_numpy(present).to(device))
            return logits.softmax(1).cpu().numpy()
    return predict, incidence, order


def identity(record, data, args):
    run = ROOT / record['run']
    paths = [ROOT / record['checkpoint'], run / 'config.json', run / 'result.json',
             ROOT / record['normalization'], ROOT / record['classes'],
             ROOT / record['classes'].replace('classes.json', 'manifest.json'),
             HERE / 'evaluate.py', HERE / 'foundation.py', HERE / 'prepare.py']
    name = record['model']
    if 'resnet' in name:
        paths += [TRAIN / 'models.py', ROOT / 'data_processing/hilbert_encoding/hilbert.npy']
    elif name in ('deepphylo_taxonomy', 'popphy_cnn'):
        base = TRAIN / f'phylogeny_baselines/{name}'
        cache = base / 'cache'
        tree = read_json(cache / 'manifest.json')
        names = ['tax_ids.npy', 'incidence.npz', 'positions.npy', 'tree_audit.json']
        if name == 'deepphylo_taxonomy':
            names.append('embeddings.npy')
        for filename in names:
            require(sha(cache / filename) == tree['artifacts'][filename], f'Corrupt tree artifact: {filename}')
        paths += [TRAIN / 'phylogeny_baselines/networks.py', cache / 'manifest.json'] + [cache / filename for filename in names]
        paths.append((ROOT / record['normalization']).parent / 'manifest.json')
    elif name.endswith('_random'):
        paths.append(TRAIN / 'classical_order_control/permutation_seed0.npy')
    import scipy
    import sklearn
    return dict(model=name, level=record['level'], fold=record['fold'], classes=read_json(ROOT / record['classes']),
                files={str(p.relative_to(ROOT)): sha(p) for p in paths}, data_files=data.manifest['files'],
                limit=args.limit, device=args.device, batch_size=args.batch_size,
                versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                              sklearn=sklearn.__version__, torch=torch.__version__))


def run_one(record, data, args, out):
    request = identity(record, data, args)
    completed = out / 'result.json'
    if completed.exists():
        result = read_json(completed)
        require(result['request'] == request, f'Result provenance changed; use a new --output-dir: {out}')
        for filename, digest in result['artifacts'].items():
            require(sha(out / filename) == digest, f'Damaged output: {out / filename}')
        print(f'SKIP {record["model"]} level={record["level"]} fold={record["fold"]}', flush=True)
        return result
    classes = request['classes']
    y = data.labels_for(record['level'], classes)
    # Always predict unknown/missing labels too; only known labels enter metrics.
    rows = np.arange(min(args.limit, len(y)) if args.limit else len(y))
    y = y[rows]
    predict, incidence, order = load_predictor(record, classes, data, args.device, args.threads)
    p = np.empty((len(rows), len(classes)), dtype=np.float64)
    started = time.perf_counter()
    offset = 0
    for ix, x, present in data.batches(rows, ROOT / record['normalization'], batch_size=args.batch_size,
            method='popphy_cnn' if record['model'] == 'popphy_cnn' else 'genus', incidence=incidence, order=order):
        prob = predict(x) if record['model'].startswith(('random_forest_', 'logistic_regression_')) else predict(x, present)
        require(prob.shape == (len(ix), len(classes)) and np.isfinite(prob).all() and
                np.all((prob >= 0) & (prob <= 1)) and np.allclose(prob.sum(1), 1, atol=1e-6), 'Invalid model probabilities')
        p[offset:offset + len(ix)] = prob
        offset += len(ix)
    require(offset == len(rows), 'Incomplete inference')
    elapsed = time.perf_counter() - started
    known = y >= 0
    metrics = classification_metrics(y[known], p[known], classes) if known.any() else None
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'predictions.csv.tmp'
    with tmp.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'row_index', 'true_label', 'true_index', 'label_status', 'predicted_index', 'predicted_class'] +
                        [f'probability_{i}' for i in range(len(classes))])
        for row, yi, prob in zip(rows, y, p):
            pred = int(prob.argmax())
            writer.writerow([data.ids[row], int(row), data.labels[row][str(record['level'])], int(yi),
                             'known' if yi >= 0 else 'missing' if yi == -1 else 'unseen', pred, classes[str(pred)], *prob.tolist()])
    tmp.replace(out / 'predictions.csv')
    with (out / 'predictions.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, ids=data.ids[rows], row_indices=rows, y=y, probabilities=p,
                            classes=np.array([classes[str(i)] for i in range(len(classes))]))
    (out / 'predictions.npz.tmp').replace(out / 'predictions.npz')
    write_json(out / 'metrics.json', metrics)
    result = dict(status='smoke_complete' if args.limit else 'complete', request=request, inference_seconds=elapsed,
                  predicted_samples=len(rows), known_samples=int(known.sum()), missing_labels=int((y == -1).sum()),
                  unseen_labels=int((y == -2).sum()), metrics=metrics,
                  artifacts={filename: sha(out / filename) for filename in ('predictions.csv', 'predictions.npz', 'metrics.json')})
    write_json(completed, result)
    print(f'DONE {record["model"]} level={record["level"]} fold={record["fold"]} predicted={len(rows)} known={known.sum()}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', nargs='+', choices=CHOICES, default=MODELS)
    parser.add_argument('--level', nargs='+', type=int, choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument('--fold', nargs='+', type=int, choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0, help='Smoke only: predict first N rows; outputs isolated under smoke/')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.limit < 0:
        parser.error('batch-size/threads must be positive; limit must be nonnegative')
    if not args.output_dir.resolve().is_relative_to(HERE):
        parser.error('--output-dir must stay inside external_validation')
    torch.set_num_threads(args.threads)
    torch.device(args.device)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    destination = args.output_dir.resolve() / ('smoke' if args.limit else 'formal')
    destination.mkdir(parents=True, exist_ok=True)
    # Serialize runs writing this result root, including completion markers and summary.
    with (destination / '.evaluate.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = ExternalData(DATA)
        records = [r for r in inventory(data) if r['model'] in args.model and r['level'] in args.level and r['fold'] in args.fold]
        statuses, results = [], []
        for record in records:
            base = dict(model=record['model'], level=record['level'], fold=record['fold'])
            if record['status'] != 'candidate':
                statuses.append(dict(base, status='unavailable', reason=record['reasons']))
                print(f'UNAVAILABLE {base}: {record["reasons"]}', flush=True)
                continue
            out = destination / f'level_{record["level"]}' / record['model'] / f'fold_{record["fold"]}'
            try:
                result = run_one(record, data, args, out)
                results.append(result)
                statuses.append(dict(base, status=result['status'], result=str((out / 'result.json').relative_to(ROOT))))
            except Exception as error:
                statuses.append(dict(base, status='failed', reason=f'{type(error).__name__}: {error}'))
                print(f'FAILED {base}: {error}', file=sys.stderr, flush=True)
        groups = []
        for level in sorted(set(args.level)):
            for model in sorted(set(args.model)):
                items = [r for r in results if r['request']['level'] == level and r['request']['model'] == model and r['metrics'] is not None]
                if not items:
                    continue
                stats = {}
                for key in ('accuracy', 'macro_f1_all_training_classes', 'macro_f1_present_classes', 'macro_auroc_valid_classes'):
                    values = [r['metrics'][key] for r in items if r['metrics'][key] is not None]
                    stats[key] = dict(mean=float(np.mean(values)) if values else None,
                                      std_ddof1=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                      valid_folds=len(values))
                groups.append(dict(level=level, model=model, folds=[r['request']['fold'] for r in items], metrics=stats))
        write_json(destination / 'summary.json', dict(scope='only configurations requested in this invocation',
                   smoke=bool(args.limit), statuses=statuses, counts=dict(Counter(r['status'] for r in statuses)), groups=groups,
                   aggregation='fold metric mean/std on repeated external cohort; not OOF; no ensemble'))
        print(f'Summary: {destination / "summary.json"}', flush=True)
        if any(r['status'] == 'failed' for r in statuses):
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
