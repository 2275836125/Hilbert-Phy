"""Audit the external package and inventory formal models, without inference."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import numpy as np
from scipy import sparse
from foundation import DATA, HERE, ROOT, TRAIN, ExternalData, read_json, require, sha, write_json


def inventory(data):
    records = []
    for level in range(1, 6):
        snapshot = read_json(DATA / f'training_classes/level_{level}.json')
        for fold in range(1, 6):
            shared = TRAIN / f'shared_cache/level_{level}/fold_{fold}'
            classes = read_json(shared / 'classes.json')
            require(classes == snapshot, f'Class snapshot mismatch: {shared}')
            require(np.array_equal(np.load(shared / 'tax_ids.npy'), data.tax_ids), f'Cache taxa mismatch: {shared}')
            manifest = read_json(shared / 'manifest.json')
            require(manifest['checks']['fit_partition'] == 'train', 'Normalization not train-fitted')
            require(sha_payload(manifest['provenance']) == manifest['signature'], 'Invalid shared cache signature')
            methods = []
            for encoding, depths in [('hilbert', (18, 34, 50)), ('rowwise', (18, 34, 50)), ('1d', (18,))]:
                for depth in depths:
                    model = f'{encoding}_resnet{depth}'
                    methods.append((model, TRAIN / f'results/experiments/level_{level}/{model}/fold_{fold}', shared, 'best.pt', 'resnet'))
            for method in ('deepphylo_taxonomy', 'popphy_cnn'):
                base = TRAIN / f'phylogeny_baselines/{method}'
                methods.append((method, base / f'results/formal/level_{level}/fold_{fold}', base / f'cache/level_{level}/fold_{fold}', 'best.pt', method))
            for method in ('random_forest', 'logistic_regression'):
                for order in ('rowwise', 'random'):
                    out = TRAIN / f'classical_order_control/results/{method}/level_{level}/fold_{fold}/{order}'
                    methods.append((f'{method}_{order}', out, shared, 'model.joblib', 'classical'))
            for model, out, cache, checkpoint, family in methods:
                reasons = []
                cfg = read_json(out / 'config.json') if (out / 'config.json').exists() else None
                result = read_json(out / 'result.json') if (out / 'result.json').exists() else None
                for path in (out / 'config.json', out / 'result.json', out / checkpoint, cache / 'normalization.npz'):
                    if not path.is_file():
                        reasons.append(f'missing:{path.relative_to(ROOT)}')
                if cfg:
                    request = cfg.get('request', cfg)
                    if request.get('smoke') is not False or request.get('level') != level or request.get('fold') != fold:
                        reasons.append('config_task_mismatch')
                    if cfg.get('classes', classes) != classes:
                        reasons.append('config_classes_mismatch')
                    if family == 'resnet' and cfg.get('signature') != manifest['signature']:
                        reasons.append('shared_signature_mismatch')
                    if family == 'classical':
                        identity = request.get('cache', {})
                        if identity.get('signature') != manifest['signature']:
                            reasons.append('shared_signature_mismatch')
                        for name in ('classes.json', 'normalization.npz', 'tax_ids.npy', 'manifest.json'):
                            if identity.get('files', {}).get(name) != sha(shared / name):
                                reasons.append(f'classical_cache_hash_mismatch:{name}')
                        if request.get('order') == 'random':
                            pi = TRAIN / 'classical_order_control/permutation_seed0.npy'
                            if not pi.exists() or sha(pi) != request.get('permutation_sha256'):
                                reasons.append('permutation_hash_mismatch')
                        if result and (result.get('status') != 'complete' or result.get('config_sha256') != sha(out / 'config.json')):
                            reasons.append('classical_completion_mismatch')
                        if (out / checkpoint).exists() and result and result.get('artifacts', {}).get(checkpoint) != sha(out / checkpoint):
                            reasons.append('model_hash_mismatch')
                    if family in ('deepphylo_taxonomy', 'popphy_cnn'):
                        base = TRAIN / f'phylogeny_baselines/{family}/cache'
                        for name in ('manifest.json', 'tree_audit.json', 'tax_ids.npy', 'incidence.npz', 'positions.npy') + (('embeddings.npy',) if family == 'deepphylo_taxonomy' else ()):
                            if not (base / name).exists():
                                reasons.append(f'missing_tree_artifact:{name}')
                        if (base / 'manifest.json').exists():
                            tree = read_json(base / 'manifest.json')
                            if cfg.get('signature') != tree['signature'] or cfg.get('tree_signature') != tree['tree_signature']:
                                reasons.append('tree_signature_mismatch')
                            if tree['provenance']['shared_signature'] != manifest['signature']:
                                reasons.append('tree_shared_signature_mismatch')
                        meta = cache / 'manifest.json'
                        if meta.exists() and (cache / 'normalization.npz').exists():
                            params = read_json(meta)
                            if params.get('signature') != cfg.get('signature') or params.get('normalization_sha256') != sha(cache / 'normalization.npz'):
                                reasons.append('method_normalization_mismatch')
                        else:
                            reasons.append('missing_method_normalization_manifest')
                    if family != 'classical' and result and result.get('signature') != cfg.get('signature'):
                        reasons.append('result_signature_mismatch')
                records.append(dict(level=level, fold=fold, model=model,
                                    status='candidate' if not reasons else 'blocked',
                                    reasons=';'.join(reasons), run=str(out.relative_to(ROOT)),
                                    checkpoint=str((out / checkpoint).relative_to(ROOT)),
                                    normalization=str((cache / 'normalization.npz').relative_to(ROOT)),
                                    classes=str((shared / 'classes.json').relative_to(ROOT))))
    return records


def sha_payload(value):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit_preprocessing(data):
    trees = {}
    for method in ('deepphylo_taxonomy', 'popphy_cnn'):
        base = TRAIN / f'phylogeny_baselines/{method}/cache'
        manifest = read_json(base / 'manifest.json')
        names = ['tax_ids.npy', 'incidence.npz', 'positions.npy', 'tree_audit.json']
        if method == 'deepphylo_taxonomy':
            names.append('embeddings.npy')
        for name in names:
            require(sha(base / name) == manifest['artifacts'][name], f'Tree artifact hash mismatch: {base / name}')
        require(np.array_equal(np.load(base / 'tax_ids.npy'), data.tax_ids), 'Tree taxon order mismatch')
        trees[method] = sparse.load_npz(base / 'incidence.npz')
    h = np.load(ROOT / 'data_processing/hilbert_encoding/hilbert.npy')
    require(np.array_equal(np.sort(h), np.arange(len(data.tax_ids))), 'Invalid Hilbert permutation')
    pi = np.load(TRAIN / 'classical_order_control/permutation_seed0.npy')
    require(np.array_equal(pi, np.random.default_rng(0).permutation(len(data.tax_ids))), 'Invalid random permutation')
    results = []
    for level in range(1, 6):
        for fold in range(1, 6):
            shared = TRAIN / f'shared_cache/level_{level}/fold_{fold}/normalization.npz'
            # Includes every row only in data alignment checks; transform check uses 32 rows.
            for method in ('genus', 'deepphylo_taxonomy', 'popphy_cnn'):
                norm = shared if method == 'genus' else TRAIN / f'phylogeny_baselines/{method}/cache/level_{level}/fold_{fold}/normalization.npz'
                _, x, present = next(data.batches(np.arange(min(32, len(data.ids))), norm,
                    method='popphy_cnn' if method == 'popphy_cnn' else 'genus', incidence=trees.get(method)))
                if method == 'genus':
                    reference = x
                    require(np.array_equal(x[:, h][:, np.argsort(h)], x), 'Hilbert inverse mismatch')
                    require(np.array_equal(x[:, pi][:, np.argsort(pi)], x), 'Random inverse mismatch')
                if method == 'deepphylo_taxonomy':
                    require(np.array_equal(reference, x), 'DeepPhylo/shared genus transform mismatch')
                results.append(dict(level=level, fold=fold, method=method, samples=len(x),
                    columns=x.shape[1], minimum=float(x.min()), maximum=float(x.max()),
                    below_zero=int((x < 0).sum()), above_one=int((x > 1).sum()),
                    raw_presence_minimum=int(present.sum(1).min()), normalization_sha256=sha(norm)))
    return results


def main():
    data = ExternalData()
    report = HERE / 'reports'
    report.mkdir(exist_ok=True)
    labels = {}
    for level in range(1, 6):
        classes = read_json(DATA / f'training_classes/level_{level}.json')
        y = data.labels_for(level, classes)
        values = [r[str(level)] for r in data.labels]
        labels[str(level)] = dict(known=int((y >= 0).sum()), missing=int((y == -1).sum()), unseen=int((y == -2).sum()),
            unseen_counts=dict(Counter(v for v, yi in zip(values, y) if yi == -2)),
            absent_training_classes=[classes[str(i)] for i in range(len(classes)) if not np.any(y == i)])
        # Keep every sample and its exclusion reason; evaluation selects y >= 0 per level.
        with (report / f'level_{level}_labels.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['row_index', 'sample_id', 'label', 'class_index', 'status'])
            writer.writerows((i, sid, value, int(yi), 'known' if yi >= 0 else 'missing' if yi == -1 else 'unseen')
                             for i, (sid, value, yi) in enumerate(zip(data.ids, values, y)))
    records = inventory(data)
    transforms = audit_preprocessing(data)
    write_json(report / 'preprocessing_checks.json', transforms)
    with (report / 'model_inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Record every source/document revision inspected; exclude this new foundation.
    sources = {}
    for base in (TRAIN, ROOT / 'data_processing'):
        for path in sorted(base.rglob('*')):
            if HERE in path.parents or path.suffix not in ('.py', '.md', '.sh') or not path.is_file():
                continue
            sources[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    write_json(report / 'source_inventory.json', sources)
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(), phase='foundation_only_no_inference',
                   samples=len(data.ids), features=len(data.tax_ids),
                   checks=dict(package_hashes=True, sample_alignment=True, feature_alignment=True,
                               training_sample_id_disjoint=True, training_class_snapshots_all_folds=True),
                   labels=labels, model_statuses=dict(Counter(r['status'] for r in records)),
                   per_model={name: dict(Counter(r['status'] for r in records if r['model'] == name)) for name in sorted({r['model'] for r in records})},
                   inventory_semantics='candidate means files/configs present; tensor loading and full inference remain untested',
                   checkpoint_policy='formal paths only; result.json required; no smoke/last.pt fallback; no refitting',
                   source_files=len(sources))
    write_json(report / 'preflight.json', summary)
    print({k: summary[k] for k in ('samples', 'features', 'model_statuses', 'per_model')})


if __name__ == '__main__':
    main()
