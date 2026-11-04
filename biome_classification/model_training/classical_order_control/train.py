"""Random forest and logistic regression feature-order controls."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import platform
import tempfile
import time
import warnings
from pathlib import Path
import fcntl
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, roc_auc_score

MODEL = 'random_forest'
BASE = Path(__file__).resolve().parent
CACHE = BASE.parent / 'shared_cache'
ORDERS = ('rowwise', 'random')
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, data):
    atomic(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def csv_bytes(fields, rows):
    s = io.StringIO(newline='')
    w = csv.writer(s)
    w.writerow(fields)
    w.writerows(rows)
    return s.getvalue().encode()


def permutation():
    expected = np.random.default_rng(0).permutation(4096)
    path = BASE / 'permutation_seed0.npy'
    if not path.exists():
        s = io.BytesIO()
        np.save(s, expected, allow_pickle=False)
        atomic(path, s.getvalue())
    if not np.array_equal(np.load(path, allow_pickle=False), expected):
        raise ValueError('permutation_seed0.npy differs from default_rng(0).permutation(4096)')
    return expected, sha(path)


def load_cache(level, fold, smoke):
    folder = CACHE / f'level_{level}' / f'fold_{fold}'
    names = ['manifest.json', 'classes.json', 'tax_ids.npy', 'normalization.npz']
    names += [f'{part}_{suffix}.npy' for part in ('train', 'val', 'test') for suffix in ('x', 'y', 'ids', 'row_indices')]
    missing = [str(folder / n) for n in names if not (folder / n).is_file()]
    if missing:
        raise FileNotFoundError('Missing cache: ' + ', '.join(missing) + '\nPrepare separately with: $PYTHON model_training/train.py --encoding hilbert --prepare-only --output-dir model_training/results')
    manifest = json.loads((folder / 'manifest.json').read_text())
    payload = json.dumps(manifest['provenance'], sort_keys=True).encode()
    if hashlib.sha256(payload).hexdigest() != manifest['signature']:
        raise ValueError('Invalid cache provenance signature')
    if manifest['checks'].get('fit_partition') != 'train':
        raise ValueError('Cache normalization was not fitted on train')
    classes = json.loads((folder / 'classes.json').read_text())
    if set(classes) != {str(i) for i in range(len(classes))}:
        raise ValueError('Noncontiguous class mapping')
    tax = np.load(folder / 'tax_ids.npy', allow_pickle=False)
    if tax.shape != (4096,):
        raise ValueError('Expected 4096 tax_ids')
    with np.load(folder / 'normalization.npz', allow_pickle=False) as norm:
        if not np.array_equal(norm['tax_ids'], tax):
            raise ValueError('Normalization tax_ids mismatch')
    parts, all_ids, all_rows = {}, [], []
    for part in ('train', 'val', 'test'):
        x, y, ids, rows = [np.load(folder / f'{part}_{s}.npy', mmap_mode='r', allow_pickle=False) for s in ('x', 'y', 'ids', 'row_indices')]
        if x.shape != (len(y), 4096) or x.dtype != np.float32 or len(ids) != len(y) or len(rows) != len(y) or len(y) != manifest['sizes'][part]:
            raise ValueError(f'Cache shape/dtype mismatch: {part}')
        if not np.array_equal(np.unique(y), np.arange(len(classes))):
            raise ValueError(f'Class coverage mismatch: {part}')
        all_ids.append(np.asarray(ids)); all_rows.append(np.asarray(rows))
        if smoke:
            ix = np.unique(np.r_[np.arange(min(64, len(y))), np.unique(y, return_index=True)[1]])
            x, y, ids = x[ix], y[ix], ids[ix]
        parts[part] = (x, np.asarray(y), np.asarray(ids))
    for arrays in (all_ids, all_rows):
        merged = np.concatenate(arrays)
        if len(np.unique(merged)) != len(merged):
            raise ValueError('Duplicate/overlapping cache samples')
    # Hash actual cache bytes, including matrices, rather than trusting a manifest alone.
    identity = dict(path=str(folder.resolve()), signature=manifest['signature'], files={n: sha(folder/n) for n in names})
    return parts, classes, tax, identity


def metrics(y, p):
    return dict(macro_f1=float(f1_score(y, p.argmax(1), labels=np.arange(p.shape[1]), average='macro', zero_division=0)),
                macro_auroc=float(roc_auc_score(np.eye(p.shape[1], dtype=np.int8)[y], p, average='macro')))


def estimator(args, iterations):
    if MODEL == 'random_forest':
        return RandomForestClassifier(n_estimators=100, criterion='gini', max_features='sqrt', max_depth=None,
            min_samples_split=2, min_samples_leaf=1, bootstrap=True, class_weight=None, random_state=0, n_jobs=args.n_jobs)
    return LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, fit_intercept=True,
                              class_weight=None, tol=1e-6, max_iter=iterations)


def ordered(x, order, pi):
    if order == 'rowwise':
        return x
    out = np.ascontiguousarray(x[:, pi])
    # Blockwise verification avoids allocating another full matrix.
    for start in range(0, len(x), 1024):
        if not np.array_equal(out[start:start+1024, np.argsort(pi)], x[start:start+1024]):
            raise ValueError('Permutation changed input values')
    return out


def read_predictions(path, classes):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    ids = np.array([r['sample_id'] for r in rows])
    y = np.array([int(r['true_index']) for r in rows])
    p = np.array([[float(r[f'probability_{i}']) for i in range(len(classes))] for r in rows])
    if not len(rows) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid saved probabilities')
    for r, yi, pred in zip(rows, y, p.argmax(1)):
        if r['true_class'] != classes[str(yi)] or int(r['predicted_index']) != pred or r['predicted_class'] != classes[str(pred)]:
            raise ValueError('Saved prediction class mismatch')
    return ids, y, p


def complete(out, request, classes):
    if not (out/'result.json').exists():
        if (out/'config.json').exists():
            old = json.loads((out/'config.json').read_text())
            if old['request'] != request:
                raise ValueError(f'Incomplete result configuration mismatch: {out}')
        return None
    if not all((out/n).is_file() for n in ('config.json', 'test_predictions.csv')):
        raise ValueError(f'Incomplete committed result: {out}')
    result = json.loads((out/'result.json').read_text())
    config = json.loads((out/'config.json').read_text())
    if config['request'] != request or result.get('status') != 'complete' or result['config_sha256'] != sha(out/'config.json'):
        raise ValueError(f'Result configuration mismatch: {out}')
    for name, digest in result['artifacts'].items():
        if not (out/name).is_file() or sha(out/name) != digest:
            raise ValueError(f'Result artifact mismatch: {out/name}')
    if request['save_model'] and 'model.joblib' not in result['artifacts']:
        raise ValueError(f'Saved estimator missing from completion record: {out}')
    ids, y, p = read_predictions(out/'test_predictions.csv', classes)
    if metrics(y,p) != result['metrics']:
        raise ValueError(f'Result metrics mismatch: {out}')
    return result


def run_fold(args, level, fold, pi, pi_hash):
    parts, classes, tax, identity = load_cache(level, fold, args.smoke)
    root = BASE / ('smoke' if args.smoke else 'results') / MODEL / f'level_{level}' / f'fold_{fold}'
    if args.retrain and root.exists():
        # Preserve the previous fold, including its completion markers and predictions.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = BASE / 'archive' / ('smoke' if args.smoke else 'results') / MODEL / f'level_{level}' / f'fold_{fold}_{stamp}'
        backup.parent.mkdir(parents=True, exist_ok=True)
        root.rename(backup)
        print(f'ARCHIVED {backup}', flush=True)
    mapping = [dict(output_column=i, original_column=int(j), tax_id=str(tax[j])) for i,j in enumerate(pi)]
    write_json(root/'random_column_mapping.json', dict(permutation_sha256=pi_hash, columns=mapping))
    request = dict(model=MODEL, level=level, fold=fold, smoke=args.smoke, smoke_prefix_samples=64 if args.smoke else None,
                   cache=identity, permutation_sha256=pi_hash, code_sha256=sha(__file__), versions=VERSIONS,
                   save_model=args.save_model, initial_parameters=estimator(args,2000).get_params(),
                   convergence_policy='paired 2000 then paired 5000; fail if either still warns', validation_use='read/shape/class diagnostics only')
    requests = {o:dict(request, order=o) for o in ORDERS}
    existing = {o:complete(root/o, requests[o], classes) for o in ORDERS}
    if all(existing.values()):
        print(f'SKIP {MODEL} level={level} fold={fold}', flush=True)
        return
    # A fold is committed only after both fits have passed convergence checks.
    fitted, attempts = {}, []
    for iterations in ([2000,5000] if MODEL == 'logistic_regression' else [2000]):
        fitted = {}
        converged = True
        for order in ORDERS:
            model = estimator(args,iterations)
            x = ordered(parts['train'][0],order,pi)
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(x,parts['train'][1])
            elapsed = time.perf_counter()-start
            notes = [dict(category=w.category.__name__,message=str(w.message)) for w in caught]
            bad = any(issubclass(w.category,ConvergenceWarning) for w in caught)
            converged &= not bad
            attempts.append(dict(order=order,max_iter=iterations if MODEL=='logistic_regression' else None,training_seconds=elapsed,warnings=notes))
            if not np.array_equal(model.classes_,np.arange(len(classes))):
                raise ValueError('estimator.classes_ does not match classes.json')
            fitted[order]=(model,elapsed,notes)
            del x
        if converged:
            break
    if not converged:
        write_json(root/'failure.json',dict(status='failed',reason='ConvergenceWarning after paired max_iter=5000',request=request,attempts=attempts))
        raise RuntimeError(f'Both orders excluded: convergence failure in {root}')
    predictions = {}
    for order in ORDERS:
        model, elapsed, notes = fitted[order]
        x = ordered(parts['test'][0],order,pi)
        start=time.perf_counter()
        p=model.predict_proba(x)
        prediction_seconds=time.perf_counter()-start
        del x
        if p.shape != (len(parts['test'][1]),len(classes)) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1):
            raise ValueError('Invalid predict_proba output')
        predictions[order]=(p,prediction_seconds)
    # Incomplete pairs are refitted together; invalidate old markers before paired commit.
    for order in ORDERS:
        marker = root/order/'result.json'
        if marker.exists(): marker.unlink()
    # Write result.json last as the completion marker; every file uses atomic replace.
    for order in ORDERS:
        out=root/order
        model,elapsed,notes=fitted[order]
        p,prediction_seconds=predictions[order]
        ids,y=parts['test'][2],parts['test'][1]
        config=dict(request=requests[order],actual_parameters=model.get_params(),classes=classes,
                    feature_mapping='random_column_mapping.json in parent fold directory' if order=='random' else 'cache tax_ids.npy, identity column indices',
                    n_iter=getattr(model,'n_iter_',np.array([],dtype=int)).tolist())
        write_json(out/'config.json',config)
        rows=([sid,int(yi),classes[str(int(yi))],int(prob.argmax()),classes[str(int(prob.argmax()))],*prob.tolist()] for sid,yi,prob in zip(ids,y,p))
        atomic(out/'test_predictions.csv',csv_bytes(['sample_id','true_index','true_class','predicted_index','predicted_class']+[f'probability_{i}' for i in range(len(classes))],rows))
        artifacts={'test_predictions.csv':sha(out/'test_predictions.csv')}
        if args.save_model:
            s=io.BytesIO(); joblib.dump(model,s)
            atomic(out/'model.joblib',s.getvalue()); artifacts['model.joblib']=sha(out/'model.joblib')
        write_json(out/'result.json',dict(status='complete',metrics=metrics(y,p),training_seconds=elapsed,
            total_training_seconds_including_retries=sum(a['training_seconds'] for a in attempts if a['order']==order),
            prediction_seconds=prediction_seconds,samples={k:len(v[1]) for k,v in parts.items()},classes=len(classes),
            warnings=notes,attempts=attempts,versions=VERSIONS,config_sha256=sha(out/'config.json'),artifacts=artifacts))
    if (root/'failure.json').exists():
        (root/'failure.json').unlink()
    print(f'COMPLETE {MODEL} level={level} fold={fold}',flush=True)


def aggregate(pi_hash):
    root=BASE/'results'/MODEL
    comparisons=[]
    for level in range(1,6):
        parent=root/f'level_{level}'
        paths=[parent/f'fold_{fold}'/order for fold in range(1,6) for order in ORDERS]
        if not all((p/'result.json').exists() for p in paths):
            continue
        results={o:[] for o in ORDERS}; pooled={o:[] for o in ORDERS}; paired=[]
        common_request=None; reference_classes=None
        for fold in range(1,6):
            cache_classes=json.loads((CACHE/f'level_{level}'/f'fold_{fold}'/'classes.json').read_text())
            if reference_classes is None: reference_classes=cache_classes
            if reference_classes != cache_classes: raise ValueError('Cross-fold classes mismatch')
            _,_,_,actual_identity=load_cache(level,fold,False)
            for order in ORDERS:
                out=parent/f'fold_{fold}'/order
                cfg=json.loads((out/'config.json').read_text()); req=cfg['request']
                if req['cache']!=actual_identity or req['permutation_sha256']!=pi_hash or req['code_sha256']!=sha(__file__) or req['smoke'] or req['versions']!=VERSIONS:
                    raise ValueError('Stale result in aggregation')
                protocol={k:v for k,v in req.items() if k not in ('fold','order','cache')}
                if common_request is None: common_request=protocol
                if protocol!=common_request: raise ValueError('Mixed protocols in aggregation')
                result=complete(out,req,cache_classes)
                results[order].append(result)
                pooled[order].append(read_predictions(out/'test_predictions.csv',cache_classes))
            a,b=pooled['rowwise'][-1],pooled['random'][-1]
            if not np.array_equal(a[0],b[0]) or not np.array_equal(a[1],b[1]): raise ValueError('Paired samples mismatch')
            paired.append(dict(fold=fold,**{k:results['rowwise'][-1]['metrics'][k]-results['random'][-1]['metrics'][k] for k in ('macro_f1','macro_auroc')},
                prediction_disagreement_fraction=float(np.mean(a[2].argmax(1)!=b[2].argmax(1))), probability_max_absolute_difference=float(np.max(np.abs(a[2]-b[2])))))
        summary=dict(level=level,folds=5,difference_direction='rowwise - random',paired_folds=paired,orders={})
        merged={}
        for order in ORDERS:
            ids,y,p=[np.concatenate([r[i] for r in pooled[order]]) for i in range(3)]
            if len(np.unique(ids))!=len(ids): raise ValueError('Nonexclusive OOF samples')
            merged[order]=(ids,y,p)
            summary['orders'][order]=dict(metrics={k:dict(mean=float(np.mean([r['metrics'][k] for r in results[order]])),std=float(np.std([r['metrics'][k] for r in results[order]],ddof=1))) for k in ('macro_f1','macro_auroc')},pooled_oof=metrics(y,p))
            for k in ('macro_f1','macro_auroc'):
                entry=summary['orders'][order]
                comparisons.append([level,order,k,entry['metrics'][k]['mean'],entry['metrics'][k]['std'],entry['pooled_oof'][k]])
        a,b=merged['rowwise'][2],merged['random'][2]
        summary['pooled_prediction_disagreement_fraction']=float(np.mean(a.argmax(1)!=b.argmax(1)))
        summary['pooled_probability_max_absolute_difference']=float(np.max(np.abs(a-b)))
        write_json(parent/'summary.json',summary)
    if comparisons:
        atomic(root/'comparison.csv',csv_bytes(['level','order','metric','five_fold_mean','five_fold_std_ddof1','pooled_oof'],comparisons))


def main():
    global MODEL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level',nargs='+',type=int,choices=range(1,6),default=list(range(1,6)))
    parser.add_argument('--fold',nargs='+',type=int,choices=range(1,6),default=list(range(1,6)))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='Save model.joblib (enabled by default; retained for compatibility)')
    parser.add_argument('--retrain', action='store_true',
                        help='Archive existing selected folds and fit again; omit when resuming')
    parser.add_argument('--model', choices=['random_forest','logistic_regression'], required=True)
    parser.add_argument('--n-jobs',type=int,default=8)
    args=parser.parse_args()
    MODEL = args.model
    if MODEL=='random_forest' and args.n_jobs==0: parser.error('--n-jobs cannot be zero')
    if args.smoke: args.level,args.fold=[1],[1]
    # Serialize both entry points, protecting permutation creation and result commits.
    with (BASE/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        pi,pi_hash=permutation()
        for level in args.level:
            for fold in args.fold:
                run_fold(args,level,fold,pi,pi_hash)
        if not args.smoke: aggregate(pi_hash)


if __name__=='__main__':
    main()
