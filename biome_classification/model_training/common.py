"""One implementation of data preparation, splitting, normalization and evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
import sklearn
import scipy
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from models import build_model, encode

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data_processing'
FIXED = BASE / 'mgnify_matched_counts_genus_ncbi_top4096_min20'
CONFIG = json.loads(Path(__file__).with_name('config.json').read_text())

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def seed_all():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def fingerprint():
    paths = [FIXED / n for n in ('counts.npz','biomes.csv','features.csv','source_row_indices.npy','source_column_indices.npy')]
    paths += [BASE / 'mgnify_matched_counts_genus_ncbi/counts.npz', BASE / 'hilbert_encoding/hilbert.npy']
    hashes = {}
    for p in paths:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
        hashes[str(p.relative_to(ROOT))] = h.hexdigest()
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py') if p.name != 'check.py'}
    # Only explicitly reviewed, exact code revisions share a training signature.
    # Device validation and aggregation checks do not change training or cached data.
    compatibility = json.loads(Path(__file__).with_name('code_signature_compatibility.json').read_text())
    actual_common_hash = code_hashes['common.py']
    code_hashes['common.py'] = compatibility.get(actual_common_hash, actual_common_hash)
    payload = {'code': code_hashes, 'files': hashes, 'config': CONFIG, 'preprocessing_schema': 1, 'sklearn': sklearn.__version__, 'numpy': np.__version__}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload

def read_labels(level):
    with (FIXED / 'biomes.csv').open(newline='') as f: rows = list(csv.DictReader(f))
    ids = np.array([r['sample_id'] for r in rows])
    values = np.array([r[str(level)] for r in rows])
    missing = np.array([v.strip().casefold() in ('', 'na', 'none') for v in values])
    names, counts = np.unique(values[~missing], return_counts=True)
    retained = names[counts >= 100]
    keep = ~missing & np.isin(values, retained)
    suspects = {str(n): int(c) for n,c in zip(names,counts) if n.strip().casefold() in ('nan','null','n/a','unknown','undefined','missing','not applicable','not available','-','.') or n != n.strip()}
    raw_names, raw_counts = np.unique(values, return_counts=True)
    report = {'raw_class_counts_including_empty':dict(zip(raw_names.tolist(),raw_counts.tolist())), 'level': level, 'before_samples':len(ids), 'empty_samples':int(missing.sum()), 'after_samples':int(keep.sum()), 'before_class_counts':dict(zip(names.tolist(),counts.tolist())), 'after_class_counts':{str(n):int(c) for n,c in zip(names,counts) if c >= 100}, 'suspected_labels_retained_unless_count_below_100':suspects}
    if len(retained) < 2: raise ValueError('Fewer than two retained classes')
    return ids, values, np.flatnonzero(keep), retained, report

def prepare(level, fold, cache, signature, provenance):
    folder = cache / f'level_{level}' / f'fold_{fold}'
    manifest = folder / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['signature'] != signature: raise ValueError('Cache provenance mismatch; use a new cache directory')
        return folder
    ids, values, keep, classes, report = read_labels(level)
    y = np.searchsorted(classes, values[keep])
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(keep,y))
    pool, test = splits[fold - 1]
    train, val = train_test_split(pool, test_size=0.2, stratify=y[pool], random_state=0)
    indices = {'train':keep[train], 'val':keep[val], 'test':keep[test]}
    assert len(set(ids.tolist())) == len(ids)
    sets = [set(v.tolist()) for v in indices.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert set.union(*sets) == set(keep.tolist())
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder.parent / 'label_report.json', report)
    source = sparse.load_npz(BASE / 'mgnify_matched_counts_genus_ncbi/counts.npz')
    rows = np.load(FIXED / 'source_row_indices.npy')
    cols = np.load(FIXED / 'source_column_indices.npy')
    with np.load(BASE / 'mgnify_matched_counts_genus_ncbi/counts.npz') as z:
        assert np.array_equal(z['sample_ids'][rows], ids)
        source_taxids = z['tax_ids'][cols]
    with np.load(FIXED / 'counts.npz') as z:
        assert np.array_equal(z['sample_ids'], ids)
        assert np.array_equal(z['tax_ids'], source_taxids)
    fixed = sparse.load_npz(FIXED / 'counts.npz')
    assert fixed.shape == (238419,4096)
    assert (fixed != source[rows][:,cols]).nnz == 0
    denominators = np.asarray(source.sum(axis=1)).ravel()[rows]
    zero_ids = ids[denominators == 0].tolist()
    write_json(folder / 'zero_counts.json', {'sample_ids':zero_ids,'count':len(zero_ids)})
    if zero_ids: raise ValueError('Zero total count in selected dataset; inspect zero_counts.json')
    lo = np.full(4096, np.inf); hi = np.full(4096, -np.inf)
    for offset in range(0,len(indices['train']),1024):
        ix = indices['train'][offset:offset+1024]
        relative = fixed[ix].toarray().astype(np.float64) / denominators[ix,None]
        lo = np.minimum(lo, relative.min(axis=0)); hi = np.maximum(hi, relative.max(axis=0))
    constant = hi == lo
    scale = np.where(constant, 1., hi-lo)
    np.savez(folder / 'normalization.npz', minimum=lo, maximum=hi, constant=constant, tax_ids=source_taxids)
    ranges = {}
    for part, ix in indices.items():
        output = np.lib.format.open_memmap(folder / f'{part}_x.npy', mode='w+', dtype=np.float32, shape=(len(ix),4096))
        stats = {'below_zero':0,'above_one':0,'minimum':float('inf'),'maximum':float('-inf')}
        for offset in range(0,len(ix),1024):
            chunk = ix[offset:offset+1024]
            x = fixed[chunk].toarray().astype(np.float64) / denominators[chunk,None]
            x = ((x-lo)/scale).astype(np.float32)
            x[:,constant] = 0
            assert np.isfinite(x).all()
            stats['below_zero'] += int((x<0).sum()); stats['above_one'] += int((x>1).sum())
            stats['minimum'] = min(stats['minimum'],float(x.min())); stats['maximum'] = max(stats['maximum'],float(x.max()))
            output[offset:offset+len(chunk)] = x
        output.flush(); del output
        ranges[part] = stats
        np.save(folder / f'{part}_y.npy', np.searchsorted(classes,values[ix]).astype(np.int64))
        np.save(folder / f'{part}_ids.npy', ids[ix])
        np.save(folder / f'{part}_row_indices.npy', ix)
    write_json(folder / 'classes.json', {str(i):c for i,c in enumerate(classes.tolist())})
    np.save(folder / 'tax_ids.npy', source_taxids)
    write_json(manifest, {'signature':signature,'provenance':provenance,'ranges':ranges,'constant_columns':int(constant.sum()), 'checks':{'ids_disjoint':True,'matrix_source_alignment':True,'fit_partition':'train'}, 'feature_selection':CONFIG['feature_selection'], 'sizes':{k:len(v) for k,v in indices.items()}})
    return folder

def load_parts(folder, limit=0):
    result = {}
    for part in ('train','val','test'):
        x = np.load(folder / f'{part}_x.npy', mmap_mode='r')
        y = np.load(folder / f'{part}_y.npy')
        ids = np.load(folder / f'{part}_ids.npy')
        ix = np.arange(len(y))
        if limit:
            # Include every class in smoke evaluation, even with a very small requested limit.
            first = np.unique(y, return_index=True)[1]
            ix = np.unique(np.concatenate([first,np.arange(min(limit,len(y)))]))
        result[part] = (torch.from_numpy(np.array(x[ix] if limit else x,copy=True)), torch.from_numpy(y[ix]), ids[ix])
    return result

def metrics(y,p):
    onehot = np.eye(p.shape[1],dtype=np.int8)[y]
    return {'macro_f1':float(f1_score(y,p.argmax(1),labels=np.arange(p.shape[1]),average='macro',zero_division=0)), 'macro_auroc':float(roc_auc_score(onehot,p,average='macro'))}

def evaluate(model, loader, device, encoding, h, probabilities=False):
    model.eval(); loss = 0.; size = 0; probs = []; ys = []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            logits = model(encode(x,encoding,h))
            loss += torch.nn.functional.cross_entropy(logits,y,reduction='sum').item(); size += len(y)
            if probabilities: probs.append(logits.softmax(1).cpu().numpy()); ys.append(y.cpu().numpy())
    return loss/size, (np.concatenate(ys),np.concatenate(probs)) if probabilities else None

def save_predictions(path, ids, y, p, classes):
    with path.open('w',newline='') as f:
        w = csv.writer(f); w.writerow(['sample_id','true_index','true_class','predicted_index','predicted_class'] + [f'probability_{i}' for i in range(len(classes))])
        for sid,actual,prob in zip(ids,y,p):
            pred = int(prob.argmax()); w.writerow([sid,int(actual),classes[str(int(actual))],pred,classes[str(pred)],*prob.tolist()])

def train_run(args, encoding, depth, level, fold, folder, signature):
    out = args.output_dir / ('smoke' if args.smoke else 'experiments') / f'level_{level}' / f'{encoding}_resnet{depth}' / f'fold_{fold}'
    out.mkdir(parents=True,exist_ok=True)
    versions = {'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
    config = {**CONFIG,'encoding':encoding,'depth':depth,'level':level,'fold':fold,'signature':signature,'device':args.device,'threads':args.threads,'versions':versions,'smoke':args.smoke,'max_epochs':args.epochs,'smoke_samples':args.smoke_samples if args.smoke else None,'precision':'float32','augmentation':None,'clip':False,'cache':str(folder.resolve())}
    if (out / 'config.json').exists():
        saved_config = json.loads((out / 'config.json').read_text())
        # Device selection is a runtime setting; checkpoints use map_location.
        saved_comparable = {k: v for k, v in saved_config.items() if k != 'device'}
        current_comparable = {k: v for k, v in config.items() if k != 'device'}
        if saved_comparable != current_comparable:
            differences = sorted(k for k in saved_comparable.keys() | current_comparable.keys() if saved_comparable.get(k) != current_comparable.get(k))
            raise ValueError(f'Configuration mismatch in {out}: {differences}; choose a new output directory')
    if (out / 'result.json').exists(): return
    write_json(out / 'config.json',config)
    data = load_parts(folder,args.smoke_samples if args.smoke else 0)
    classes = json.loads((folder / 'classes.json').read_text())
    seed_all()
    model = build_model(encoding,depth,len(classes)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    write_json(out / 'effective_optimizer.json', {k:v for k,v in optimizer.param_groups[0].items() if k != 'params'})
    generator = torch.Generator().manual_seed(0)
    loaders = {p:DataLoader(TensorDataset(x,y),batch_size=32,shuffle=p=='train',num_workers=0,generator=generator if p=='train' else None) for p,(x,y,_) in data.items()}
    h = torch.as_tensor(np.load(BASE / 'hilbert_encoding/hilbert.npy'),device=args.device)
    best = float('inf'); stale = 0; start_epoch = 1; history = []; elapsed = 0.; best_epoch = 0
    last = out / 'last.pt'
    if last.exists():
        state = torch.load(last,map_location=args.device,weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        best, stale, start_epoch, history, elapsed, best_epoch = state['best'],state['stale'],state['epoch']+1,state['history'],state['elapsed'],state['best_epoch']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'].cpu()); generator.set_state(state['loader_rng'].cpu())
        if torch.cuda.is_available() and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    started = time.perf_counter()
    for epoch in range(start_epoch,args.epochs+1):
        if stale >= 10: break
        model.train(); total = 0.; n = 0
        for x,y in loaders['train']:
            x,y = x.to(args.device),y.to(args.device)
            optimizer.zero_grad(); loss = torch.nn.functional.cross_entropy(model(encode(x,encoding,h)),y)
            loss.backward(); optimizer.step(); total += loss.item()*len(y); n += len(y)
        vl,_ = evaluate(model,loaders['val'],args.device,encoding,h)
        if not np.isfinite(vl): raise RuntimeError('Nonfinite validation loss')
        if vl < best:
            best,stale,best_epoch = vl,0,epoch
            torch.save(model.state_dict(),out / 'best.pt')
        else: stale += 1
        history.append({'epoch':epoch,'train_loss':total/n,'validation_loss':vl})
        state = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,stale=stale,epoch=epoch,history=history,elapsed=elapsed+time.perf_counter()-started,best_epoch=best_epoch,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),loader_rng=generator.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(state,out / 'last.tmp'); (out / 'last.tmp').replace(last)
        print(f'{encoding} ResNet{depth} level={level} fold={fold} epoch={epoch} val_loss={vl:.6f}',flush=True)
    model.load_state_dict(torch.load(out / 'best.pt',map_location=args.device,weights_only=True))
    _, (y,p) = evaluate(model,loaders['test'],args.device,encoding,h,True)
    save_predictions(out / 'predictions.csv',data['test'][2],y,p,classes)
    np.savez(out / 'predictions.npz',ids=data['test'][2],y=y,probabilities=p)
    write_json(out / 'history.json',history)
    write_json(out / 'result.json',{**metrics(y,p),'best_epoch':best_epoch,'best_validation_loss':best,'parameters':sum(t.numel() for t in model.parameters()),'seconds':elapsed+time.perf_counter()-started,'signature':signature,'config':config})

def aggregate(args,encoding,depth,level):
    parent = args.output_dir / ('smoke' if args.smoke else 'experiments') / f'level_{level}' / f'{encoding}_resnet{depth}'
    files = [parent / f'fold_{i}' / 'result.json' for i in range(1,6)]
    if not all(p.exists() for p in files): return
    results = [json.loads(p.read_text()) for p in files]
    if len({r['signature'] for r in results}) != 1: raise ValueError('Inconsistent fold provenance')
    predictions = [np.load(p.parent / 'predictions.npz') for p in files]
    ids = np.concatenate([p['ids'] for p in predictions]); y = np.concatenate([p['y'] for p in predictions]); prob = np.concatenate([p['probabilities'] for p in predictions])
    assert len(np.unique(ids)) == len(ids)
    classes = json.loads((args.cache_dir / f'level_{level}/fold_1/classes.json').read_text())
    save_predictions(parent / 'oof_predictions.csv',ids,y,prob,classes)
    write_json(parent / 'summary.json',{'folds':5,'metrics':{k:{'mean':float(np.mean([r[k] for r in results])),'std':float(np.std([r[k] for r in results],ddof=1))} for k in ('macro_f1','macro_auroc')},'pooled_oof':metrics(y,prob)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', choices=('hilbert','rowwise','1d'), default='hilbert')
    parser.add_argument('--level',type=int,choices=range(1,6),nargs='+',default=list(range(1,6)))
    parser.add_argument('--fold',type=int,choices=range(1,6),nargs='+',default=list(range(1,6)))
    parser.add_argument('--resnet',type=int,choices=(18,34,50),nargs='+',default=[18,34,50])
    parser.add_argument('--output-dir',type=Path,default=ROOT / 'model_training/results')
    parser.add_argument('--cache-dir',type=Path,default=ROOT / 'model_training/shared_cache')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-samples',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    encoding = args.encoding
    if not args.smoke and args.epochs != 200: parser.error('Formal experiments require --epochs 200; use --smoke for checks')
    if args.epochs < 1 or args.smoke_samples < 1 or args.threads < 1: parser.error('epochs, smoke-samples and threads must be positive')
    torch.set_num_threads(args.threads)
    signature,provenance = fingerprint()
    write_json(args.output_dir / 'data_code_audit.json',{'signature':signature,'provenance':provenance,'deviations':['Fixed 4096 genera and min20 sample cohort selected before splitting using global data; fold-local feature selection is intentionally not performed.'],'denominator':'all 10437 NCBI-matched genera before selecting fixed columns','zero_count_policy':'report and fail if present in fixed cohort', 'suspected_labels':{str(level):read_labels(level)[4]['suspected_labels_retained_unless_count_below_100'] for level in args.level},'code_status':'new shared training implementation; old results excluded'})
    hilbert = np.load(BASE / 'hilbert_encoding/hilbert.npy')
    assert hilbert.shape == (4096,) and np.array_equal(np.sort(hilbert),np.arange(4096)), 'Invalid Hilbert permutation'
    for level in args.level:
        for fold in args.fold:
            import fcntl
            args.cache_dir.mkdir(parents=True,exist_ok=True)
            with (args.cache_dir / '.prepare.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                folder = prepare(level,fold,args.cache_dir,signature,provenance)
            if not args.prepare_only:
                for depth in ([18] if encoding == '1d' else args.resnet): train_run(args,encoding,depth,level,fold,folder,signature)
        if not args.prepare_only:
            for depth in ([18] if encoding == '1d' else args.resnet): aggregate(args,encoding,depth,level)
