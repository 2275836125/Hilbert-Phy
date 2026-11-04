"""Method-local execution using the existing common.py protocol primitives."""
import argparse
import importlib
import json
import platform
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import common
sys.path.insert(0,str(HERE))
from tree import ROOT, write_json, sha, digest
from data import provenance,prepare_tree,read_counts,shared_fold,fit_transform,FoldDataset

NAMES = {'popphy_cnn':'共享协议下的 PopPhy-CNN 适配基线',
         'deepphylo_taxonomy':'基于 NCBI 分类树距离的 genus 适配版 DeepPhylo'}

def forward(model,x,present,args,positions,shape):
    if args.method == 'deepphylo_taxonomy': return model(x,present)
    matrix = x.new_zeros((len(x),1,*shape))
    matrix[:,0,positions[:,0],positions[:,1]] = x
    return model(matrix)

def evaluate(model,loader,args,positions,shape,probabilities=False):
    model.eval(); total = 0.; n = 0; ys = []; ps = []
    with torch.no_grad():
        for x,present,y in loader:
            x,present,y = x.to(args.device),present.to(args.device),y.to(args.device)
            logits = forward(model,x,present,args,positions,shape)
            total += torch.nn.functional.cross_entropy(logits,y,reduction='sum').item(); n += len(y)
            if probabilities: ys.append(y.cpu().numpy()); ps.append(logits.softmax(1).cpu().numpy())
    return total/n,(np.concatenate(ys),np.concatenate(ps)) if probabilities else None

def train(args,parts,classes,datasets,signature,tree):
    out = args.output_dir/('smoke' if args.smoke else 'formal')/f'level_{args.current_level}'/f'fold_{args.current_fold}'
    out.mkdir(parents=True,exist_ok=True)
    config = {'method':NAMES[args.method],'method_key':args.method,'signature':signature,
              'tree_signature':tree['tree_signature'],'level':args.current_level,'fold':args.current_fold,
              'shared_protocol':common.CONFIG,'batch_size':32,'max_epochs':args.epochs,'patience':10,
              'device':args.device,'threads':args.threads,'smoke':args.smoke,
              'smoke_samples':args.smoke_samples if args.smoke else None,
              'classes':classes,'seed':0,'optimizer':'Adam(lr=0.001), other PyTorch defaults',
              'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                          'scipy':common.scipy.__version__,'sklearn':common.sklearn.__version__,
                          'torchvision':common.torchvision.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},
              'preprocessing':'See method cache fold manifest; train-only minmax; no clip; float64 fit/float32 input',
              'architecture':{'source':'YDaiLab/PopPhy-CNN master, selected commit 685250753c7f73f8c1d6721efdbf4cb519df813e; pinned raw source unavailable, see README',
                              'conv':[32,3,10],'pool':2,'fc':32,'dropout':0.3,'noise_std':0.01,'l2':0,
                              'matrix_shape':tree['matrix_shape']} if args.method=='popphy_cnn' else
                             {'source':'Wang et al. DOI 10.1002/advs.202404277 equations 2-6, no verified official repository',
                              'pca_dim':32,'hidden':64,'kernel':3,'activation':'ReLU','fusion':'elementwise multiplication',
                              'fixed_adaptation_parameters':['PCA=32','hidden=64','kernel=3','ReLU','single linear output']}}
    if (out/'config.json').exists():
        old = json.loads((out/'config.json').read_text())
        if {k:v for k,v in old.items() if k!='device'} != {k:v for k,v in config.items() if k!='device'}:
            raise ValueError(f'Run configuration mismatch: {out}')
    if (out/'result.json').exists():
        saved = json.loads((out/'result.json').read_text())
        if saved['signature'] != signature: raise ValueError('Completed run signature mismatch')
        print(f'Skip completed {out}',flush=True); return
    write_json(out/'config.json',config)
    common.seed_all()
    if args.method == 'popphy_cnn':
        from networks import PopPhyCNN
        model = PopPhyCNN(tree['matrix_shape'],len(classes))
    else:
        from networks import DeepPhylo
        model = DeepPhylo(np.load(args.cache_dir/'embeddings.npy'),len(classes))
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    write_json(out/'effective_optimizer.json',{k:v for k,v in optimizer.param_groups[0].items() if k!='params'})
    generator = torch.Generator().manual_seed(0)
    loaders = {p:DataLoader(ds,batch_size=32,shuffle=p=='train',num_workers=0,generator=generator if p=='train' else None) for p,ds in datasets.items()}
    positions = torch.as_tensor(np.load(args.cache_dir/'positions.npy'),device=args.device)
    shape = tree['matrix_shape']
    best = float('inf'); stale = 0; start = 1; history = []; elapsed = 0.; best_epoch = 0
    if (out/'last.pt').exists():
        state = torch.load(out/'last.pt',map_location=args.device,weights_only=False)
        if state['signature'] != signature: raise ValueError('Checkpoint signature mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        best,stale,start,history,elapsed,best_epoch = state['best'],state['stale'],state['epoch']+1,state['history'],state['elapsed'],state['best_epoch']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu()); generator.set_state(state['loader_rng'].cpu())
        if torch.cuda.is_available() and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    clock = time.perf_counter()
    for epoch in range(start,args.epochs+1):
        if stale >= 10: break
        model.train(); total = 0.; n = 0
        for x,present,y in loaders['train']:
            x,present,y = x.to(args.device),present.to(args.device),y.to(args.device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(forward(model,x,present,args,positions,shape),y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); optimizer.step(); total += loss.item()*len(y); n += len(y)
        vl,_ = evaluate(model,loaders['val'],args,positions,shape)
        if not np.isfinite(vl): raise RuntimeError('Nonfinite validation loss')
        if vl < best:
            best,stale,best_epoch = vl,0,epoch
            torch.save(model.state_dict(),out/'best.tmp'); (out/'best.tmp').replace(out/'best.pt')
        else: stale += 1
        history.append({'epoch':epoch,'train_loss':total/n,'validation_loss':vl})
        state = dict(signature=signature,model=model.state_dict(),optimizer=optimizer.state_dict(),
                     best=best,stale=stale,epoch=epoch,history=history,elapsed=elapsed+time.perf_counter()-clock,
                     best_epoch=best_epoch,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(),loader_rng=generator.get_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(state,out/'last.tmp'); (out/'last.tmp').replace(out/'last.pt')
        print(f'{args.method} level={args.current_level} fold={args.current_fold} epoch={epoch} val_loss={vl:.6f}',flush=True)
    training_seconds = elapsed+time.perf_counter()-clock
    model.load_state_dict(torch.load(out/'best.pt',map_location=args.device,weights_only=True))
    clock = time.perf_counter()
    _,(y,p) = evaluate(model,loaders['test'],args,positions,shape,True)
    evaluation_seconds = time.perf_counter()-clock
    common.save_predictions(out/'predictions.csv',parts['test'][2],y,p,classes)
    np.savez(out/'predictions.npz',ids=parts['test'][2],y=y,probabilities=p)
    write_json(out/'history.json',history)
    write_json(out/'result.json',{**common.metrics(y,p),'best_epoch':best_epoch,'best_validation_loss':best,
                                 'parameters':sum(p.numel() for p in model.parameters()),
                                 'training_seconds':training_seconds,'evaluation_seconds':evaluation_seconds,
                                 'signature':signature,'config':config})

def aggregate(args,level,signature):
    parent = args.output_dir/('smoke' if args.smoke else 'formal')/f'level_{level}'
    paths = [parent/f'fold_{i}' for i in range(1,6)]
    if not all((p/'result.json').exists() for p in paths): return
    results = [json.loads((p/'result.json').read_text()) for p in paths]
    if any(r['signature']!=signature for r in results): raise ValueError('OOF signature mismatch')
    classes = results[0]['config']['classes']
    if any(r['config']['classes']!=classes for r in results): raise ValueError('OOF class mapping mismatch')
    arrays = [np.load(p/'predictions.npz') for p in paths]
    ids = np.concatenate([p['ids'] for p in arrays]); y = np.concatenate([p['y'] for p in arrays]); prob = np.concatenate([p['probabilities'] for p in arrays])
    if len(np.unique(ids))!=len(ids): raise ValueError('OOF repeated sample ID')
    for i,a in enumerate(arrays,1):
        expected = np.load(args.shared_cache_dir/f'level_{level}'/f'fold_{i}'/'test_ids.npy')
        if not args.smoke and not np.array_equal(a['ids'],expected): raise ValueError('OOF test ID mismatch')
    common.save_predictions(parent/'oof_predictions.csv',ids,y,prob,classes)
    write_json(parent/'summary.json',{'method':NAMES[args.method],'folds':5,'signature':signature,'smoke':args.smoke,
               'metrics':{k:{'mean':float(np.mean([r[k] for r in results])),
                             'std':float(np.std([r[k] for r in results],ddof=1))} for k in ('macro_f1','macro_auroc')},
               'pooled_oof':common.metrics(y,prob),'oof_samples':len(ids)})

def main():
    parser = argparse.ArgumentParser(description='Taxonomy-based classification baselines')
    parser.add_argument('--method', choices=tuple(NAMES), required=True)
    parser.add_argument('--level',type=int,nargs='+',choices=range(1,6),default=list(range(1,6)))
    parser.add_argument('--fold',type=int,nargs='+',choices=range(1,6),default=list(range(1,6)))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=None)
    parser.add_argument('--cache-dir',type=Path,default=None)
    parser.add_argument('--shared-cache-dir',type=Path,default=HERE.parent/'shared_cache')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-samples',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=200)
    args = parser.parse_args(); method = args.method
    args.output_dir = args.output_dir or HERE/method/'results'
    args.cache_dir = args.cache_dir or HERE/method/'cache'
    if args.epochs<1 or args.threads<1 or args.smoke_samples<1: parser.error('positive epochs/threads/smoke-samples required')
    if not args.smoke and args.epochs!=200: parser.error('Formal runs require 200 maximum epochs')
    # Every result/checkpoint remains under this method results, including custom paths.
    for field,root in [('output_dir',HERE/method/'results'),('cache_dir',HERE/method/'cache')]:
        path = getattr(args,field).resolve()
        if not path.is_relative_to(root.resolve()): parser.error(f'--{field.replace("_","-")} must stay inside {root}')
        setattr(args,field,path)
    torch.set_num_threads(args.threads)
    torch.device(args.device)  # Invalid devices fail before preprocessing.
    signature,payload = provenance(common,method)
    import fcntl
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    with (args.cache_dir/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        tree = prepare_tree(method,args.cache_dir,signature,payload)
    write_json(args.output_dir/'data_code_audit.json',{'signature':signature,'provenance':payload,
               'method':NAMES[method],'tree_signature':tree['tree_signature'],'formal_runs_planned':25,
               'denominator':'all 10437 matched genera; fixed columns not reclosed'})
    counts,denom,ids = read_counts()
    incidence = sparse.load_npz(args.cache_dir/'incidence.npz')
    for level in args.level:
        for fold in args.fold:
            args.current_level,args.current_fold = level,fold
            folder,classes,parts = shared_fold(common,args,level,fold,payload,ids)
            with (args.cache_dir/'.prepare.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                minimum,scale,constant = fit_transform(args,parts,counts,denom,incidence,signature)
            if args.prepare_only: continue
            if args.smoke:
                reduced = {}
                for p,(rows,y,sids) in parts.items():
                    first = np.unique(y,return_index=True)[1]
                    ix = np.unique(np.concatenate([first,np.arange(min(args.smoke_samples,len(y)))]))
                    reduced[p] = rows[ix],y[ix],sids[ix]
                parts = reduced
            datasets = {p:FoldDataset(counts,denom,v[0],v[1],minimum,scale,constant,incidence,method) for p,v in parts.items()}
            train(args,parts,classes,datasets,signature,tree)
        if not args.prepare_only: aggregate(args,level,signature)

if __name__ == '__main__': main()
