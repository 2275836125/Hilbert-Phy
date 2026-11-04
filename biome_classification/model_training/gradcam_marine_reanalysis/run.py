"""Frozen Marine Grad-CAM, training-only association and input sensitivity pipeline."""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/marine_matplotlib')
import numpy as np
import pandas as pd
from scipy import sparse, stats
import scipy, torch, torchvision, sklearn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_model, encode
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'data_processing'
FIXED = BASE/'mgnify_matched_counts_genus_ncbi_top4096_min20'
CACHE = ROOT/'model_training/shared_cache/level_3'
MODELS = ROOT/'model_training/results/experiments/level_3/hilbert_resnet18'
LAYERS = ['layer1.1', 'layer2.1']

def sha(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def js(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); tmp.replace(p)

def bh(p):
    p=np.asarray(p); order=np.argsort(p); q=np.empty(len(p)); q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; return np.minimum(q,1)

def associate(x,y):
    """Tie-corrected two-sided asymptotic Mann-Whitney; abundance direction AUC."""
    x=x.tocsc(); n=len(y); n1=int(y.sum()); n0=n-n1; rows=[]
    for j in range(x.shape[1]):
        a,b=x.indptr[j:j+2]; v=x.data[a:b]; iy=x.indices[a:b]; keep=v!=0; v=v[keep]; iy=iy[keep]
        nz=len(v); z=n-nz
        if nz:
            u,inv,cnt=np.unique(v,return_inverse=True,return_counts=True)
            ranks=z+np.cumsum(cnt)-(cnt-1)/2
            r1=float(ranks[inv[y[iy]]].sum())+(n1-int(y[iy].sum()))*(z+1)/2
            tie=float(np.sum(cnt.astype(float)**3-cnt)) + float(z)**3-z
        else: r1=n1*(n+1)/2; tie=float(n)**3-n
        U=r1-n1*(n1+1)/2; auc=U/(n1*n0)
        var=n1*n0/12*((n+1)-tie/(n*(n-1)))
        p=1. if var<=0 else float(2*stats.norm.sf(max(0,abs(U-n1*n0/2)-.5)/np.sqrt(var)))
        marine_values=x[y,j].toarray().ravel(); other_values=x[~y,j].toarray().ravel()
        rows.append((auc,p,nz/n,n1,n0,float((marine_values>0).mean()),float((other_values>0).mean()),float(marine_values.mean()),float(other_values.mean()),bool(var<=0)))
    out=pd.DataFrame(rows,columns=['auroc','p','prevalence','marine_n','nonmarine_n','marine_detection','nonmarine_detection','marine_mean_relative','nonmarine_mean_relative','constant']); out['detection_difference']=out.marine_detection-out.nonmarine_detection; out['mean_relative_difference']=out.marine_mean_relative-out.nonmarine_mean_relative; out['q']=bh(out.p); return out

def taxonomy(features,out):
    tax=BASE/'ncbi_taxonomy'; nodes={}
    with (tax/'nodes.dmp').open() as f:
        for line in f:
            p=[s.strip() for s in line.split('|')]; nodes[int(p[0])]=(int(p[1]),p[2])
    merged={}
    with (tax/'merged.dmp').open() as f:
        for line in f:
            p=line.split('|'); merged[int(p[0])]=int(p[1])
    rec=[]; needed=set()
    for row in features.itertuples():
        original=int(row.tax_id); tid=original; seen=set()
        while tid in merged and tid not in seen: seen.add(tid); tid=merged[tid]
        d={'feature_index':int(row.column_index),'tax_id':original,'resolved_tax_id':tid,'genus':row.scientific_name,'node_rank':nodes.get(tid,(None,'missing'))[1]}
        cur=tid; seen=set()
        while cur in nodes and cur not in seen:
            seen.add(cur); parent,rank=nodes[cur]
            if rank in ('family','order','phylum'):
                d[rank+'_tax_id']=cur; needed.add(cur)
            if parent==cur: break
            cur=parent
        for rank in ('family','order','phylum'): d.setdefault(rank+'_tax_id',None)
        needed.add(tid); rec.append(d)
    names={}
    with (tax/'names.dmp').open() as f:
        for line in f:
            p=[s.strip() for s in line.split('|')]
            if p[3]=='scientific name' and int(p[0]) in needed: names[int(p[0])]=p[1]
    for d in rec:
        d['ncbi_scientific_name']=names.get(d['resolved_tax_id'],'missing')
        d['name_matches']=d['genus']==d['ncbi_scientific_name']
        for rank in ('family','order','phylum'): d[rank]=names.get(d[rank+'_tax_id'],'missing')
    frame=pd.DataFrame(rec); frame.to_csv(out/'taxonomy_mapping.csv',index=False)
    js(out/'taxonomy_audit.json',{'n_input':len(frame),'non_genus_nodes':frame[frame.node_rank!='genus'].tax_id.tolist(),'name_mismatches':frame[~frame.name_matches].tax_id.tolist(),'missing':{r:int((frame[r]=='missing').sum()) for r in ('family','order','phylum')},'source':'local NCBI taxdump; merged IDs resolved; unknown ranks not guessed'})
    return frame

def model(fold):
    m=build_model('hilbert',18,28); m.load_state_dict(torch.load(MODELS/f'fold_{fold}/best.pt',map_location='cpu',weights_only=True)); m.eval()
    return m

def get_cam(m,x,h):
    acts={}; handles=[]
    for name in LAYERS:
        handles.append(m.get_submodule(name).register_forward_hook(lambda mod,inp,out,n=name:acts.__setitem__(n,out)))
    try:
        logits=m(encode(x,'hilbert',h)); grads=torch.autograd.grad(logits[:,14].sum(),[acts[n] for n in LAYERS])
        maps=[]
        for name,g in zip(LAYERS,grads):
            cam=F.relu((acts[name]*g.mean((2,3),keepdim=True)).sum(1,keepdim=True))
            cam=F.interpolate(cam,size=(64,64),mode='bilinear',align_corners=False)[:,0]
            # Per-sample min-max after upsampling. Constant maps become all zero.
            lo=cam.amin((1,2),keepdim=True); span=cam.amax((1,2),keepdim=True)-lo
            cam=torch.where(span>0,(cam-lo)/span.clamp_min(1e-30),torch.zeros_like(cam))
            maps.append(cam.detach().numpy())
        return logits.detach().numpy(),np.stack(maps,1)
    finally:
        for handle in handles: handle.remove()

def fold_cam(fold,out,args,h):
    dest=out/f'fold_{fold}'; dest.mkdir(exist_ok=True)
    if (dest/'cam_complete.json').exists(): return
    cache=CACHE/f'fold_{fold}'; x=np.load(cache/'test_x.npy',mmap_mode='r'); y=np.load(cache/'test_y.npy'); ids=np.load(cache/'test_ids.npy')
    saved=np.load(MODELS/f'fold_{fold}/predictions.npz'); assert np.array_equal(ids,saved['ids']) and np.array_equal(y,saved['y'])
    marine=np.flatnonzero(y==14)
    if args.smoke: marine=marine[:16]
    m=model(fold); sums=np.zeros((2,64,64),dtype=np.float64); input_sum=np.zeros(4096,dtype=np.float64); n=0; chosen=[]; per=[]; maxerr=0.
    for start in range(0,len(marine),args.batch):
        ix=marine[start:start+args.batch]; xb=torch.from_numpy(np.array(x[ix])); logits,maps=get_cam(m,xb,h)
        p=torch.from_numpy(logits).softmax(1).numpy(); maxerr=max(maxerr,float(np.max(np.abs(p-saved['probabilities'][ix]))))
        assert np.array_equal(logits.argmax(1),saved['probabilities'][ix].argmax(1)), 'Prediction disagreement'
        good=logits.argmax(1)==14; chosen.extend(ix[good].tolist()); n+=int(good.sum()); sums+=maps[good].sum(0,dtype=np.float64); input_sum+=xb.numpy()[good].sum(0,dtype=np.float64)
        for k,i in enumerate(ix):
            per.append({'sample_id':ids[i],'test_index':int(i),'fold':fold,'true_index':14,'true_label':'Marine','predicted_index':int(logits[k].argmax()),'predicted_label':json.loads((cache/'classes.json').read_text())[str(int(logits[k].argmax()))],'checkpoint':str(MODELS/f'fold_{fold}/best.pt'),'correct':bool(good[k]),'marine_logit':float(logits[k,14]),**{f'{name}_mean':float(maps[k,j].mean()) for j,name in enumerate(LAYERS)},**{f'{name}_zero':bool(not maps[k,j].any()) for j,name in enumerate(LAYERS)}})
        if start%(args.batch*32)==0: print(f'CAM fold {fold}: {start}/{len(marine)}',flush=True)
    assert n>0
    means=sums/n; np.save(dest/'mean_input_vector.npy',input_sum/n); np.save(dest/'mean_input.npy',(input_sum/n)[h.numpy()].reshape(64,64)); np.save(dest/'mean_cam.npy',means); np.save(dest/'correct_marine_test_indices.npy',chosen); pd.DataFrame(per).to_csv(dest/'marine_sample_stats.csv',index=False)
    rec=[]
    for j,name in enumerate(LAYERS):
        threshold=float(means[j].mean()+means[j].std(ddof=0)); mask=means[j]>threshold
        np.save(dest/f'{name}_mask.npy',mask)
        rec.append({'module':name,'mean':float(means[j].mean()),'sd':float(means[j].std(ddof=0)),'threshold':threshold,'highlight_positions':int(mask.sum())})
    js(dest/'cam_complete.json',{'fold':fold,'marine_test':len(marine),'correct_marine_test':n,'prediction_probability_max_abs_error':maxerr,'partitions':json.loads((cache/'manifest.json').read_text())['sizes'],'accuracy':n/len(marine),'zero_cam_selected':{name:sum(r['correct'] and r[name+'_zero'] for r in per) for name in LAYERS},'native_CAM_shapes':{'layer1.1':[64,16,16],'layer2.1':[128,8,8]},'input_mean_min':float((input_sum/n).min()),'input_mean_max':float((input_sum/n).max()),'cam':rec})

def associations(out,mapping,args):
    groups=[]
    for rank in ('family','order','phylum'):
        for tid,part in mapping.dropna(subset=[rank+'_tax_id']).groupby(rank+'_tax_id',sort=True):
            groups.append({'rank':rank,'tax_id':int(tid),'name':part.iloc[0][rank],'members':part.feature_index.to_numpy()})
    pd.DataFrame([{**{k:v for k,v in g.items() if k!='members'},'member_count':len(g['members']),'member_indices':';'.join(map(str,g['members']))} for g in groups]).to_csv(out/'higher_taxon_membership.csv',index=False)
    unfinished=[f for f in range(1,6) if not (out/f'fold_{f}/genus_association.csv').exists()]
    if not unfinished: return groups
    counts=sparse.load_npz(FIXED/'counts.npz').astype(np.float64)
    source=sparse.load_npz(BASE/'mgnify_matched_counts_genus_ncbi/counts.npz'); denom=np.asarray(source.sum(1)).ravel()[np.load(FIXED/'source_row_indices.npy')]; del source
    relative=counts.multiply(1/denom[:,None]).tocsr(); del counts
    membership=sparse.csc_matrix((np.ones(sum(len(g['members']) for g in groups)),(np.concatenate([g['members'] for g in groups]),np.repeat(np.arange(len(groups)),[len(g['members']) for g in groups]))),shape=(4096,len(groups)))
    higher=(relative@membership).tocsr()
    for fold in unfinished:
        dest=out/f'fold_{fold}'; dest.mkdir(exist_ok=True); cache=CACHE/f'fold_{fold}'
        ix=np.load(cache/'train_row_indices.npy'); y=np.load(cache/'train_y.npy')==14
        if args.smoke:
            ii=np.r_[np.flatnonzero(y)[:64],np.flatnonzero(~y)[:64]]; ix=ix[ii]; y=y[ii]
        print(f'Association fold {fold}: n={len(y)}',flush=True)
        a=associate(relative[ix],y); a.insert(0,'feature_index',np.arange(4096)); a.to_csv(dest/'genus_association.csv',index=False)
        a=associate(higher[ix],y)
        # Independent BH families by rank, pre-specified.
        for rank in ('family','order','phylum'):
            mask=np.array([g['rank']==rank for g in groups]); a.loc[mask,'q']=bh(a.loc[mask,'p'])
        a.insert(0,'group_index',np.arange(len(groups))); a.to_csv(dest/'higher_association.csv',index=False)
    return groups

def select(out,mapping,groups):
    inverse=np.argsort(np.load(BASE/'hilbert_encoding/hilbert.npy')); cams=np.stack([np.load(out/f'fold_{f}/mean_cam.npy') for f in range(1,6)])
    n=np.array([json.loads((out/f'fold_{f}/cam_complete.json').read_text())['correct_marine_test'] for f in range(1,6)])
    pooled=np.average(cams,axis=0,weights=n); np.save(out/'pooled_mean_cam.npy',pooled); mean_input=np.average(np.stack([np.load(out/f'fold_{f}/mean_input.npy') for f in range(1,6)]),axis=0,weights=n); np.save(out/'pooled_mean_input.npy',mean_input)
    table=mapping.copy(); layer_support=[]; threshold_records=[]
    for j,name in enumerate(LAYERS):
        a=pooled[j].mean()+pooled[j].std(ddof=0); scores=pooled[j].ravel()[inverse]; np.save(out/f'{name}_mask.npy',pooled[j]>a); masks=[]
        for f in range(5): masks.append((cams[f,j]>cams[f,j].mean()+cams[f,j].std(ddof=0)).ravel()[inverse])
        count=np.stack(masks).sum(0); table[name+'_rank']=stats.rankdata(-scores,method='min').astype(int); table[name+'_cam']=scores; table[name+'_A']=scores>a; table[name+'_folds_B']=count
        support=(scores>a)&(count>=3); table[name+'_AB']=support; layer_support.append(support)
        threshold_records.append({'module':name,'mean':float(pooled[j].mean()),'sd':float(pooled[j].std(ddof=0)),'threshold':float(a),'highlight_positions':int((scores>a).sum())})
    table['both_layers_AB']=layer_support[0]&layer_support[1]; table['any_layer_AB']=layer_support[0]|layer_support[1]
    af=[]
    for f in range(1,6):
        a=pd.read_csv(out/f'fold_{f}/genus_association.csv'); af.append((a.auroc.to_numpy()>=.6)&(a.q.to_numpy()<.05))
        table[f'fold_{f}_auroc']=a.auroc; table[f'fold_{f}_q']=a.q
        table[f'fold_{f}_marine_direction']=a.auroc>0.5
    table['marine_direction_folds']=sum(table[f'fold_{f}_marine_direction'] for f in range(1,6)); table['association_pass_folds']=np.stack(af).sum(0); table['C']=table.association_pass_folds>=4; table['candidate']=table.any_layer_AB&table.C
    table['cam_rank_score']=np.maximum(*[table[name+'_cam'] for name in LAYERS]); table['support_label']=np.where(table.both_layers_AB,'两个层级均支持','单层级或未获CAM支持')
    table['failure_reason']=[('passed' if r.candidate else ';'.join(([ 'A/B: neither module jointly meets A and B'] if not r.any_layer_AB else [])+(['C: fewer than four association-positive folds'] if not r.C else []))) for r in table.itertuples()]
    table=table.sort_values('cam_rank_score',ascending=False); table.to_csv(out/'genus_full.csv',index=False); table[table.candidate].to_csv(out/'genus_candidates.csv',index=False); table[~table.candidate].to_csv(out/'genus_failures.csv',index=False)
    controls=table[~table.C].head(5); controls.to_csv(out/'genus_controls.csv',index=False)
    # Higher taxa remain descriptive; no arbitrary promotion to primary candidates.
    high=[]; haa=[pd.read_csv(out/f'fold_{f}/higher_association.csv') for f in range(1,6)]
    byindex=table.set_index('feature_index')
    for i,g in enumerate(groups):
        member=byindex.loc[g['members']]; d={k:v for k,v in g.items() if k!='members'}; d['member_count']=len(member)
        for name in LAYERS:
            v=member[name+'_cam'].to_numpy()
            for key,val in zip(['mean','sd','q25','median','q75','max'],[v.mean(),v.std(ddof=0) if len(v)>1 else 0,*np.quantile(v,[.25,.5,.75]),v.max()]): d[name+'_'+key]=float(val)
            d[name+'_high_cam_fraction']=float(member[name+'_A'].mean())
        for f,a in enumerate(haa,1): d[f'fold_{f}_auroc']=a.auroc[i]; d[f'fold_{f}_q']=a.q[i]
        for f,a in enumerate(haa,1):
            for metric in ('marine_n','nonmarine_n','marine_detection','nonmarine_detection','marine_mean_relative','nonmarine_mean_relative','detection_difference','mean_relative_difference','p','constant'): d[f'fold_{f}_{metric}']=a.loc[i,metric]
        d['association_pass_folds']=sum(a.auroc[i]>=.6 and a.q[i]<.05 for a in haa); d['primary_candidate']=False; high.append(d)
    pd.DataFrame(high).to_csv(out/'higher_taxa_full.csv',index=False)
    js(out/'pooled_cam_statistics.json',{'weighting':'sample-weighted correct Marine test samples; fold masks use fold-specific means','n_correct':int(n.sum()),'layers':threshold_records,'genus_candidates':int(table.candidate.sum()),'both_layers_candidates':int((table.candidate&table.both_layers_AB).sum()),'higher_primary_candidates':0})
    return table,controls

def perturb(out,table,controls,args,h):
    selected=pd.concat([table[table.candidate],controls]).drop_duplicates('feature_index')
    if args.smoke: selected=selected.head(2)
    selected.to_csv(out/'perturbation_targets.csv',index=False)
    all_stats=[]
    for fold in range(1,6):
        dest=out/f'fold_{fold}'; statpath=dest/'perturbation_summary.csv'
        if statpath.exists(): all_stats.append(pd.read_csv(statpath)); continue
        cache=CACHE/f'fold_{fold}'; train=np.load(cache/'train_x.npy',mmap_mode='r'); x=np.load(cache/'test_x.npy',mmap_mode='r'); ids=np.load(cache/'test_ids.npy'); ix=np.load(dest/'correct_marine_test_indices.npy')
        if args.smoke: ix=ix[:4]
        indices=selected.feature_index.to_numpy(dtype=int)
        med=np.array([np.median(train[:,j]) for j in indices]); np.savez(dest/'perturbation_training_medians.npz',feature_indices=indices,normalized_training_medians=med)
        del train
        reuse=None
        if args.reuse_perturbations and not args.smoke:
            reuse=np.load(args.reuse_perturbations/f'fold_{fold}/perturbation_distributions.npz')
            assert np.array_equal(reuse['test_indices'],ix) and np.array_equal(reuse['sample_ids'],ids[ix]), 'Perturbation reuse sample mismatch'
        m=model(fold); records=[]; distributions=[]
        with torch.no_grad():
            original=[]
            for start in range(0,len(ix),args.batch): original.extend(m(encode(torch.from_numpy(np.array(x[ix[start:start+args.batch]])),'hilbert',h))[:,14].numpy())
            original=np.asarray(original)
            for j,(_,row) in enumerate(selected.iterrows()):
                oldk=np.flatnonzero(reuse['feature_indices']==int(row.feature_index)) if reuse is not None else []
                if len(oldk):
                    v=np.array(reuse['deltas'][oldk[0]],copy=True)
                else:
                    deltas=[]
                    for start in range(0,len(ix),args.batch):
                        xb=torch.from_numpy(np.array(x[ix[start:start+args.batch]])); xb[:,int(row.feature_index)]=float(med[j]); changed=m(encode(xb,'hilbert',h))[:,14].numpy(); deltas.extend(original[start:start+len(changed)]-changed)
                    v=np.asarray(deltas)
                distributions.append(v)
                records.append({'fold':fold,'feature_index':int(row.feature_index),'tax_id':int(row.tax_id),'genus':row.genus,'role':'candidate' if row.candidate else 'CAM_control','n':len(v),'member_count':1,'mean_delta':float(v.mean()),'sd_delta':float(v.std(ddof=0)) if len(v)>1 else 0.,'q05':float(np.quantile(v,.05)),'median':float(np.median(v)),'q95':float(np.quantile(v,.95)),'positive_sample_fraction':float((v>0).mean())})
                print(f'Perturb fold {fold}: {j+1}/{len(selected)} {row.genus} mean={v.mean():.5f}',flush=True)
        np.savez_compressed(dest/'perturbation_distributions.npz',deltas=np.stack(distributions) if distributions else np.empty((0,len(ix))),feature_indices=indices,sample_ids=ids[ix],test_indices=ix)
        pd.DataFrame(records).to_csv(statpath,index=False); all_stats.append(pd.DataFrame(records))
    joined=pd.concat(all_stats); joined.to_csv(out/'perturbation_by_fold.csv',index=False)
    pooled=[]
    for feature,part in joined.groupby('feature_index'):
        vv=[]
        for fold in range(1,6):
            z=np.load(out/f'fold_{fold}/perturbation_distributions.npz'); k=np.flatnonzero(z['feature_indices']==feature)[0]; vv.append(z['deltas'][k])
        v=np.concatenate(vv); first=part.iloc[0]; pos=int((part.mean_delta>0).sum())
        pooled.append({'feature_index':int(feature),'tax_id':int(first.tax_id),'genus':first.genus,'role':first.role,'n':len(v),'mean_delta':float(v.mean()),'sd_delta':float(v.std(ddof=0)),'q05':float(np.quantile(v,.05)),'median':float(np.median(v)),'q95':float(np.quantile(v,.95)),'positive_folds':pos,'positive_sample_fraction':float((v>0).mean()),'sensitivity_support':bool(v.mean()>0 and pos>=4)})
    pd.DataFrame(pooled).to_csv(out/'perturbation_pooled.csv',index=False)

def plots(out):
    from plot import render
    render(out)
    return

def unused_legacy_plot(out):
    """Original three-panel design; no taxonomy annotations or threshold overlays."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    pooled=np.load(out/'pooled_mean_cam.npy'); mean_input=np.load(out/'pooled_mean_input.npy')
    fig=plt.figure(figsize=(7.2,7.2),layout='constrained'); gs=fig.add_gridspec(2,2)
    top=fig.add_subplot(gs[0,:]); bottom=[fig.add_subplot(gs[1,j]) for j in range(2)]
    im=top.imshow(mean_input,cmap='viridis',vmin=0,vmax=float(mean_input.max()),interpolation='nearest')
    top.set_title('(a) Mean Hilbert-Phy model input'); fig.colorbar(im,ax=top,shrink=.85,label='Mean training-fold-normalized input')
    for j,ax in enumerate(bottom):
        im=ax.imshow(pooled[j],cmap='magma',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(f'({chr(98+j)}) Residual stage {j+1} ({LAYERS[j]})')
        fig.colorbar(im,ax=ax,shrink=.8,label='Mean normalized Grad-CAM')
    for ax in [top,*bottom]: ax.set_xlabel('Hilbert column'); ax.set_ylabel('Hilbert row')
    for ext in ('pdf','svg','png'): fig.savefig(out/f'figure_marine_stages12.{ext}',dpi=400)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--batch',type=int,default=64); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--stage',choices=['all','prepare','cam','association','select','finish'],default='all'); ap.add_argument('--reuse-perturbations',type=Path); args=ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(0); torch.use_deterministic_algorithms(True); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),ROOT/'model_training/models.py',FIXED/'features.csv',FIXED/'counts.npz',FIXED/'biomes.csv',FIXED/'source_row_indices.npy',FIXED/'source_column_indices.npy',BASE/'mgnify_matched_counts_genus_ncbi/counts.npz',BASE/'hilbert_encoding/hilbert.npy',BASE/'ncbi_taxonomy/nodes.dmp',BASE/'ncbi_taxonomy/names.dmp',BASE/'ncbi_taxonomy/merged.dmp']
    for f in range(1,6):
        sources.extend([MODELS/f'fold_{f}/best.pt',MODELS/f'fold_{f}/config.json',MODELS/f'fold_{f}/predictions.npz'])
        sources.extend(CACHE/f'fold_{f}'/n for n in ('manifest.json','normalization.npz','tax_ids.npy','classes.json','train_y.npy','test_y.npy','train_row_indices.npy','test_row_indices.npy','test_ids.npy','train_ids.npy','val_ids.npy','val_y.npy','val_row_indices.npy','train_x.npy','test_x.npy'))
    cfg={'schema':2,'figure_layout':'top: mean normalized model input; bottom: layer1 and layer2 mean Grad-CAM; no taxonomy annotations','input_averaging':'mean of already training-fold-normalized inputs over exactly the CAM samples; no per-sample rescaling or clipping','reuse_perturbations':str(args.reuse_perturbations.resolve()) if args.reuse_perturbations else None,'smoke':args.smoke,'batch':args.batch,'threads':args.threads,'biome_level':3,'marine_label':'Marine','marine_index':14,'model':'hilbert_resnet18','modules':LAYERS,'CAM_samples':'correctly classified Marine outer-test samples','CAM_normalization':'per sample min-max after ReLU and bilinear upsampling to 64x64; zero-span -> zero','CAM_SD_ddof':0,'align_corners':False,'pooled_weight':'number of correctly classified Marine samples','selection':'A strict pooled mean+SD AND B >=3/5 masks in same module; C marine abundance AUROC>=0.60 and BH q<0.05 in >=4/5 train folds','association':'two-sided tie-corrected asymptotic Mann-Whitney U with continuity correction; BH over 4096 genera per fold; higher BH separately per rank','controls':'top 5 max-CAM genera failing C','higher_primary_policy':'descriptive only; no post-hoc primary promotion, no group perturbation required','sensitivity_support':'pooled mean delta >0 and >=4/5 positive fold means; descriptive sensitivity criterion','versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'torchvision':torchvision.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'git_commit':None,'git_note':'code and data versions recorded by SHA256; see preflight audit for Git metadata','sources':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    if args.reuse_perturbations:
        oldcfg=json.loads((args.reuse_perturbations/'config.json').read_text())
        assert not args.smoke and not oldcfg['smoke']
        for key,value in oldcfg['sources'].items():
            if key in cfg['sources']: assert cfg['sources'][key]==value, f'Reuse provenance mismatch: {key}'
        assert oldcfg['batch']==args.batch and oldcfg['threads']==args.threads
        cfg['reused_perturbation_sources']={str(p):sha(p) for f in range(1,6) for p in [args.reuse_perturbations/f'fold_{f}/perturbation_distributions.npz']}
        cfg['reused_perturbation_config_sha256']=sha(args.reuse_perturbations/'config.json')
    cp=out/'config.json'
    if cp.exists():
        if json.loads(cp.read_text())!=cfg: raise RuntimeError('Configuration/source mismatch; choose a new output directory')
    else: js(cp,cfg)
    js(out/'last_invocation.json',{'argv':sys.argv,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    features=pd.read_csv(FIXED/'features.csv'); assert np.array_equal(features.column_index,np.arange(4096))
    for f in range(1,6):
        c=json.loads((MODELS/f'fold_{f}/config.json').read_text()); manifest=json.loads((CACHE/f'fold_{f}/manifest.json').read_text()); assert c['signature']==manifest['signature']; assert not c['smoke']; assert np.array_equal(np.load(CACHE/f'fold_{f}/tax_ids.npy'),features.tax_id); assert json.loads((CACHE/f'fold_{f}/classes.json').read_text())['14']=='Marine'
    mapping=taxonomy(features,out) if not (out/'taxonomy_mapping.csv').exists() else pd.read_csv(out/'taxonomy_mapping.csv')
    h=torch.from_numpy(np.load(BASE/'hilbert_encoding/hilbert.npy')); assert np.array_equal(np.sort(h.numpy()),np.arange(4096))
    if args.stage in ('all','cam'):
        for f in range(1,6): fold_cam(f,out,args,h)
    if args.stage in ('all','association'): associations(out,mapping,args)
    if args.stage in ('all','select','finish'):
        groups=[]
        for r in pd.read_csv(out/'higher_taxon_membership.csv').itertuples(): groups.append({'rank':r.rank,'tax_id':r.tax_id,'name':r.name,'members':np.array(list(map(int,str(r.member_indices).split(';'))))})
        table,controls=select(out,mapping,groups)
        plots(out)
        if args.stage=='select': return
        perturb(out,table,controls,args,h); plots(out)
        pd.DataFrame([json.loads((out/f'fold_{f}/cam_complete.json').read_text())|{'cam':json.dumps(json.loads((out/f'fold_{f}/cam_complete.json').read_text())['cam']),'partitions':json.dumps(json.loads((out/f'fold_{f}/cam_complete.json').read_text())['partitions'])} for f in range(1,6)]).to_csv(out/'fold_sample_cam_statistics.csv',index=False)
        pd.concat([pd.read_csv(out/f'fold_{f}/marine_sample_stats.csv') for f in range(1,6)]).to_csv(out/'marine_test_predictions.csv',index=False)
        sample_list=pd.concat([pd.read_csv(out/f'fold_{f}/marine_sample_stats.csv').query('correct') for f in range(1,6)])
        assert sample_list.sample_id.is_unique
        sample_list.to_csv(out/'plot_samples.csv',index=False)
        js(out/'complete.json',{'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'smoke':args.smoke})

if __name__=='__main__': main()
