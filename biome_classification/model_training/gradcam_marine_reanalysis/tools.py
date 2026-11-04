"""Marine analysis checks, prediction, plotting and reporting subcommands."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'hilbert_phy_matplotlib'))


def preflight():
    """Read-only formal-experiment audit; no preprocessing parameters are fitted."""

    import json, subprocess

    import numpy as np

    import pandas as pd

    from sklearn.model_selection import StratifiedKFold,train_test_split

    import run

    def main():
        out=run.Path(__file__).parent/'checks'; out.mkdir(exist_ok=True)
        labels=pd.read_csv(run.FIXED/'biomes.csv',keep_default_na=False)
        levels={}
        for level in range(1,6):
            c=json.loads((run.CACHE.parent/f'level_{level}/fold_1/classes.json').read_text())
            levels[level]={k:v for k,v in c.items() if 'marine' in v.casefold()}
        assert levels[1]=={} and levels[2]=={} and levels[3]=={'14':'Marine','15':'Non-marine Saline and Alkaline'}
        values=labels['3'].to_numpy(); valid=~np.isin(np.char.lower(np.char.strip(values.astype(str))),['','na','none']); names,counts=np.unique(values[valid],return_counts=True); classes=names[counts>=100]; keep=np.flatnonzero(valid & np.isin(values,classes)); y=np.searchsorted(classes,values[keep]); folds=list(StratifiedKFold(5,shuffle=True,random_state=0).split(keep,y))
        features=pd.read_csv(run.FIXED/'features.csv'); records=[]
        for f,(pool,test) in enumerate(folds,1):
            cache=run.CACHE/f'fold_{f}'; train,val=train_test_split(pool,test_size=.2,stratify=y[pool],random_state=0)
            manifest=json.loads((cache/'manifest.json').read_text()); cfg=json.loads((run.MODELS/f'fold_{f}/config.json').read_text())
            assert cfg['signature']==manifest['signature'] and cfg['encoding']=='hilbert' and cfg['depth']==18 and cfg['level']==3 and not cfg['smoke']
            assert json.loads((cache/'classes.json').read_text())==dict(enumerate(classes)) or json.loads((cache/'classes.json').read_text())=={str(i):v for i,v in enumerate(classes)}
            norm=np.load(cache/'normalization.npz'); assert np.array_equal(norm['tax_ids'],features.tax_id) and np.array_equal(np.load(cache/'tax_ids.npy'),features.tax_id)
            for part,ix in [('train',train),('val',val),('test',test)]:
                assert np.array_equal(np.load(cache/f'{part}_row_indices.npy'),keep[ix]); assert np.array_equal(np.load(cache/f'{part}_ids.npy'),labels.sample_id.to_numpy()[keep[ix]]); assert np.array_equal(np.load(cache/f'{part}_y.npy'),y[ix])
            for path,digest in manifest['provenance']['files'].items(): assert run.sha(run.ROOT/path)==digest
            records.append({'fold':f,'checkpoint':str(run.MODELS/f'fold_{f}/best.pt'),'checkpoint_sha256':run.sha(run.MODELS/f'fold_{f}/best.pt'),'signature':cfg['signature'],'sizes':manifest['sizes'],'normalization_sha256':run.sha(cache/'normalization.npz'),'split_reproduced':True})
        h=np.load(run.BASE/'hilbert_encoding/hilbert.npy'); assert np.array_equal(np.sort(h),np.arange(4096)); inverse=np.argsort(h)
        # Single nonzero feature passed through actual model encoder, for every input column.
        import torch
        for start in range(0,4096,64):
            indices=np.arange(start,min(start+64,4096)); v=torch.zeros(len(indices),4096); v[np.arange(len(indices)),indices]=1
            encoded=run.encode(v,'hilbert',torch.from_numpy(h)).numpy().reshape(len(indices),4096); assert np.array_equal(encoded.argmax(1),inverse[indices]); assert np.all(encoded.sum(1)==1)
        mapping=features.copy(); mapping['hilbert_flat_position']=inverse; mapping['hilbert_row']=inverse//64; mapping['hilbert_column']=inverse%64; mapping.to_csv(out/'verified_position_to_genus.csv',index=False)
        git=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True)
        run.js(out/'preflight.json',{'passed':True,'marine_like_labels_by_level':levels,'selected_level':3,'selected_label':'Marine','excluded_distinct_label':'Non-marine Saline and Alkaline','folds':records,'hilbert_permutation':True,'all_4096_one_hot_mapping_checks':True,'git_commit':git.stdout.strip() or None,'git_status_error':git.stderr.strip(),'reviewed_code':['common.py','models.py','train.py'],'global_feature_preselection_limitation':True})
        print('Preflight passed',flush=True)

    main()


def check():
    """Independent stage-boundary and mean-input checks before the full rerun."""

    import argparse

    import numpy as np

    import torch

    import run

    def main():
        p=argparse.ArgumentParser(); p.add_argument('--output',type=run.Path,required=True); out=p.parse_args().output
        assert (out/'complete.json').exists(); torch.set_num_threads(4); m=run.model(1)
        x=np.load(run.CACHE/'fold_1/test_x.npy',mmap_mode='r'); ix=np.load(out/'fold_1/correct_marine_test_indices.npy'); h=np.load(run.BASE/'hilbert_encoding/hilbert.npy'); activations={}; handles=[]
        for name in ('layer1','layer1.1','layer2','layer2.1'):
            handles.append(m.get_submodule(name).register_forward_hook(lambda mod,inp,value,key=name:activations.__setitem__(key,value.detach().clone())))
        with torch.no_grad(): m(run.encode(torch.from_numpy(np.array(x[ix[:2]])),'hilbert',torch.from_numpy(h)))
        for handle in handles: handle.remove()
        assert tuple(activations['layer1'].shape[1:])==(64,16,16)
        assert tuple(activations['layer2'].shape[1:])==(128,8,8)
        assert torch.equal(activations['layer1'],activations['layer1.1']) and torch.equal(activations['layer2'],activations['layer2.1'])
        counts=[]; means=[]
        for fold in range(1,6):
            dest=out/f'fold_{fold}'; ids=np.load(dest/'correct_marine_test_indices.npy'); xx=np.load(run.CACHE/f'fold_{fold}/test_x.npy',mmap_mode='r'); expected=np.asarray(xx[ids],dtype=np.float64).mean(0)
            assert np.allclose(np.load(dest/'mean_input_vector.npy'),expected,rtol=1e-12,atol=1e-12)
            assert np.array_equal(np.load(dest/'mean_input.npy'),np.load(dest/'mean_input_vector.npy')[h].reshape(64,64))
            counts.append(len(ids)); means.append(expected[h].reshape(64,64))
        assert np.allclose(np.load(out/'pooled_mean_input.npy'),np.average(means,axis=0,weights=counts),rtol=1e-12,atol=1e-12)
        run.js(out/'smoke_audit.json',{'passed':True,'modules':['layer1','layer2'],'native_shapes':{'layer1':[64,16,16],'layer2':[128,8,8]},'complete_stage_matches_last_block':True,'mean_input_reconstructed':True,'pooled_input_sample_weighted':True,'check_code_sha256':run.sha(run.Path(__file__))})
        print('Stage boundaries and input averaging checks passed.')

    main()


def audit():
    """Check full-run completeness, selection, perturbation coverage and saved distributions."""

    import argparse, json

    import numpy as np

    import pandas as pd

    import torch

    import run

    def main():
        p=argparse.ArgumentParser(); p.add_argument('--output',type=run.Path,required=True); args=p.parse_args(); out=args.output
        cfg=json.loads((out/'config.json').read_text()); assert not cfg['smoke']; assert cfg['modules']==['layer1.1','layer2.1']
        g=pd.read_csv(out/'genus_full.csv').sort_values('feature_index'); assert len(g)==4096
        expected_c=np.stack([(g[f'fold_{f}_auroc']>=.60)&(g[f'fold_{f}_q']<.05) for f in range(1,6)]).sum(0)>=4
        assert np.array_equal(expected_c,g.C)
        pooled=np.load(out/'pooled_mean_cam.npy'); inv=np.argsort(np.load(run.BASE/'hilbert_encoding/hilbert.npy')); ab=[]
        for j,name in enumerate(run.LAYERS):
            v=pooled[j]; A=v.ravel()[inv]>v.mean()+v.std(ddof=0); count=np.stack([np.load(out/f'fold_{f}/{name}_mask.npy').ravel()[inv] for f in range(1,6)]).sum(0)
            assert np.array_equal(A,g[name+'_A']); assert np.array_equal(count,g[name+'_folds_B']); ab.append(A&(count>=3))
        assert np.array_equal((ab[0]|ab[1])&expected_c,g.candidate); assert np.array_equal(ab[0]&ab[1],g.both_layers_AB)
        targets=pd.read_csv(out/'perturbation_targets.csv'); assert set(g.loc[g.candidate,'feature_index']).issubset(set(targets.feature_index)); controls=pd.read_csv(out/'genus_controls.csv'); assert len(controls)==5 and not controls.C.any()
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        test_ids=[]; n_marine=0; n_correct=0; zeros={}; intervention_checks=[]; input_sum=np.zeros(4096,dtype=np.float64); input_checks=[]
        per=pd.read_csv(out/'perturbation_by_fold.csv'); pooled_per=pd.read_csv(out/'perturbation_pooled.csv')
        for f in range(1,6):
            cache=run.CACHE/f'fold_{f}'; ids=np.load(cache/'test_ids.npy'); test_ids.extend(ids.tolist()); y=np.load(cache/'test_y.npy'); ix=np.load(out/f'fold_{f}/correct_marine_test_indices.npy'); info=json.loads((out/f'fold_{f}/cam_complete.json').read_text()); assert int((y==14).sum())==info['marine_test']; assert np.all(y[ix]==14); assert len(ix)==len(np.unique(ix))==info['correct_marine_test']; n_marine+=info['marine_test']; n_correct+=len(ix)
            z=np.load(out/f'fold_{f}/perturbation_distributions.npz'); assert np.array_equal(z['sample_ids'],ids[ix]); assert set(z['feature_indices'])==set(targets.feature_index); assert z['deltas'].shape==(len(targets),len(ix)); assert np.isfinite(z['deltas']).all()
            for k,feature in enumerate(z['feature_indices']):
                ref=per[(per.fold==f)&(per.feature_index==feature)].iloc[0]; assert np.isclose(z['deltas'][k].mean(),ref.mean_delta,atol=1e-8)
            # Reconstruct input panel directly from cached test inputs, independently of CAM accumulation.
            cached=np.load(cache/'test_x.npy',mmap_mode='r'); fold_sum=np.zeros(4096,dtype=np.float64)
            for offset in range(0,len(ix),257): fold_sum+=np.asarray(cached[ix[offset:offset+257]],dtype=np.float64).sum(0)
            direct=fold_sum/len(ix); saved_input=np.load(out/f'fold_{f}/mean_input_vector.npy')
            assert np.allclose(direct,saved_input,rtol=1e-12,atol=1e-12)
            assert np.allclose(np.load(out/f'fold_{f}/mean_input.npy'),direct[np.argsort(inv)].reshape(64,64),rtol=1e-12,atol=1e-12)
            input_sum+=fold_sum; input_checks.append({'fold':f,'mean_input_max_abs_error':float(np.max(np.abs(direct-saved_input)))})
            del cached
            # Independently replay two saved interventions on the first full-size batch.
            x=np.load(cache/'test_x.npy',mmap_mode='r'); train=np.load(cache/'train_x.npy',mmap_mode='r'); med=np.load(out/f'fold_{f}/perturbation_training_medians.npz'); m=run.model(f); h=torch.from_numpy(np.load(run.BASE/'hilbert_encoding/hilbert.npy')); original=torch.from_numpy(np.array(x[ix[:64]]))
            with torch.no_grad():
                base=m(run.encode(original,'hilbert',h))[:,14].numpy()
                for k in (0,len(targets)-1):
                    feature=int(z['feature_indices'][k]); median=float(np.median(train[:,feature])); assert median==float(med['normalized_training_medians'][k]); changed=original.clone(); changed[:,feature]=median
                    other=np.arange(4096)!=feature; assert torch.equal(changed[:,other],original[:,other]); replay=base-m(run.encode(changed,'hilbert',h))[:,14].numpy(); assert np.allclose(replay,z['deltas'][k,:len(replay)],rtol=1e-6,atol=1e-6)
                    intervention_checks.append({'fold':f,'feature_index':feature,'training_median_verified':True,'only_target_changed':True,'max_replay_error':float(np.max(np.abs(replay-z['deltas'][k,:len(replay)])))})
            del train,x,m
            stats=pd.read_csv(out/f'fold_{f}/marine_sample_stats.csv'); assert len(stats)==info['marine_test']; zeros[str(f)]={name:int(stats[name+'_zero'].sum()) for name in run.LAYERS}
        assert len(test_ids)==len(set(test_ids))==199150; assert n_marine==40426; assert np.all(pooled_per.n==n_correct)
        assert np.allclose(np.load(out/'pooled_mean_input.npy'),(input_sum/n_correct)[np.argsort(inv)].reshape(64,64),rtol=1e-12,atol=1e-12)
        run.js(out/'final_audit.json',{'passed':True,'modules':run.LAYERS,'input_panel_independently_reconstructed':True,'input_checks':input_checks,'figure_has_no_taxon_annotations':True,'genus_rows':len(g),'test_samples_unique':len(test_ids),'true_marine_test':n_marine,'correct_marine_test':n_correct,'selection_recomputed':True,'all_candidates_perturbed':True,'n_perturbation_targets':len(targets),'n_controls':len(controls),'perturbation_distributions_complete':True,'zero_cam_maps_by_fold':zeros,'independent_intervention_replays':intervention_checks,'audit_script_sha256':run.sha(run.Path(__file__))})
        print('Full-result audit passed.',flush=True)

    main()


def predict():
    """Recompute every outer-test prediction; original files remain read-only."""

    import argparse,json

    import numpy as np

    import pandas as pd

    import torch

    import run

    p=argparse.ArgumentParser(); p.add_argument('--output',type=run.Path,required=True); args=p.parse_args(); out=args.output

    p=argparse.ArgumentParser(); p.add_argument('--output',type=run.Path,required=True); args=p.parse_args(); out=args.output

    p=argparse.ArgumentParser(); p.add_argument('--output',type=run.Path,required=True); args=p.parse_args(); out=args.output

    p=argparse.ArgumentParser(); p.add_argument('--output',type=run.Path,required=True); args=p.parse_args(); out=args.output

    assert (out/'config.json').exists()

    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)

    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)

    h=torch.from_numpy(np.load(run.BASE/'hilbert_encoding/hilbert.npy')); all_records=[]; summaries=[]

    h=torch.from_numpy(np.load(run.BASE/'hilbert_encoding/hilbert.npy')); all_records=[]; summaries=[]

    h=torch.from_numpy(np.load(run.BASE/'hilbert_encoding/hilbert.npy')); all_records=[]; summaries=[]

    for f in range(1,6):
        dest=out/f'fold_{f}'; dest.mkdir(exist_ok=True); target=dest/'recomputed_test_predictions.csv'
        if target.exists():
            all_records.append(pd.read_csv(target)); continue
        cache=run.CACHE/f'fold_{f}'; ids=np.load(cache/'test_ids.npy'); y=np.load(cache/'test_y.npy'); x=np.load(cache/'test_x.npy',mmap_mode='r'); saved=np.load(run.MODELS/f'fold_{f}/predictions.npz'); m=run.model(f); probs=[]
        with torch.no_grad():
            for start in range(0,len(ids),64):
                probs.append(m(run.encode(torch.from_numpy(np.array(x[start:start+64])),'hilbert',h)).softmax(1).numpy())
        probs=np.concatenate(probs); pred=probs.argmax(1); assert np.array_equal(ids,saved['ids']) and np.array_equal(y,saved['y'])
        oldpred=saved['probabilities'].argmax(1); different=pred!=oldpred
        differences=pd.DataFrame({'sample_id':ids[different],'true_index':y[different],'recomputed_index':pred[different],'saved_index':oldpred[different],'new_top_probability':probs.max(1)[different],'saved_top_probability':saved['probabilities'].max(1)[different],'max_abs_probability_error':np.abs(probs-saved['probabilities']).max(1)[different]})
        differences.to_csv(dest/'prediction_differences.csv',index=False)
        rec=pd.DataFrame({'sample_id':ids,'fold':f,'true_index':y,'predicted_index':pred,'correct':pred==y,'checkpoint':str(run.MODELS/f'fold_{f}/best.pt')})
        for j in range(probs.shape[1]): rec[f'probability_{j}']=probs[:,j]
        rec.to_csv(target,index=False); all_records.append(rec)
        run.js(dest/'prediction_recheck.json',{'fold':f,'n':len(ids),'all_saved_classes_match':bool(not different.any()),'class_disagreements':int(different.sum()),'marine_true_disagreements':int((different&(y==14)).sum()),'max_probability_error':float(np.abs(probs-saved['probabilities']).max()),'checkpoint_sha256':run.sha(run.MODELS/f'fold_{f}/best.pt')})
        print(f'Full prediction fold {f} passed: {len(ids)}',flush=True)

    joined=pd.concat(all_records); assert joined.sample_id.is_unique; joined.to_csv(out/'recomputed_oof_predictions.csv',index=False)

    joined=pd.concat(all_records); assert joined.sample_id.is_unique; joined.to_csv(out/'recomputed_oof_predictions.csv',index=False)

    joined=pd.concat(all_records); assert joined.sample_id.is_unique; joined.to_csv(out/'recomputed_oof_predictions.csv',index=False)

    run.js(out/'prediction_audit.json',{'all_test_samples':len(joined),'unique_ids':True,'code_sha256':run.sha(run.Path(__file__)),'analysis_config_sha256':run.sha(out/'config.json')})


def plot():
    """Publication figure with strict population-SD threshold contours."""

    import argparse

    import numpy as np

    import matplotlib

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    import run

    def render(out):
        maps=np.load(out/'pooled_mean_cam.npy'); image=np.load(out/'pooled_mean_input.npy')
        dest=out/'publication'; dest.mkdir(exist_ok=True)
        cfg={'source_config':run.sha(out/'config.json'),'plot_code':run.sha(run.Path(__file__)),'shared_cam_range':[0,float(maps.max())],'dpi':400,'width_inches':7.2,'threshold':'strict > spatial mean + population SD (ddof=0)','taxon_annotations':False}
        if (dest/'plot_config.json').exists(): assert __import__('json').loads((dest/'plot_config.json').read_text())==cfg
        run.js(dest/'plot_config.json',cfg)
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
        fig=plt.figure(figsize=(7.2,7.4)); top=fig.add_axes([.30,.57,.36,.35]); lower=[fig.add_axes([.07,.10,.36,.35]),fig.add_axes([.55,.10,.36,.35])]
        im=top.imshow(image,cmap='viridis',vmin=0,vmax=float(image.max()),interpolation='nearest'); top.set_title('(a) Mean Hilbert-Phy input')
        fig.colorbar(im,cax=fig.add_axes([.70,.57,.016,.35]),label='Mean normalized model input')
        for j,ax in enumerate(lower):
            im=ax.imshow(maps[j],cmap='magma',vmin=0,vmax=float(maps.max()),interpolation='nearest')
            mask=maps[j]>maps[j].mean()+maps[j].std(ddof=0)
            ax.contour(mask.astype(float),levels=[.5],colors=['cyan'],linewidths=.7)
            ax.set_title(f'({chr(98+j)}) Residual stage {j+1} (layer{j+1}[-1])')
        fig.colorbar(im,cax=fig.add_axes([.94,.10,.016,.35]),label='Mean normalized Grad-CAM')
        for ax in [top,*lower]:
            ax.set_xlabel('Hilbert column'); ax.set_ylabel('Hilbert row'); ax.set_xticks([0,16,32,48,63]); ax.set_yticks([0,16,32,48,63])
        fig.text(.5,.025,'Cyan contours: mean CAM > spatial mean + 1 SD',ha='center',fontsize=9)
        for ext in ['pdf','png','svg']: fig.savefig(dest/f'marine_input_and_stages12.{ext}',dpi=400,bbox_inches='tight')
        fig.savefig(dest/'layout_preview.png',dpi=140,bbox_inches='tight'); plt.close(fig)
        n=__import__('json').loads((out/'pooled_cam_statistics.json').read_text())['n_correct']
        (dest/'FIGURE_CAPTION_EN.md').write_text(f'''Mean Hilbert-Phy input and Marine-targeted Grad-CAM at the first and second residual stages of ResNet18. (a) Mean actual model input for {n:,} correctly classified Marine test samples from five disjoint outer folds (biome level 3, Marine class index 14). Relative abundances were transformed using feature-wise min–max parameters fitted on the corresponding training subset; these values are not raw counts. (b,c) Mean Grad-CAM targeting the Marine logit at layer1[-1] (layer1.1; 64 channels, 16 × 16) and layer2[-1] (layer2.1; 128 channels, 8 × 8), respectively, using exactly the same samples as panel a. Channel weights were spatially averaged gradients; weighted activations were summed, rectified, and bilinearly upsampled to 64 × 64 with align_corners=False. Each sample CAM was min–max normalized to [0,1], with constant maps set to zero, before averaging. Pooled means give each sample equal weight. CAM panels share a color scale; cyan contours enclose positions strictly exceeding each pooled map's spatial mean plus one population standard deviation (ddof=0). Taxon screening and training-only abundance associations are reported separately. Upsampled CAM locations suggest candidates rather than precise genus contributions or ecological causation.\n''')

    p=argparse.ArgumentParser(); p.add_argument('--output',type=run.Path,required=True); render(p.parse_args().output)


def report():
    """Generate final descriptive report from completed, audited results."""

    import argparse,json,importlib.metadata

    import numpy as np

    import pandas as pd

    import run

    p=argparse.ArgumentParser();p.add_argument('--output',type=run.Path,required=True);out=p.parse_args().output

    p=argparse.ArgumentParser();p.add_argument('--output',type=run.Path,required=True);out=p.parse_args().output

    p=argparse.ArgumentParser();p.add_argument('--output',type=run.Path,required=True);out=p.parse_args().output

    assert json.loads((out/'final_audit.json').read_text())['passed']

    fold=pd.read_csv(out/'fold_sample_cam_statistics.csv'); cam=json.loads((out/'pooled_cam_statistics.json').read_text()); genus=pd.read_csv(out/'genus_candidates.csv'); per=pd.read_csv(out/'perturbation_pooled.csv'); high=pd.read_csv(out/'higher_taxa_full.csv')

    fold=pd.read_csv(out/'fold_sample_cam_statistics.csv'); cam=json.loads((out/'pooled_cam_statistics.json').read_text()); genus=pd.read_csv(out/'genus_candidates.csv'); per=pd.read_csv(out/'perturbation_pooled.csv'); high=pd.read_csv(out/'higher_taxa_full.csv')

    fold=pd.read_csv(out/'fold_sample_cam_statistics.csv'); cam=json.loads((out/'pooled_cam_statistics.json').read_text()); genus=pd.read_csv(out/'genus_candidates.csv'); per=pd.read_csv(out/'perturbation_pooled.csv'); high=pd.read_csv(out/'higher_taxa_full.csv')

    fold=pd.read_csv(out/'fold_sample_cam_statistics.csv'); cam=json.loads((out/'pooled_cam_statistics.json').read_text()); genus=pd.read_csv(out/'genus_candidates.csv'); per=pd.read_csv(out/'perturbation_pooled.csv'); high=pd.read_csv(out/'higher_taxa_full.csv')

    fold=pd.read_csv(out/'fold_sample_cam_statistics.csv'); cam=json.loads((out/'pooled_cam_statistics.json').read_text()); genus=pd.read_csv(out/'genus_candidates.csv'); per=pd.read_csv(out/'perturbation_pooled.csv'); high=pd.read_csv(out/'higher_taxa_full.csv')

    full=pd.read_csv(out/'genus_full.csv')

    h=np.load(run.BASE/'hilbert_encoding/hilbert.npy'); inv=np.argsort(h)

    h=np.load(run.BASE/'hilbert_encoding/hilbert.npy'); inv=np.argsort(h)

    full['hilbert_position']=inv[full.feature_index]; full['hilbert_row']=full.hilbert_position//64; full['hilbert_column']=full.hilbert_position%64

    full['hilbert_position']=inv[full.feature_index]; full['hilbert_row']=full.hilbert_position//64; full['hilbert_column']=full.hilbert_position%64

    full['hilbert_position']=inv[full.feature_index]; full['hilbert_row']=full.hilbert_position//64; full['hilbert_column']=full.hilbert_position%64

    for f in range(1,6):
        a=pd.read_csv(out/f'fold_{f}/genus_association.csv').set_index('feature_index')
        for metric in ['marine_n','nonmarine_n','marine_detection','nonmarine_detection','marine_mean_relative','nonmarine_mean_relative','detection_difference','mean_relative_difference','p','constant']:
            full[f'fold_{f}_{metric}']=full.feature_index.map(a[metric])

    reasons=[]

    for _,r in full.iterrows():
        parts=[]
        for layer in run.LAYERS:
            if not r[layer+'_A']: parts.append(layer+': fails A')
            if r[layer+'_folds_B']<3: parts.append(layer+': fails B')
        if not r.C: parts.append('fails C: association-positive folds <4')
        reasons.append('passed' if r.candidate else '; '.join(parts))

    full['detailed_failure_reason']=reasons

    full.to_csv(out/'genus_complete_with_effects.csv',index=False)

    high['marine_direction_folds']=sum(high[f'fold_{f}_auroc']>.5 for f in range(1,6)); high.to_csv(out/'higher_taxa_complete.csv',index=False)

    high['marine_direction_folds']=sum(high[f'fold_{f}_auroc']>.5 for f in range(1,6)); high.to_csv(out/'higher_taxa_complete.csv',index=False)

    merged=genus.merge(per[['feature_index','mean_delta','positive_folds','sensitivity_support']],on='feature_index',how='left');merged.to_csv(out/'main_candidates_with_sensitivity.csv',index=False)

    merged=genus.merge(per[['feature_index','mean_delta','positive_folds','sensitivity_support']],on='feature_index',how='left');merged.to_csv(out/'main_candidates_with_sensitivity.csv',index=False)

    evidence=pd.DataFrame(columns=['genus','data_claim','external_status','external_claim','source_url','scope_limit','checked_date'])

    for name in genus.genus:
        if name not in set(evidence.genus):
            evidence.loc[len(evidence)]={'genus':name,'data_claim':'在本数据中与 Marine 标签相关（预设训练折标准）','external_status':'本次运行未进行外部文献核验','external_claim':'未提供外部生态结论','source_url':'','scope_limit':'未穷尽文献，不能解释为无任何相关研究','checked_date':''}

    evidence['is_current_primary_candidate']=evidence.genus.isin(genus.genus)

    evidence.to_csv(out/'external_evidence.csv',index=False)

    lines=['# Marine Hilbert-Phy 重分析实际结果','', '实际 biome 层级：3，Marine（索引14，28类别）。level 1/2 无 Marine；Non-marine Saline and Alkaline 保留为独立类别。','', '正式模型：torchvision ResNet18，Hilbert-Phy 编码，五折各自 best.pt；路径和 SHA256 见 config.json 与 checks/preflight.json。目标模块 layer1[-1] = layer1.1（64×16×16）、layer2[-1] = layer2.1（128×8×8）。','', '|折|Marine 测试数|正确分类数|Marine 正确率|','|---|---:|---:|---:|']

    for r in fold.itertuples(): lines.append(f'|{r.fold}|{r.marine_test}|{r.correct_marine_test}|{r.correct_marine_test/r.marine_test:.4%}|')

    n=int(fold.marine_test.sum()); correct=int(fold.correct_marine_test.sum()); lines += [f'|合计|{n}|{correct}|{correct/n:.4%}|','', '三个面板完全共享上述正确分类 Marine 测试样本；五折测试互斥，样本不重复。平均输入为相对丰度经所属折训练参数 min–max 转换后的实际模型输入，不是原始 count。','', '|模块|空间均值|总体 SD (ddof=0)|严格大于阈值|高亮位置数|','|---|---:|---:|---:|---:|']

    n=int(fold.marine_test.sum()); correct=int(fold.correct_marine_test.sum()); lines += [f'|合计|{n}|{correct}|{correct/n:.4%}|','', '三个面板完全共享上述正确分类 Marine 测试样本；五折测试互斥，样本不重复。平均输入为相对丰度经所属折训练参数 min–max 转换后的实际模型输入，不是原始 count。','', '|模块|空间均值|总体 SD (ddof=0)|严格大于阈值|高亮位置数|','|---|---:|---:|---:|---:|']

    n=int(fold.marine_test.sum()); correct=int(fold.correct_marine_test.sum()); lines += [f'|合计|{n}|{correct}|{correct/n:.4%}|','', '三个面板完全共享上述正确分类 Marine 测试样本；五折测试互斥，样本不重复。平均输入为相对丰度经所属折训练参数 min–max 转换后的实际模型输入，不是原始 count。','', '|模块|空间均值|总体 SD (ddof=0)|严格大于阈值|高亮位置数|','|---|---:|---:|---:|---:|']

    for r in cam['layers']: lines.append(f"|{r['module']}|{r['mean']:.10f}|{r['sd']:.10f}|{r['threshold']:.10f}|{r['highlight_positions']}|")

    lines += ['',f'预设 A/B/C 通过 genus：{len(genus)}；两个层级均支持：{int(genus.both_layers_AB.sum())}。完整名单及扰动见 main_candidates_with_sensitivity.csv；未通过原因见 genus_failures.csv。','', '|Genus|平均 ΔMarine logit|正向折数|敏感性支持|','|---|---:|---:|---|']

    for r in merged.itertuples():lines.append(f'|{r.genus}|{r.mean_delta:.6g}|{r.positive_folds}/5|{r.sensitivity_support}|')

    lines += ['',f'通过预设筛选的 genus 中，描述性扰动支持 {int(merged.sensitivity_support.sum())}/{len(merged)}。无支持者仍保留候选资格与真实结果，不能称为模型依赖的关键属。','',f'高阶元 {len(high)} 个，仅描述性汇总（成员分布和高 CAM 比例，而非 CAM 总和）；无主要高阶元候选，因此没有主要高阶元组扰动。汇总仅覆盖当前输入中的 genus。缺失映射明示 missing。','', '训练关联使用每折实际训练子集全部合格样本；检出率、平均相对丰度、差值、AUROC、双侧 MWU p 和阶元内 BH q 见各折 genus_association.csv/higher_association.csv。五个训练子集重叠，未合并 p；方向一致性仅作描述。常数特征 AUC=.5、p=1；极小 p/q 可能数值下溢为0。','', '空间 CAM 高值指上采样 CAM 高，并非原始梯度绝对值大。前两个 stage 分辨率有限，单属位置仅提出候选；输入中位数替换是模型敏感性核查，非生物因果实验。固定 4096 属及 min20 样本在全数据预筛选，继承正式实验限制。','', '外部证据与本数据关联分列于 external_evidence.csv；个别物种或菌株的海洋分离记录不能外推至整个属或本数据测到的具体物种。未找到可核查来源者明确标记。','', '论文图：publication/marine_input_and_stages12.pdf、400 dpi PNG；英文图注：publication/FIGURE_CAPTION_EN.md。数值数组和阈值掩码均保存；缩版预览用于视觉检查。代码/数据/缓存/检查点 SHA256 与实际依赖见 config.json、requirements_exact.txt。无可读取 Git 提交，使用代码 SHA256 记录版本。']

    lines += ['', '入选样本中全零 CAM 的数量（安全置零并纳入平均，不剔除样本）：', '', '|折|layer1.1 全零|layer2.1 全零|','|---|---:|---:|']

    for f in range(1,6):
        z=json.loads((out/f'fold_{f}/cam_complete.json').read_text())['zero_cam_selected']; lines.append(f"|{f}|{z['layer1.1']}|{z['layer2.1']}|")

    lines += ['', '尤其部分折第二 stage 的全零比例较高；这是 Marine-logit 的 ReLU Grad-CAM 真实结果，不代表样本未被模型正确识别，也不将这些零图剔除或重新加权。']

    prediction_checks=[json.loads((out/f'fold_{f}/prediction_recheck.json').read_text()) for f in range(1,6)]

    lines += ['', '全独立测试预测核对（本次 CPU 重算；原训练设备见各折config.json）：', '', '|折|测试样本数|与旧预测类别不同|真实 Marine 不同|最大概率绝对差|','|---|---:|---:|---:|---:|']

    for r in prediction_checks: lines.append(f"|{r['fold']}|{r['n']}|{r['class_disagreements']}|{r['marine_true_disagreements']}|{r['max_probability_error']:.8g}|")

    lines += ['', '差异样本完整保存在各折 prediction_differences.csv。所有面板和扰动均依据本次对应检查点重新计算的预测，未修改旧预测。CPU/GPU 与不同批大小可能导致浮点差异，接近决策边界的类别变化如实保留；未将差异强行改为旧结果。']

    run.js(out/'runtime_and_training_provenance.json',{'reanalysis_device':'cpu','training_configs':{str(f):json.loads((run.MODELS/f'fold_{f}/config.json').read_text()) for f in range(1,6)},'prediction_checks':prediction_checks,'script_hashes':{p.name:run.sha(p) for p in run.Path(__file__).parent.glob('*.py')}})

    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

    (out/'requirements_exact.txt').write_text('\n'.join(f'{n}=={importlib.metadata.version(n)}' for n in ['numpy','pandas','scipy','scikit-learn','torch','torchvision','matplotlib'])+'\n')


def verify():
    """Independent numerical and cached-input checks; run before the formal analysis."""

    import json

    from pathlib import Path

    import numpy as np

    from scipy import sparse, stats

    import run

    def main():
        rng=np.random.default_rng(20261026); a=rng.choice([0.,0.,0.,.1,.2,.3],size=(151,9)); y=np.arange(151)%3==0
        ours=run.associate(sparse.csr_matrix(a),y)
        p=[]
        for j in range(9):
            ref=stats.mannwhitneyu(a[y,j],a[~y,j],alternative='two-sided',method='asymptotic',use_continuity=True)
            assert np.isclose(ours.auroc[j],ref.statistic/(y.sum()*(~y).sum()),rtol=1e-12)
            assert np.isclose(ours.p[j],ref.pvalue,rtol=1e-10); p.append(ref.pvalue)
        assert np.allclose(ours.q,stats.false_discovery_control(p))
        features=np.load(run.CACHE/'fold_1/tax_ids.npy'); src=sparse.load_npz(run.BASE/'mgnify_matched_counts_genus_ncbi/counts.npz'); rows=np.load(run.FIXED/'source_row_indices.npy'); cols=np.load(run.FIXED/'source_column_indices.npy')
        assert len(features)==4096
        records=[]
        for f in range(1,6):
            cache=run.CACHE/f'fold_{f}'; norm=np.load(cache/'normalization.npz'); lo=norm['minimum']; hi=norm['maximum']; constant=norm['constant']; scale=np.where(constant,1,hi-lo)
            splits={part:np.load(cache/f'{part}_row_indices.npy') for part in ('train','val','test')}
            assert len(np.unique(np.concatenate(list(splits.values()))))==199150
            error=0.
            for part,indices in splits.items():
                take=np.unique(np.linspace(0,len(indices)-1,8,dtype=int)); ix=indices[take]; raw=src[rows[ix]]; denom=np.asarray(raw.sum(1)).ravel(); relative=raw[:,cols].toarray().astype(np.float64)/denom[:,None]; expected=((relative-lo)/scale).astype(np.float32); expected[:,constant]=0
                actual=np.load(cache/f'{part}_x.npy',mmap_mode='r')[take]; error=max(error,float(np.abs(actual-expected).max())); assert np.array_equal(expected,actual)
            records.append({'fold':f,'partition_disjoint_and_exhaustive':True,'normalized_input_reconstruction_max_error':error})
        h=np.load(run.BASE/'hilbert_encoding/hilbert.npy'); inverse=np.argsort(h); v=np.arange(4096); assert np.array_equal(v[h][inverse],v)
        out=Path(__file__).parent/'checks/numerical'; out.mkdir(parents=True,exist_ok=True)
        run.js(out/'checks.json',{'association_matches_scipy':True,'bh_matches_scipy':True,'hilbert_roundtrip':True,'cached_input_reconstruction':records,'verification_code_sha256':run.sha(Path(__file__))})
        print('All independent numerical and cached-input checks passed.',flush=True)

    main()


def diagnose_predictions():
    """Describe saved/recomputed decision-boundary differences without changing predictions."""

    import json

    import numpy as np

    import pandas as pd

    import run

    out=run.Path(__file__).parent/'results/formal_v1'; rows=[]

    out=run.Path(__file__).parent/'results/formal_v1'; rows=[]

    for f in range(1,6):
        z=np.load(run.MODELS/f'fold_{f}/predictions.npz'); new=pd.read_csv(out/f'fold_{f}/recomputed_test_predictions.csv'); old=z['probabilities']; p=new[[f'probability_{i}' for i in range(28)]].to_numpy(); diff=p.argmax(1)!=old.argmax(1)
        x=np.load(run.CACHE/f'fold_{f}/test_x.npy',mmap_mode='r')
        for i in np.flatnonzero(diff):
            oi=int(old[i].argmax()); ni=int(p[i].argmax()); rows.append({'fold':f,'sample_id':new.sample_id[i],'true_index':int(new.true_index[i]),'old_index':oi,'new_index':ni,'old_margin_between_changed_classes':float(old[i,oi]-old[i,ni]),'new_margin_between_changed_classes':float(p[i,ni]-p[i,oi]),'max_probability_error':float(np.abs(p[i]-old[i]).max()),'normalized_input_max':float(x[i].max())})

    pd.DataFrame(rows).to_csv(out/'prediction_boundary_diagnostics.csv',index=False)

    run.js(out/'prediction_boundary_notes.json',{'class_disagreements':len(rows),'interpretation':'CPU reanalysis versus archived CUDA evaluation; small probability changes at close competing classes are compatible with numerical/device/batch effects, not independently proven to arise from one specific cause. All differences retained; no source predictions modified.','code_sha256':run.sha(run.Path(__file__))})


def delivery():
    """Final cross-file validation and version manifest after the numerical audit."""

    import json

    from pathlib import Path

    import numpy as np

    import pandas as pd

    import run

    base=Path(__file__).parent; out=base/'results/formal_v1'

    base=Path(__file__).parent; out=base/'results/formal_v1'

    assert json.loads((out/'final_audit.json').read_text())['passed']

    assert json.loads((out/'prediction_audit.json').read_text())['all_test_samples']==199150

    samples=pd.read_csv(out/'plot_samples.csv'); all_marine=pd.read_csv(out/'marine_test_predictions.csv'); oof=pd.read_csv(out/'recomputed_oof_predictions.csv')

    samples=pd.read_csv(out/'plot_samples.csv'); all_marine=pd.read_csv(out/'marine_test_predictions.csv'); oof=pd.read_csv(out/'recomputed_oof_predictions.csv')

    samples=pd.read_csv(out/'plot_samples.csv'); all_marine=pd.read_csv(out/'marine_test_predictions.csv'); oof=pd.read_csv(out/'recomputed_oof_predictions.csv')

    assert samples.sample_id.is_unique and len(samples)==39734

    assert len(all_marine)==40426 and all_marine.sample_id.is_unique

    ref=oof[(oof.true_index==14)&(oof.predicted_index==14)]

    assert set(ref.sample_id)==set(samples.sample_id)

    cam=np.load(out/'pooled_mean_cam.npy'); stats=json.loads((out/'pooled_cam_statistics.json').read_text())

    cam=np.load(out/'pooled_mean_cam.npy'); stats=json.loads((out/'pooled_cam_statistics.json').read_text())

    for j,layer in enumerate(run.LAYERS):
        mask=cam[j]>cam[j].mean()+cam[j].std(ddof=0); assert np.array_equal(mask,np.load(out/f'{layer}_mask.npy')); assert mask.sum()==stats['layers'][j]['highlight_positions']
        assert np.all((cam[j]>=0)&(cam[j]<=1))
        for f in range(1,6):
            v=np.load(out/f'fold_{f}/mean_cam.npy')[j]; assert np.array_equal(v>v.mean()+v.std(ddof=0),np.load(out/f'fold_{f}/{layer}_mask.npy'))

    full=pd.read_csv(out/'genus_complete_with_effects.csv'); assert len(full)==4096 and full.feature_index.is_unique

    full=pd.read_csv(out/'genus_complete_with_effects.csv'); assert len(full)==4096 and full.feature_index.is_unique

    assert all(full.groupby('feature_index').size()==1)

    from PIL import Image

    im=Image.open(out/'publication/marine_input_and_stages12.png'); assert min(im.info['dpi'])>=300

    im=Image.open(out/'publication/marine_input_and_stages12.png'); assert min(im.info['dpi'])>=300

    import xml.etree.ElementTree as ET

    svg=ET.parse(out/'publication/marine_input_and_stages12.svg'); labels=[''.join(el.itertext()) for el in svg.getroot().iter() if el.tag.endswith('}text')]

    svg=ET.parse(out/'publication/marine_input_and_stages12.svg'); labels=[''.join(el.itertext()) for el in svg.getroot().iter() if el.tag.endswith('}text')]

    assert not set(full.genus).intersection(labels)

    run.js(out/'delivery_audit.json',{'passed':True,'all_outer_test_predictions_recomputed':True,'test_count':len(oof),'marine_test_count':len(all_marine),'correct_marine_count':len(samples),'same_samples_all_panels':True,'all_threshold_masks_verified_ddof0':True,'full_4096_table':True,'no_genus_figure_annotations':True,'png_dpi':list(im.info['dpi']),'figure_visual_inspection':'See layout_preview.png; shared CAM scale, readable 9 pt labels, cyan threshold contours','versions':json.loads((out/'config.json').read_text())['versions']})

    paths=[p for p in base.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='artifact_manifest.json']

    run.js(base/'artifact_manifest.json',{str(p.relative_to(base)):{'bytes':p.stat().st_size,'sha256':run.sha(p)} for p in sorted(paths)})

    print('Delivery audit passed')


def perturb_worker():
    """Optional independent fold worker; same intervention as run.py, no shared summaries."""

    import argparse, time

    import numpy as np

    import pandas as pd

    import torch

    import run

    def main():
        ap=argparse.ArgumentParser(); ap.add_argument('--output',type=run.Path,required=True); ap.add_argument('--fold',type=int,choices=range(1,6),required=True); args=ap.parse_args(); out=args.output; fold=args.fold
        cfg=run.json.loads((out/'config.json').read_text()); assert not cfg['smoke']; assert cfg['sources']['model_training/gradcam_marine_reanalysis/run.py']==run.sha(run.Path(run.__file__))
        torch.set_num_threads(cfg['threads']); torch.manual_seed(0); torch.use_deterministic_algorithms(True); batch=cfg['batch']; dest=out/f'fold_{fold}'; cache=run.CACHE/f'fold_{fold}'
        if (dest/'perturbation_summary.csv').exists(): return
        selected=pd.read_csv(out/'perturbation_targets.csv'); indices=selected.feature_index.to_numpy(dtype=int); train=np.load(cache/'train_x.npy',mmap_mode='r'); x=np.load(cache/'test_x.npy',mmap_mode='r'); ids=np.load(cache/'test_ids.npy'); ix=np.load(dest/'correct_marine_test_indices.npy')
        med=np.array([np.median(train[:,j]) for j in indices]); del train
        np.savez(dest/'perturbation_training_medians.npz',feature_indices=indices,normalized_training_medians=med)
        reuse=None
        if cfg['reuse_perturbations']:
            source=run.Path(cfg['reuse_perturbations'])/f'fold_{fold}/perturbation_distributions.npz'
            assert run.sha(source)==cfg['reused_perturbation_sources'][str(source)] if str(source) in cfg['reused_perturbation_sources'] else run.sha(source) in cfg['reused_perturbation_sources'].values()
            reuse=np.load(source); assert np.array_equal(reuse['test_indices'],ix) and np.array_equal(reuse['sample_ids'],ids[ix])
        m=run.model(fold); h=torch.from_numpy(np.load(run.BASE/'hilbert_encoding/hilbert.npy')); records=[]; distributions=[]
        with torch.no_grad():
            original=[]
            for start in range(0,len(ix),batch): original.extend(m(run.encode(torch.from_numpy(np.array(x[ix[start:start+batch]])),'hilbert',h))[:,14].numpy())
            original=np.asarray(original)
            for j,row in enumerate(selected.itertuples()):
                oldk=np.flatnonzero(reuse['feature_indices']==int(row.feature_index)) if reuse is not None else []
                if len(oldk): v=np.array(reuse['deltas'][oldk[0]],copy=True)
                else:
                    deltas=[]
                    for start in range(0,len(ix),batch):
                        xb=torch.from_numpy(np.array(x[ix[start:start+batch]])); xb[:,int(row.feature_index)]=float(med[j]); changed=m(run.encode(xb,'hilbert',h))[:,14].numpy(); deltas.extend(original[start:start+len(changed)]-changed)
                    v=np.asarray(deltas)
                distributions.append(v)
                records.append({'fold':fold,'feature_index':int(row.feature_index),'tax_id':int(row.tax_id),'genus':row.genus,'role':'candidate' if row.candidate else 'CAM_control','n':len(v),'member_count':1,'mean_delta':float(v.mean()),'sd_delta':float(v.std(ddof=0)) if len(v)>1 else 0.,'q05':float(np.quantile(v,.05)),'median':float(np.median(v)),'q95':float(np.quantile(v,.95)),'positive_sample_fraction':float((v>0).mean())})
                print(f'Perturb fold {fold}: {j+1}/{len(selected)} {row.genus} mean={v.mean():.5f}',flush=True)
        np.savez_compressed(dest/'perturbation_distributions.npz',deltas=np.stack(distributions),feature_indices=indices,sample_ids=ids[ix],test_indices=ix)
        run.js(dest/'perturbation_worker_provenance.json',{'worker_sha256':run.sha(run.Path(__file__)),'protocol_code_sha256':run.sha(run.Path(run.__file__)),'fold':fold,'batch':batch,'threads':cfg['threads'],'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'numpy':np.__version__,'torch':torch.__version__})
        tmp=dest/'perturbation_summary.worker.tmp'; pd.DataFrame(records).to_csv(tmp,index=False); tmp.replace(dest/'perturbation_summary.csv')

    main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['preflight', 'check', 'audit', 'predict', 'plot', 'report', 'verify', 'diagnose_predictions', 'delivery', 'perturb_worker'])
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    globals()[args.task]()


if __name__ == '__main__':
    main()
