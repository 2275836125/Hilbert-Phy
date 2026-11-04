"""Read-only shared protocol cache plus method-local preprocessing."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from tree import FIXED, SOURCE, TAXDUMP, ROOT, RULES, PCA_CONFIG, sha, digest, write_json, audit_tree, path_distance

BASE = Path(__file__).resolve().parent

def provenance(common, method):
    signature, source = common.fingerprint()
    files = dict(source['files'])
    for p in [TAXDUMP,FIXED/'taxonomy_postorder.csv',ROOT/'data_processing/mgnify_matched_counts_genus_ncbi/features.csv']:
        files[str(p.relative_to(ROOT))] = sha(p)
    code = {str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*.py')}
    payload = {'shared_signature':signature,'shared_provenance':source,'files':files,'code':code,
               'tree_rules':RULES,'pca':PCA_CONFIG,'method':method,
               'versions':{'numpy':np.__version__,'sklearn':common.sklearn.__version__,'scipy':common.scipy.__version__,'torch':torch.__version__},
               'limitation':'4096 genera and min20 cohort preselected on all data before splits'}
    return digest(payload),payload

def prepare_tree(method,cache,signature,payload):
    cache.mkdir(parents=True,exist_ok=True)
    manifest = cache/'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['signature'] != signature: raise ValueError(f'Tree cache signature mismatch: {cache}; choose a new method cache')
        for name,h in old['artifacts'].items():
            if sha(cache/name) != h: raise ValueError(f'Corrupt tree artifact: {name}')
        return json.loads((cache/'tree_audit.json').read_text())
    ids, paths, nodes, positions, report = audit_tree(cache/'tree_audit.json')
    index = {n:i for i,n in enumerate(nodes)}
    rr = []; cc = []
    for j,path in enumerate(paths):
        for n in path: rr.append(j); cc.append(index[n])
    incidence = sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(4096,len(nodes)))
    sparse.save_npz(cache/'incidence.npz',incidence)
    np.save(cache/'positions.npy',np.array([positions[n] for n in nodes],dtype=np.int64))
    np.save(cache/'tax_ids.npy',ids)
    h,w = report['matrix_shape']
    estimate = {'matrix_shape':[h,w], 'nodes':len(nodes),'float32_matrix_bytes_per_sample':h*w*4,
                'float32_all_samples_bytes':238419*h*w*4,'float32_batch32_bytes':32*h*w*4,
                'distance_float64_bytes':4096**2*8,'embedding_float32_bytes':4096*32*4,
                'strategy':'CSR counts resident; sample tree matrices allocated per batch only; fold parameters per node, never full dense tree cohort',
                'pca_workspace':'allow approximately 1 GiB additional RAM for distance/PCA temporaries',
                'popphy_flatten_dense_parameters':(32*((h-2)//2)*((w-9)//2))*32+32,
                'popphy_checkpoint_note':'float32 weights 4 bytes/parameter; last.pt includes two Adam moments (~12 bytes/parameter plus RNG)'}
    write_json(cache/'resources.json',estimate)
    print(json.dumps(estimate,ensure_ascii=False),flush=True)
    if method == 'deepphylo_taxonomy':
        # Full uncontracted ancestor paths, exact original edge distances.
        distance = np.lib.format.open_memmap(cache/'distances.npy',mode='w+',dtype=np.float64,shape=(4096,4096))
        for i,a in enumerate(paths):
            for j in range(i+1):
                d = path_distance(a,paths[j])
                distance[i,j] = distance[j,i] = d
        distance.flush()
        pca = PCA(**{k:v for k,v in PCA_CONFIG.items() if k != 'adaptation_parameter'})
        embedding = pca.fit_transform(distance)
        np.save(cache/'embeddings.npy',embedding.astype(np.float32))
        np.savez(cache/'pca.npz',components=pca.components_,mean=pca.mean_,explained_variance=pca.explained_variance_,explained_variance_ratio=pca.explained_variance_ratio_)
    artifacts = {p.name:sha(p) for p in cache.iterdir() if p.is_file() and p.name not in ('manifest.json','.gitkeep','.prepare.lock') and not p.name.endswith('.tmp')}
    write_json(manifest,{'signature':signature,'provenance':payload,'artifacts':artifacts,'tree_signature':report['tree_signature']})
    return report

def read_counts():
    fixed = sparse.load_npz(FIXED/'counts.npz').tocsr()
    source = sparse.load_npz(SOURCE).tocsr()
    rows = np.load(FIXED/'source_row_indices.npy'); cols = np.load(FIXED/'source_column_indices.npy')
    assert fixed.shape == (238419,4096)
    # Validate numeric column mapping, blockwise to avoid a second full CSR slice.
    denominators = np.asarray(source.sum(axis=1,dtype=np.float64)).ravel()[rows]
    for start in range(0,len(rows),2048):
        if (fixed[start:start+2048] != source[rows[start:start+2048]][:,cols]).nnz:
            raise ValueError(f'Fixed/source counts differ at block {start}')
    with np.load(FIXED/'counts.npz') as z: ids = z['sample_ids']
    bad = ids[denominators <= 0].tolist()
    if bad: raise ValueError(f'Nonpositive full denominator sample IDs: {bad}')
    return fixed,denominators,ids

def shared_fold(common,args,level,fold,payload,ids):
    folder = args.shared_cache_dir/f'level_{level}'/f'fold_{fold}'
    if not (folder/'manifest.json').exists():
        raise FileNotFoundError(f'Missing shared split cache {folder}; no new split will be generated')
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['signature'] != payload['shared_signature'] or manifest['provenance'] != payload['shared_provenance']:
        raise ValueError(f'Shared cache signature/provenance mismatch: {folder}')
    label_ids,values,keep,names,_ = common.read_labels(level)
    assert np.array_equal(ids,label_ids)
    classes = json.loads((folder/'classes.json').read_text())
    assert classes == {str(i):v for i,v in enumerate(names.tolist())}
    assert np.array_equal(np.load(folder/'tax_ids.npy'),np.load(args.cache_dir/'tax_ids.npy'))
    parts = {}; seen = set()
    for part in ('train','val','test'):
        ix = np.load(folder/f'{part}_row_indices.npy'); y = np.load(folder/f'{part}_y.npy'); sid = np.load(folder/f'{part}_ids.npy')
        assert len(ix) == len(set(ix.tolist())) and not seen.intersection(ix.tolist())
        seen.update(ix.tolist())
        assert np.array_equal(ids[ix],sid) and np.array_equal(y,np.searchsorted(names,values[ix]))
        parts[part] = (ix,y,sid)
    assert seen == set(keep.tolist())
    return folder,classes,parts

class FoldDataset(Dataset):
    def __init__(self,counts,denom,rows,labels,minimum,scale,constant,incidence,method):
        self.counts,self.denom,self.rows,self.labels = counts,denom,rows,labels
        self.minimum,self.scale,self.constant = minimum,scale,constant
        self.incidence,self.method = incidence,method
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row = int(self.rows[i]); raw = self.counts[row].toarray().ravel()
        relative = raw.astype(np.float64)/self.denom[row]
        if self.method == 'popphy_cnn': relative = np.log1p(np.asarray(relative @ self.incidence).ravel())
        transformed = (relative-self.minimum)/self.scale
        transformed[self.constant] = 0
        if not np.isfinite(transformed).all(): raise ValueError('Nonfinite input')
        return torch.from_numpy(transformed.astype(np.float32)),torch.from_numpy(raw>0),int(self.labels[i])

def fit_transform(args,parts,counts,denom,incidence,signature):
    # PopPhy fits node-wise log1p abundance min/max after summing descendants.
    # DeepPhylo fits per-genus min/max, keeping count>0 as a separate input.
    folder = args.cache_dir/f'level_{args.current_level}'/f'fold_{args.current_fold}'
    folder.mkdir(parents=True,exist_ok=True)
    target = folder/'normalization.npz'; meta = folder/'manifest.json'
    split_hash = digest({p:{'rows':digest(v[0].tolist()),'y':digest(v[1].tolist()),'ids':digest(v[2].tolist())} for p,v in parts.items()})
    expected = {'signature':signature,'split_hash':split_hash}
    if meta.exists():
        saved = json.loads(meta.read_text())
        if any(saved.get(k)!=v for k,v in expected.items()) or saved['normalization_sha256'] != sha(target):
            raise ValueError(f'Fold encoding cache mismatch: {folder}')
        with np.load(target) as z: return z['minimum'],z['scale'],z['constant']
    size = incidence.shape[1] if args.method == 'popphy_cnn' else 4096
    lo = np.full(size,np.inf); hi = np.full(size,-np.inf)
    rows = parts['train'][0]
    for start in range(0,len(rows),256):
        ix = rows[start:start+256]
        x = counts[ix].toarray().astype(np.float64)/denom[ix,None]
        if args.method == 'popphy_cnn': x = np.log1p(x @ incidence)
        lo = np.minimum(lo,x.min(0)); hi = np.maximum(hi,x.max(0))
    constant = hi==lo; scale = np.where(constant,1.,hi-lo)
    np.savez(target,minimum=lo,maximum=hi,scale=scale,constant=constant)
    write_json(meta,{**expected,'normalization_sha256':sha(target),'fit_partition':'complete train fold, including for smoke',
                     'transform':'relative abundance -> selected descendant sums -> log1p -> node minmax -> zero padding' if args.method=='popphy_cnn' else 'relative abundance -> genus minmax; separate count>0 mask',
                     'constant_columns':int(constant.sum()),'clip':False})
    return lo,scale,constant
