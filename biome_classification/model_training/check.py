"""CPU structural checks plus real fold preprocessing audit (no model training)."""
import argparse
import hashlib
import json
import numpy as np
import torch
from common import BASE, ROOT, seed_all, read_labels, write_json, load_parts
from models import encode, build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir',type=__import__('pathlib').Path,default=ROOT / 'model_training/shared_cache')
    parser.add_argument('--output',type=__import__('pathlib').Path,default=ROOT / 'model_training/check_results.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    h = torch.as_tensor(np.load(BASE / 'hilbert_encoding/hilbert.npy'))
    assert h.shape == (4096,) and np.array_equal(np.sort(h.numpy()),np.arange(4096))
    reports = {}
    for level in range(1,6):
        _,_,_,_,report = read_labels(level); reports[str(level)] = report
    folder = args.cache_dir / 'level_1/fold_1'
    data = load_parts(folder,32)
    x = data['train'][0][:2]
    a = encode(x,'hilbert',h).flatten(1)
    b = encode(x,'rowwise',h).flatten(1)
    c = encode(x,'1d',h).flatten(1)
    assert torch.equal(a[:,torch.argsort(h)],x) and torch.equal(b,x) and torch.equal(c,x)
    assert x.dtype == torch.float32
    ids_by_part = [np.load(folder / f'{part}_ids.npy') for part in ('train','val','test')]
    sets = [set(ids.tolist()) for ids in ids_by_part]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    from scipy import sparse
    fixed = sparse.load_npz(BASE / 'mgnify_matched_counts_genus_ncbi_top4096_min20/counts.npz')
    full = sparse.load_npz(BASE / 'mgnify_matched_counts_genus_ncbi/counts.npz')
    rows = np.load(BASE / 'mgnify_matched_counts_genus_ncbi_top4096_min20/source_row_indices.npy')
    denominator = np.asarray(full.sum(axis=1)).ravel()[rows]
    training = np.load(folder / 'train_row_indices.npy')
    minimum = np.full(4096,np.inf); maximum = np.full(4096,-np.inf)
    for start in range(0,len(training),1024):
        ix = training[start:start+1024]
        relative = fixed[ix].toarray().astype(np.float64)/denominator[ix,None]
        minimum = np.minimum(minimum,relative.min(axis=0)); maximum = np.maximum(maximum,relative.max(axis=0))
    normalization = np.load(folder / 'normalization.npz')
    assert np.array_equal(minimum,normalization['minimum']) and np.array_equal(maximum,normalization['maximum'])
    scale = np.where(maximum == minimum,1,maximum-minimum)
    for part in ('train','val','test'):
        ix = np.load(folder / f'{part}_row_indices.npy')[:16]
        expected = ((fixed[ix].toarray()/denominator[ix,None]-minimum)/scale).astype(np.float32)
        expected[:,maximum == minimum] = 0
        actual = np.load(folder / f'{part}_x.npy',mmap_mode='r')[:16]
        assert np.array_equal(actual,expected)
    del full, fixed

    assert np.array_equal(normalization['minimum'].shape,[4096])
    models = {}; weights = {}
    for encoding,depth in [('hilbert',18),('hilbert',34),('hilbert',50),('rowwise',18),('rowwise',34),('rowwise',50),('1d',18)]:
        seed_all(); model = build_model(encoding,depth,len(json.loads((folder / 'classes.json').read_text())))
        digest = hashlib.sha256()
        for key,v in model.state_dict().items(): digest.update(key.encode()); digest.update(v.numpy().tobytes())
        weights[(encoding,depth)] = digest.hexdigest()
        model.eval()
        with torch.no_grad(): out = model(encode(x,encoding,h))
        assert torch.isfinite(out).all() and out.shape[0] == len(x)
        models[f'{encoding}_resnet{depth}'] = {'output_shape':list(out.shape),'parameters':sum(p.numel() for p in model.parameters()),'initial_state_sha256':digest.hexdigest()}
        del model
    for depth in (18,34,50): assert weights[('hilbert',depth)] == weights[('rowwise',depth)]
    write_json(args.output,{'hilbert_permutation':True,'three_encodings_equal_values':True,'paired_initial_weights_equal':True,'normalization_recomputed_from_training_only':True,'cached_vectors_independently_verified':True,'sample_ids_disjoint':True,'models':models,'labels':reports,'fold_manifest':json.loads((folder / 'manifest.json').read_text()),'device':'cpu'})
    print('All structural checks passed',flush=True)
if __name__ == '__main__': main()
