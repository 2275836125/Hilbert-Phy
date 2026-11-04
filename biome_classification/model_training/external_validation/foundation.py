"""Read-only inputs and reusable external-evaluation primitives; never fits models."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAIN = HERE.parent
DATA = HERE / 'data'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def label_indices(values, classes):
    """-1 missing, -2 unseen; preserve nonmissing strings exactly as in training."""
    require(set(classes) == {str(i) for i in range(len(classes))}, 'Noncontiguous classes')
    require(len(set(classes.values())) == len(classes), 'Duplicate class names')
    lookup = {name: int(i) for i, name in classes.items()}
    return np.array([-1 if v.strip().casefold() in ('', 'na', 'none') else lookup.get(v, -2)
                     for v in values], dtype=np.int64)


class ExternalData:
    def __init__(self, folder=DATA):
        self.folder = Path(folder)
        manifest = read_json(self.folder / 'manifest.json')
        for name, expected in manifest['files'].items():
            require(sha(self.folder / name) == expected, f'Data hash mismatch: {name}')
        self.counts = sparse.load_npz(self.folder / 'counts.npz').tocsr()
        self.counts.sum_duplicates()
        self.counts.eliminate_zeros()
        with np.load(self.folder / 'counts.npz', allow_pickle=False) as z:
            self.ids, self.tax_ids = z['sample_ids'], z['tax_ids']
        self.labels = read_csv(self.folder / 'biomes.csv')
        self.samples = read_csv(self.folder / 'samples.csv')
        metrics = read_csv(self.folder / 'sample_metrics.csv')
        features = read_csv(self.folder / 'features.csv')
        for name, rows in [('labels', self.labels), ('samples', self.samples), ('metrics', metrics)]:
            require(np.array_equal(self.ids, [r['sample_id'] for r in rows]), f'{name} row mismatch')
        require(self.counts.shape == (manifest['samples'], manifest['features']), 'Shape mismatch')
        require(len(set(self.ids)) == len(self.ids), 'Duplicate sample IDs')
        require(len(self.tax_ids) == len(set(self.tax_ids)) == self.counts.shape[1], 'Duplicate/invalid taxa')
        require(np.array_equal(self.tax_ids, [int(r['tax_id']) for r in features]), 'Feature order mismatch')
        require([int(r['column_index']) for r in features] == list(range(len(self.tax_ids))), 'Feature indices mismatch')
        require(np.issubdtype(self.counts.dtype, np.integer) and np.all(self.counts.data > 0), 'Invalid counts')
        self.denominators = np.array([float(r['total_counts_all6873']) for r in metrics])
        totals = np.asarray(self.counts.sum(axis=1, dtype=np.float64)).ravel()
        require(np.isfinite(self.denominators).all() and np.all(self.denominators > 0), 'Invalid denominators')
        require(np.all(self.denominators >= totals), 'Full denominator smaller than selected counts')
        reference = ROOT / 'data_processing/mgnify_matched_counts_genus_ncbi_top4096_min20/counts.npz'
        with np.load(reference, allow_pickle=False) as z:
            require(np.array_equal(self.tax_ids, z['tax_ids']), 'Training taxon order mismatch')
            require(not set(self.ids).intersection(z['sample_ids']), 'External/training ID overlap')
        self.manifest = manifest

    def labels_for(self, level, classes):
        return label_indices([r[str(level)] for r in self.labels], classes)

    def batches(self, rows, normalization, batch_size=256, method='genus', incidence=None, order=None):
        """Yield (row indices, float32 inputs, raw count>0 mask), without clipping.

        method='genus' for ResNet/classical/DeepPhylo; 'popphy_cnn' for node values.
        Hilbert/rowwise image reshaping and tree matrix placement happen downstream.
        """
        require(batch_size > 0, 'batch_size must be positive')
        require(method in ('genus', 'popphy_cnn'), 'Unsupported preprocessing method')
        rows = np.asarray(rows, dtype=np.int64)
        require(rows.ndim == 1 and np.all((rows >= 0) & (rows < len(self.ids))), 'Invalid row indices')
        with np.load(normalization, allow_pickle=False) as norm:
            lo, hi, constant = norm['minimum'], norm['maximum'], norm['constant']
            if 'tax_ids' in norm:
                require(np.array_equal(norm['tax_ids'], self.tax_ids), 'Normalization taxon mismatch')
            scale = norm['scale'] if 'scale' in norm else np.where(constant, 1., hi - lo)
        require(lo.ndim == 1 and hi.shape == scale.shape == constant.shape == lo.shape, 'Normalization shape mismatch')
        require(constant.dtype == np.bool_ and np.array_equal(constant, hi == lo), 'Invalid constant columns')
        require(np.isfinite(lo).all() and np.isfinite(hi).all() and np.isfinite(scale).all() and np.all(scale > 0), 'Invalid normalization')
        require(np.array_equal(scale, np.where(constant, 1., hi - lo)), 'Normalization scale mismatch')
        if method == 'popphy_cnn':
            require(incidence is not None and incidence.shape == (len(self.tax_ids), len(lo)), 'Tree incidence mismatch')
        else:
            require(len(lo) == len(self.tax_ids), 'Genus normalization size mismatch')
        if order is not None:
            order = np.asarray(order)
            require(np.array_equal(np.sort(order), np.arange(len(lo))), 'Invalid feature permutation')
        for start in range(0, len(rows), batch_size):
            ix = rows[start:start + batch_size]
            raw = self.counts[ix].toarray()
            x = raw.astype(np.float64) / self.denominators[ix, None]
            if method == 'popphy_cnn':
                x = np.log1p(x @ incidence)
            x = ((x - lo) / scale).astype(np.float32)
            x[:, constant] = 0
            require(np.isfinite(x).all(), 'Nonfinite external input')
            present = raw > 0
            if order is not None:
                x, present = x[:, order], present[:, order]
            yield ix, x, present


def classification_metrics(y, probabilities, classes):
    """Closed-set metrics with explicit absent-class AUROC semantics."""
    y = np.asarray(y)
    p = np.asarray(probabilities)
    n = len(classes)
    require(y.ndim == 1 and len(y) > 0 and np.issubdtype(y.dtype, np.integer), 'Need nonempty integer labels')
    require(p.shape == (len(y), n) and np.all((y >= 0) & (y < n)), 'Label/probability shape mismatch')
    require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(1), 1, atol=1e-6), 'Invalid probabilities')
    pred = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(n), zero_division=0)
    per_class = []
    aucs = []
    for i in range(n):
        valid = 0 < support[i] < len(y)
        auc = float(roc_auc_score(y == i, p[:, i])) if valid else None
        if valid:
            aucs.append(auc)
        per_class.append(dict(index=i, label=classes[str(i)], support=int(support[i]), precision=float(precision[i]),
                              recall=float(recall[i]), f1=float(f1[i]), auroc=auc,
                              auroc_undefined_reason=None if valid else ('no_positive_samples' if support[i] == 0 else 'no_negative_samples')))
    return dict(samples=len(y), accuracy=float(np.mean(pred == y)),
                macro_f1_all_training_classes=float(f1.mean()),
                macro_f1_present_classes=float(f1[support > 0].mean()),
                macro_auroc_valid_classes=float(np.mean(aucs)) if aucs else None,
                auroc_valid_classes=len(aucs), training_classes=n, per_class=per_class)
