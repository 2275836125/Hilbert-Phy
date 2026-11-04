"""DATASET_A: the sole external training-data entry point.

--input points to a prepared data_processing directory, not raw MGnify downloads.
The directory contains:
* mgnify_matched_counts_genus_ncbi/{counts.npz,features.csv}
* mgnify_matched_counts_genus_ncbi_top4096_min20/{counts.npz,biomes.csv,
  features.csv,source_row_indices.npy,source_column_indices.npy,taxonomy_postorder.csv}
* hilbert_encoding/hilbert.npy
* taxdump.tar.gz (NCBI archive used for the original feature order)
* ncbi_taxonomy/{nodes.dmp,names.dmp,merged.dmp} (Marine analysis only)

counts.npz: scipy CSR NPZ with extra sample_ids and tax_ids arrays (no pickle).
Full counts: (262942,10437); fixed counts: (238419,4096), integer nonnegative.
biomes.csv: sample_id,1,2,3,4,5; rows exactly match fixed sample_ids.
features.csv: column_index,tax_id,scientific_name; rows match matrix columns.
source_row_indices/source_column_indices: zero-based mappings to full counts.
Fixed columns must already follow NCBI postorder, siblings by numeric tax_id.
Counts must equal full_counts[rows][:,cols]; no normalization at this boundary.
hilbert.npy: permutation (4096,), row-major pixel -> ordered vector index.
The training code validates IDs, mapping, counts, and shape before fitting.

Raw-data preparation: run-ID matching -> terminal genus records -> NCBI genus
matching/merging -> occurrence >50 -> top4096 mean relative abundance ->
nonzero selected genera >=20 -> NCBI postorder. Do not aggregate species into
training genera. The three data modules contain all preprocessing operations.
Raw-data genus extraction is available as the genus subcommand; the default
classification entry point uses the prepared bundle above.
"""
import argparse
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent.parent
FILES = [
    'mgnify_matched_counts_genus_ncbi/counts.npz',
    'mgnify_matched_counts_genus_ncbi/features.csv',
    *['mgnify_matched_counts_genus_ncbi_top4096_min20/' + name for name in
      ('counts.npz', 'biomes.csv', 'features.csv', 'source_row_indices.npy',
       'source_column_indices.npy', 'taxonomy_postorder.csv')],
    'hilbert_encoding/hilbert.npy', 'taxdump.tar.gz',
]
MARINE = ['ncbi_taxonomy/' + name for name in ('nodes.dmp', 'names.dmp', 'merged.dmp')]


def import_dataset(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='DATASET_A prepared data_processing directory')
    parser.add_argument('--check-only', action='store_true', help='Check required files without copying or numerical validation')
    parser.add_argument('--marine', action='store_true', help='Also import unpacked taxonomy for Marine analysis')
    args = parser.parse_args(argv)
    source = args.input.resolve()
    destination = HERE / 'data_processing'
    names = FILES + (MARINE if args.marine else [])
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        parser.error('DATASET_A missing files:\n' + '\n'.join(missing))
    if source == destination.resolve():
        print('DATASET_A already in the local data directory; required files are present.')
        return
    if args.check_only:
        print(f'DATASET_A: {len(names)} required files present; no files copied.')
        return
    existing = [name for name in names if (destination / name).exists()]
    if existing:
        parser.error('Refusing to overwrite local data; use a fresh checkout or --check-only:\n' + '\n'.join(existing))
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    print(f'Imported {len(names)} files. Training preparation performs numerical validation.')



def match_counts():
    """Extract labelled runs from full SSU taxonomy tables into a sparse count matrix."""

    import csv
    import json
    import re
    from collections import Counter
    from pathlib import Path
    import numpy as np
    from scipy import sparse

    BASE = Path(__file__).resolve().parent

    OUT = BASE / 'mgnify_matched_counts'

    def main():
        with (BASE / 'mgnify_biomes.csv').open(newline='') as f:
            reader = csv.reader(f)
            label_header = next(reader)
            labels = list(reader)
        wanted = {r[0] for r in labels}
        assert len(wanted) == len(labels)
        paths = sorted(p for p in (BASE / 'raw_data/studies').glob('*/abundance/*taxonomy_abundances_SSU*.tsv')
                       if '_phylum_' not in p.name)
        selected = {}
        headers = {}
        for p in paths:
            with p.open() as f:
                header = next(csv.reader(f, delimiter='\t'))
            headers[p] = header
            version = tuple(map(int, re.search(r'_v([\d.]+)\.tsv$', p.name)[1].split('.')))
            for col, sid in enumerate(header[1:], 1):
                if sid in wanted and (sid not in selected or version > selected[sid][0]):
                    selected[sid] = (version, p, col)
        kept = [r for r in labels if r[0] in selected]
        sample_ids = [r[0] for r in kept]
        sample_index = {sid: i for i, sid in enumerate(sample_ids)}
        by_file = {}
        for sid in sample_ids:
            _, p, col = selected[sid]
            by_file.setdefault(p, []).append((col, sample_index[sid]))
        print(f'Found {len(kept)} samples in {len(by_file)} selected full taxonomy files', flush=True)
        features = {}
        rr, cc, vv = [], [], []
        noninteger_files = []
        for num, (p, columns) in enumerate(sorted(by_file.items()), 1):
            source_cols = np.array([c - 1 for c, _ in columns])
            dest_rows = np.array([r for _, r in columns], dtype=np.int32)
            file_rows, file_cols, file_vals = [], [], []
            with p.open() as f:
                next(f)
                for line in f:
                    taxon, sep, rest = line.rstrip('\r\n').partition('\t')
                    if not sep:
                        raise ValueError(f'Missing counts: {p}: {taxon}')
                    vals = np.fromstring(rest, sep='\t', dtype=np.float64)
                    if len(vals) != len(headers[p]) - 1:
                        raise ValueError(f'Invalid row width in {p}: {taxon}')
                    values = vals[source_cols]
                    if not np.all(np.isfinite(values)) or np.any(values < 0):
                        raise ValueError(f'Invalid counts in {p}')
                    if np.any(values != np.floor(values)):
                        noninteger_files.append(str(p))
                        raise ValueError(f'Noninteger counts in {p}')
                    feature = features.setdefault(taxon, len(features))
                    nz = np.flatnonzero(values)
                    if len(nz):
                        file_rows.append(dest_rows[nz])
                        file_cols.append(np.full(len(nz), feature, dtype=np.int32))
                        file_vals.append(values[nz].astype(np.int64))
            if file_vals:
                rr.append(np.concatenate(file_rows))
                cc.append(np.concatenate(file_cols))
                vv.append(np.concatenate(file_vals))
            if num % 100 == 0:
                print(f'Processed {num}/{len(by_file)} files', flush=True)
        matrix = sparse.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                                   shape=(len(kept), len(features)), dtype=np.int64).tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        # Extra ID arrays are ignored by scipy.sparse.load_npz, but accessible with numpy.load.
        np.savez_compressed(OUT / 'counts.npz', format=np.array('csr', dtype='S3'),
                            shape=np.array(matrix.shape), data=matrix.data,
                            indices=matrix.indices, indptr=matrix.indptr,
                            sample_ids=np.array(sample_ids), feature_ids=np.array(list(features)))
        with (OUT / 'biomes.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_id'] + label_header[1:])
            w.writerows(kept)
        loaded = sparse.load_npz(OUT / 'counts.npz')
        with np.load(OUT / 'counts.npz') as archive:
            assert archive['sample_ids'].tolist() == sample_ids
            assert archive['feature_ids'].tolist() == list(features)
        assert loaded.shape == matrix.shape and (loaded != matrix).nnz == 0
        with (OUT / 'biomes.csv').open() as f:
            assert [r['sample_id'] for r in csv.DictReader(f)] == sample_ids
        summary = dict(input_samples=len(labels), matched_samples=len(kept),
                       unmatched_samples=len(labels)-len(kept), features=len(features),
                       nonzero_entries=matrix.nnz, selected_files=len(by_file),
                       versions=dict(Counter('.'.join(map(str, selected[s][0])) for s in sample_ids)),
                       zero_count_samples=int(np.sum(np.diff(matrix.indptr) == 0)),
                       selection='highest numeric pipeline version; equal versions use lexicographically first path',
                       feature_semantics='exact full taxonomy strings; absent entries filled with zero; no rank aggregation',
                       outputs=['mgnify_matched_counts/counts.npz', 'mgnify_matched_counts/biomes.csv'])
        (BASE / 'reports/mgnify_matched_counts_summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)

    main()


def filter_genera():
    """Keep genera present in >50 samples, then top 4096 by mean abundance."""

    from pathlib import Path
    import json
    import shutil
    import numpy as np
    import pandas as pd
    from scipy import sparse

    ROOT = Path(__file__).resolve().parent

    SOURCE = ROOT / 'mgnify_matched_counts_genus_ncbi'

    OUT = ROOT / 'mgnify_matched_counts_genus_ncbi_top4096'

    OUT.mkdir(exist_ok=True)

    x = sparse.load_npz(SOURCE / 'counts.npz').tocsr()

    x.sum_duplicates()

    x.eliminate_zeros()

    assert np.all(x.data > 0)

    features = pd.read_csv(SOURCE / 'features.csv')

    with np.load(SOURCE / 'counts.npz') as z:
        samples = z['sample_ids']
        assert np.array_equal(z['tax_ids'], features.tax_id.to_numpy())

    labels = pd.read_csv(SOURCE / 'biomes.csv', keep_default_na=False)

    assert np.array_equal(labels.sample_id.to_numpy(), samples)

    assert np.array_equal(features.column_index, np.arange(x.shape[1]))

    occ = np.bincount(x.indices, minlength=x.shape[1])

    totals = np.asarray(x.sum(axis=1)).ravel()

    valid = totals > 0

    weights = np.zeros(x.shape[0], dtype=float)

    weights[valid] = 1.0 / totals[valid]

    mean = np.asarray(x.T.dot(weights)).ravel() / valid.sum()

    assert np.isclose(mean.sum(), 1)

    audit = features.rename(columns={'column_index': 'source_column_index'}).copy()

    audit['n_samples_present'] = occ

    audit['mean_relative_abundance'] = mean

    audit['passes_occurrence_filter'] = occ > 50

    eligible = audit.loc[audit.passes_occurrence_filter].sort_values(
        ['mean_relative_abundance', 'tax_id'], ascending=[False, True])

    assert len(eligible) >= 4096

    selected = eligible.head(4096).copy()

    columns = selected.source_column_index.to_numpy(dtype=np.int64)

    np.save(OUT / 'source_column_indices.npy', columns)

    audit['abundance_rank_after_occurrence_filter'] = pd.Series(
        np.arange(1, len(eligible)+1), index=eligible.index).reindex(audit.index).astype('Int64')

    audit['selected'] = audit.index.isin(selected.index)

    audit['filter_reason'] = np.where(~audit.passes_occurrence_filter, 'occurrence_le_50',
                                       np.where(audit.selected, 'retained', 'outside_top4096'))

    audit.to_csv(OUT / 'feature_selection.csv', index=False)

    selected.insert(0, 'column_index', np.arange(len(selected)))

    selected['abundance_rank_after_occurrence_filter'] = np.arange(1, len(selected)+1)

    selected.to_csv(OUT / 'features.csv', index=False)

    result = x[:, columns].tocsr()

    result.sort_indices()

    np.savez_compressed(OUT / 'counts.npz', format=np.array('csr', dtype='S3'),
                        shape=np.array(result.shape), data=result.data,
                        indices=result.indices, indptr=result.indptr,
                        sample_ids=samples, feature_ids=selected.tax_id.astype(str).to_numpy(dtype=str),
                        tax_ids=selected.tax_id.to_numpy(dtype=np.int64),
                        genus_names=selected.scientific_name.to_numpy(dtype=str))

    shutil.copyfile(SOURCE / 'biomes.csv', OUT / 'biomes.csv')

    loaded = sparse.load_npz(OUT / 'counts.npz')

    assert loaded.shape == (x.shape[0], 4096)

    assert (loaded != x[:, columns]).nnz == 0

    assert np.all(np.bincount(loaded.indices, minlength=4096) > 50)

    with np.load(OUT / 'counts.npz') as z:
        assert np.array_equal(z['sample_ids'], samples)
        assert np.array_equal(z['tax_ids'], selected.tax_id)

    assert (OUT / 'biomes.csv').read_bytes() == (SOURCE / 'biomes.csv').read_bytes()

    summary = {'source': str(SOURCE.relative_to(ROOT)), 'samples': x.shape[0],
               'input_genera': x.shape[1], 'removed_occurrence_le_50': int((occ <= 50).sum()),
               'eligible_genera': len(eligible), 'removed_by_abundance_rank': len(eligible)-4096,
               'retained_genera': 4096, 'nonzero_entries': result.nnz,
               'zero_count_samples_before': int((~valid).sum()),
               'zero_count_samples_after': int((np.diff(result.indptr) == 0).sum()),
               'abundance_mean_denominator_samples': int(valid.sum()),
               'minimum_selected_mean_relative_abundance': float(selected.mean_relative_abundance.min()),
               'minimum_selected_sample_occurrence': int(selected.n_samples_present.min()),
               'mean_fraction_of_source_genus_abundance_retained': float(selected.mean_relative_abundance.sum()),
               'tie_breaker': 'tax_id ascending', 'column_order': 'mean abundance descending',
               'verification': 'Saved counts equal source selected columns; all sample IDs, labels and tax IDs aligned; all selected occurrence >50.'}

    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')


    print(json.dumps(summary, ensure_ascii=False, indent=2))


def filter_samples():
    """Retain samples with at least 20 nonzero genera in top-4096 matrix."""

    from pathlib import Path
    import json
    import shutil
    import numpy as np
    import pandas as pd
    from scipy import sparse

    ROOT = Path(__file__).resolve().parent

    SOURCE = ROOT / 'mgnify_matched_counts_genus_ncbi_top4096'

    OUT = ROOT / 'mgnify_matched_counts_genus_ncbi_top4096_min20'

    OUT.mkdir(exist_ok=True)

    x = sparse.load_npz(SOURCE / 'counts.npz').tocsr()

    x.sum_duplicates()

    x.eliminate_zeros()

    assert np.all(x.data > 0)

    n = np.diff(x.indptr)

    keep = n >= 20

    rows = np.flatnonzero(keep)

    labels = pd.read_csv(SOURCE / 'biomes.csv', keep_default_na=False)

    with np.load(SOURCE / 'counts.npz') as z:
        metadata = {key: z[key] for key in ['sample_ids', 'feature_ids', 'tax_ids', 'genus_names']}

    assert np.array_equal(metadata['sample_ids'], labels.sample_id.to_numpy())

    metadata['sample_ids'] = metadata['sample_ids'][keep]

    result = x[rows].tocsr()

    np.savez_compressed(OUT / 'counts.npz', format=np.array('csr', dtype='S3'),
                        shape=np.array(result.shape), data=result.data,
                        indices=result.indices, indptr=result.indptr, **metadata)

    labels.loc[keep].to_csv(OUT / 'biomes.csv', index=False)

    shutil.copyfile(SOURCE / 'features.csv', OUT / 'features.csv')

    shutil.copyfile(SOURCE / 'source_column_indices.npy', OUT / 'source_column_indices.npy')

    np.save(OUT / 'source_row_indices.npy', rows)

    audit = labels.copy()

    audit.insert(1, 'n_nonzero_genera', n)

    audit['retained'] = keep

    audit.to_csv(OUT / 'sample_selection.csv', index=False)

    audit.loc[~keep].to_csv(OUT / 'removed_samples.csv', index=False)

    loaded = sparse.load_npz(OUT / 'counts.npz')

    assert loaded.shape == (int(keep.sum()), 4096)

    assert (loaded != x[rows]).nnz == 0

    assert np.all(np.diff(loaded.indptr) >= 20)

    saved_labels = pd.read_csv(OUT / 'biomes.csv', keep_default_na=False)

    assert saved_labels.equals(labels.loc[keep].reset_index(drop=True))

    with np.load(OUT / 'counts.npz') as z:
        assert np.array_equal(z['sample_ids'], saved_labels.sample_id.to_numpy())
        assert all(np.array_equal(z[key], metadata[key]) for key in metadata)

    assert (OUT / 'features.csv').read_bytes() == (SOURCE / 'features.csv').read_bytes()

    summary = {'source': str(SOURCE.relative_to(ROOT)), 'input_samples': x.shape[0],
               'removed_samples_nonzero_genera_lt20': int((~keep).sum()),
               'retained_samples': result.shape[0], 'retained_genera': result.shape[1],
               'samples_with_exactly_20_genera_retained': int((n == 20).sum()),
               'minimum_nonzero_genera_after': int(np.diff(result.indptr).min()),
               'nonzero_entries': result.nnz,
               'verification': 'Saved counts match source selected rows; sample IDs and all five labels aligned; feature table and column metadata preserved.'}

    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')


    print(json.dumps(summary, ensure_ascii=False, indent=2))


def extract_genera():
    """Select terminal nonempty g__ paths; do not aggregate species records."""
    import numpy as np
    from scipy import sparse
    base = Path(__file__).resolve().parent
    source = base / 'mgnify_matched_counts'
    target = base / 'mgnify_matched_counts_genus'
    with np.load(source / 'counts.npz', allow_pickle=False) as z:
        features = z['feature_ids']; ids = z['sample_ids']
    keep = [i for i, value in enumerate(features.astype(str))
            if value.split(';')[-1].startswith('g__') and value.split(';')[-1][3:].strip()]
    if not keep:
        raise ValueError('No terminal genus records found')
    x = sparse.load_npz(source / 'counts.npz')[:, keep].tocsr()
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / 'counts.npz', format=np.array('csr', dtype='S3'),
                        shape=np.array(x.shape), data=x.data, indices=x.indices, indptr=x.indptr,
                        sample_ids=ids, feature_ids=features[keep])
    shutil.copy2(source / 'biomes.csv', target / 'biomes.csv')


def main():
    parser = argparse.ArgumentParser(description='DATASET_A import and abundance/sample filtering')
    sub = parser.add_subparsers(dest='task', required=True)
    sub.add_parser('import', help='Import DATASET_A; append --help for the detailed data interface', add_help=False)
    for task in ('match-counts', 'genus', 'features', 'samples', 'select'):
        sub.add_parser(task)
    args, rest = parser.parse_known_args()
    if args.task == 'import':
        import_dataset(rest)
        return
    if rest:
        parser.error('Unexpected arguments: ' + ' '.join(rest))
    if args.task == 'match-counts':
        (Path(__file__).resolve().parent / 'reports').mkdir(exist_ok=True)
        match_counts()
    elif args.task == 'genus':
        extract_genera()
    elif args.task == 'features':
        filter_genera()
    elif args.task == 'samples':
        filter_samples()
    else:
        filter_genera()
        filter_samples()


if __name__ == '__main__':
    main()
