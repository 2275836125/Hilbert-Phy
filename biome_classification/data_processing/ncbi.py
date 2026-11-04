"""NCBI taxonomy extraction, genus matching, and deterministic postorder."""
import argparse
from pathlib import Path


def _sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_taxonomy():
    """Validate a complete local taxdump archive before publishing required dump files."""

    import gzip
    import hashlib
    import json
    import shutil
    import tarfile
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    BASE = Path(__file__).resolve().parent

    ARCHIVE = BASE / 'taxdump.tar.gz'

    DEST = BASE / 'ncbi_taxonomy'

    def main():
        initial = ARCHIVE.stat()
        # Read through gzip EOF, including its checksum, before extracting anything.
        with gzip.open(ARCHIVE, 'rb') as f:
            while f.read(1024 * 1024):
                pass
        digest = hashlib.sha256()
        with ARCHIVE.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        DEST.mkdir(exist_ok=True)
        entries = []
        with tempfile.TemporaryDirectory(dir=DEST, prefix='extract_') as temporary:
            staged = Path(temporary)
            with tarfile.open(ARCHIVE, 'r:gz') as archive:
                for member in archive:
                    if member.isfile() and Path(member.name).name == member.name:
                        with archive.extractfile(member) as source, (staged / member.name).open('wb') as output:
                            shutil.copyfileobj(source, output)
                        assert (staged / member.name).stat().st_size == member.size
                        entries.append({'name': member.name, 'size': member.size, 'archive_mtime': member.mtime})
            assert (staged / 'names.dmp').is_file() and (staged / 'nodes.dmp').is_file()
            final = ARCHIVE.stat()
            if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
                raise RuntimeError('Archive changed during validation; retry once copying/downloading finishes.')
            for path in staged.iterdir():
                path.replace(DEST / path.name)
        metadata = {'source_archive': str(ARCHIVE), 'archive_bytes': initial.st_size,
                    'sha256': digest.hexdigest(), 'gzip_integrity_verified': True,
                    'extracted_utc': datetime.now(timezone.utc).isoformat(), 'files': entries}
        (DEST / 'archive_metadata.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2))

    main()


def match_genera():
    """Match genus features to NCBI Taxonomy and sum counts sharing a tax_id.

    Requires ncbi_taxonomy/names.dmp and nodes.dmp extracted from taxdump.tar.gz.
    Run with the same SciPy C++ runtime setup as prepare_matched_counts.py.
    """

    import csv
    import hashlib
    import json
    import shutil
    import unicodedata
    from collections import Counter, defaultdict
    from pathlib import Path
    import numpy as np
    from scipy import sparse

    BASE = Path(__file__).resolve().parent

    DB = BASE / 'ncbi_taxonomy'

    SOURCE = BASE / 'mgnify_matched_counts_genus'

    OUT = BASE / 'mgnify_matched_counts_genus_ncbi'

    NAME_CLASSES = {'scientific name', 'synonym', 'equivalent name', 'genbank synonym'}

    def normalize(name):
        return ' '.join(unicodedata.normalize('NFKC', name).replace('_', ' ').split()).casefold()

    def records(path):
        with path.open() as f:
            for line in f:
                yield [v.strip() for v in line.split('|')]

    def main():
        for filename in ['nodes.dmp', 'names.dmp']:
            if not (DB / filename).is_file():
                raise SystemExit(f'Missing {DB / filename}; download and extract NCBI taxdump first.')
        with np.load(SOURCE / 'counts.npz') as z:
            samples = z['sample_ids'].copy()
            features = z['feature_ids'].tolist()
        wanted = {normalize(t.split(';')[-1][3:]) for t in features}
        parent, genus = {}, set()
        for row in records(DB / 'nodes.dmp'):
            tid, pid = int(row[0]), int(row[1])
            parent[tid] = pid
            if row[2] == 'genus':
                genus.add(tid)
        # Only ancestor names are needed for resolving duplicate genus names.
        relevant = set(genus)
        for tid in genus:
            seen = set()
            current = parent[tid]
            while current not in seen and current not in relevant:
                relevant.add(current)
                seen.add(current)
                current = parent.get(current, current)
        names = {}
        aliases = defaultdict(set)
        scientific = defaultdict(set)
        for row in records(DB / 'names.dmp'):
            tid, name, _, kind = row[:4]
            tid = int(tid)
            normalized = normalize(name)
            if kind == 'scientific name' and tid in relevant:
                names[tid] = name
            if tid in genus and normalized in wanted and kind in NAME_CLASSES:
                aliases[normalized].add(tid)
                if kind == 'scientific name':
                    scientific[normalized].add(tid)
        lineage_cache = {}

        def lineage(tid):
            if tid not in lineage_cache:
                result, seen = set(), set()
                current = parent[tid]
                while current not in seen:
                    seen.add(current)
                    if current in names:
                        result.add(normalize(names[current]))
                    current = parent.get(current, current)
                lineage_cache[tid] = result
            return lineage_cache[tid]

        assignments, detail = [], []
        for feature in features:
            name = feature.split(';')[-1][3:]
            normalized = normalize(name)
            candidates = scientific.get(normalized) or aliases.get(normalized, set())
            method = 'scientific_name' if scientific.get(normalized) else 'synonym'
            tid = None
            if len(candidates) == 1:
                tid = next(iter(candidates))
            elif len(candidates) > 1:
                ancestors = {normalize(v.split('__', 1)[1]) for v in feature.split(';')[:-1]
                             if '__' in v and v.split('__', 1)[1].strip()}
                scores = {candidate: len(ancestors & lineage(candidate)) for candidate in candidates}
                best = max(scores.values())
                winners = [candidate for candidate, score in scores.items() if score == best]
                if best > 0 and len(winners) == 1:
                    tid = winners[0]
                    method += '_lineage_disambiguated'
            if tid is None:
                method = 'unmatched' if not candidates else 'ambiguous'
            assignments.append(tid)
            detail.append([feature, name, method, tid if tid is not None else '',
                           names.get(tid, ''), ';'.join(map(str, sorted(candidates)))])
        taxids = sorted({t for t in assignments if t is not None})
        if not taxids:
            raise SystemExit('No genus features matched; refusing to create an empty dataset.')
        index = {t: i for i, t in enumerate(taxids)}
        old_columns = [i for i, tid in enumerate(assignments) if tid is not None]
        new_columns = [index[assignments[i]] for i in old_columns]
        projection = sparse.csr_matrix((np.ones(len(old_columns), dtype=np.int64),
                                       (old_columns, new_columns)), shape=(len(features), len(taxids)))
        source = sparse.load_npz(SOURCE / 'counts.npz')
        result = (source @ projection).tocsr()
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()
        expected_sums = np.asarray(source[:, old_columns].sum(axis=1)).ravel()
        assert np.array_equal(np.asarray(result.sum(axis=1)).ravel(), expected_sums)
        OUT.mkdir(exist_ok=True)
        np.savez_compressed(OUT / 'counts.npz', format=np.array('csr', dtype='S3'),
                            shape=np.array(result.shape), data=result.data,
                            indices=result.indices, indptr=result.indptr,
                            sample_ids=samples, feature_ids=np.array([str(t) for t in taxids]),
                            tax_ids=np.array(taxids, dtype=np.int64),
                            genus_names=np.array([names[t] for t in taxids]))
        shutil.copyfile(SOURCE / 'biomes.csv', OUT / 'biomes.csv')
        with (OUT / 'features.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['column_index', 'tax_id', 'scientific_name', 'merged_source_features'])
            frequencies = Counter(t for t in assignments if t is not None)
            writer.writerows([i, tid, names[tid], frequencies[tid]] for i, tid in enumerate(taxids))
        with (OUT / 'genus_mapping.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source_feature', 'source_genus', 'status', 'tax_id', 'scientific_name', 'candidate_tax_ids'])
            writer.writerows(detail)
        loaded = sparse.load_npz(OUT / 'counts.npz')
        assert loaded.shape == result.shape and (loaded != result).nnz == 0
        with (OUT / 'biomes.csv').open() as f:
            assert [r['sample_id'] for r in csv.DictReader(f)] == samples.tolist()
        summary = {'source_features': len(features), 'matched_source_features': len(old_columns),
                   'discarded_source_features': len(features) - len(old_columns),
                   'unique_ncbi_genus_tax_ids': len(taxids),
                   'columns_merged_away': len(old_columns) - len(taxids),
                   'samples': result.shape[0], 'nonzero_entries': result.nnz,
                   'zero_count_samples': int(np.sum(np.diff(result.indptr) == 0)),
                   'mapping_status': dict(Counter(r[2] for r in detail)),
                   'name_classes': sorted(NAME_CLASSES),
                   'normalization': 'Unicode NFKC, underscores to spaces, whitespace normalization, casefold; no fuzzy matching',
                   'database_sha256': {filename: sha256(DB / filename) for filename in ['names.dmp', 'nodes.dmp']}}
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)

    def sha256(path):
        value = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                value.update(chunk)
        return value.hexdigest()

    main()


def _order_genera():
    """Order selected genera by NCBI post-order; siblings use numeric tax_id order."""

    import csv
    import hashlib
    import io
    import json
    import shutil
    import tarfile
    from collections import defaultdict
    from pathlib import Path
    import numpy as np
    from scipy import sparse

    BASE = Path(__file__).resolve().parent

    SOURCE = BASE / 'mgnify_matched_counts_genus_ncbi_top4096_min20'

    OUT = BASE / 'mgnify_matched_counts_genus_ncbi_top4096_min20_postorder'

    ARCHIVE = BASE / 'taxdump.tar.gz'

    def records(archive, filename):
        members = [m for m in archive.getmembers() if Path(m.name).name == filename and m.isfile()]
        if len(members) != 1:
            raise ValueError(f'Expected one {filename}, found {len(members)}')
        with archive.extractfile(members[0]) as stream:
            for line in io.TextIOWrapper(stream, encoding='utf-8'):
                yield [field.strip() for field in line.split('|')]

    def write_csv(path, fields, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def source_file(name):
        path = SOURCE / name
        backup = path.with_name(path.stem + '_abundance_order' + path.suffix)
        return backup if backup.exists() else path

    def main():
        with (source_file('features.csv')).open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            features = list(reader)
        ids = [int(row['tax_id']) for row in features]
        assert len(ids) == len(set(ids)) == 4096
        assert [int(row['column_index']) for row in features] == list(range(len(ids)))
        parent, rank = {}, {}
        with tarfile.open(ARCHIVE, 'r:gz') as archive:
            for row in records(archive, 'nodes.dmp'):
                tid = int(row[0])
                parent[tid], rank[tid] = int(row[1]), row[2]
            relevant = set()
            for tid in ids:
                if rank.get(tid) != 'genus':
                    raise ValueError(f'{tid} missing or not genus')
                path = set()
                current = tid
                while current not in relevant:
                    if current in path or current not in parent:
                        raise ValueError(f'Invalid lineage for {tid}: {current}')
                    path.add(current)
                    if current == 1:
                        if parent[current] != 1:
                            raise ValueError('Invalid root')
                        break
                    current = parent[current]
                relevant.update(path)
            names = {int(row[0]): row[1] for row in records(archive, 'names.dmp')
                     if row[3] == 'scientific name' and int(row[0]) in relevant}
        children = defaultdict(list)
        for tid in relevant:
            if tid != 1:
                children[parent[tid]].append(tid)
        for values in children.values():
            values.sort()
        sequence = []
        stack = [(1, False)]
        while stack:
            tid, visited = stack.pop()
            if visited:
                sequence.append(tid)
            else:
                stack.append((tid, True))
                stack.extend((child, False) for child in reversed(children[tid]))
        positions = {tid: i for i, tid in enumerate(sequence)}
        assert len(sequence) == len(relevant) == len(positions)
        assert all(positions[tid] < positions[parent[tid]] for tid in relevant if tid != 1)
        old_columns = {tid: i for i, tid in enumerate(ids)}
        ordered_ids = [tid for tid in sequence if tid in old_columns]
        permutation = np.array([old_columns[tid] for tid in ordered_ids], dtype=np.int64)
        inverse = np.argsort(permutation)
        original = sparse.load_npz(source_file('counts.npz'))
        ordered = original[:, permutation].tocsr()
        ordered.sort_indices()
        with np.load(source_file('counts.npz')) as z:
            assert np.array_equal(z['tax_ids'], ids)
            payload = {key: z[key].copy() for key in z.files}
        for key in ('feature_ids', 'tax_ids', 'genus_names'):
            payload[key] = payload[key][permutation]
        payload.update(data=ordered.data, indices=ordered.indices, indptr=ordered.indptr,
                       shape=np.array(ordered.shape))
        OUT.mkdir(exist_ok=True)
        np.savez_compressed(OUT / 'counts.npz', **payload)
        np.save(OUT / 'column_permutation.npy', permutation)
        np.save(OUT / 'inverse_column_permutation.npy', inverse)
        np.save(OUT / 'source_column_indices.npy', np.load(source_file('source_column_indices.npy'))[permutation])
        for name in ('biomes.csv', 'source_row_indices.npy'):
            shutil.copy2(SOURCE / name, OUT / name)
        rows = []
        for new, old in enumerate(permutation):
            row = dict(features[old])
            row['input_column_index'] = int(old)
            row['column_index'] = new
            rows.append(row)
        write_csv(OUT / 'features.csv', fields + ['input_column_index'], rows)
        write_csv(OUT / 'taxonomy_postorder.csv',
                  ['postorder_index', 'tax_id', 'parent_tax_id', 'rank', 'scientific_name', 'is_selected_genus'],
                  [dict(postorder_index=i, tax_id=tid, parent_tax_id=parent[tid], rank=rank[tid],
                        scientific_name=names.get(tid, ''), is_selected_genus=tid in old_columns)
                   for i, tid in enumerate(sequence)])
        saved = sparse.load_npz(OUT / 'counts.npz')
        assert (saved[:, inverse] != original).nnz == 0
        with np.load(OUT / 'counts.npz') as z:
            assert np.array_equal(z['tax_ids'], ordered_ids)
            assert np.array_equal(z['sample_ids'], payload['sample_ids'])
            assert np.array_equal(z['genus_names'], [row['scientific_name'] for row in rows])
        with (OUT / 'biomes.csv').open(newline='') as stream:
            assert [r['sample_id'] for r in csv.DictReader(stream)] == payload['sample_ids'].tolist()
        summary = dict(source=SOURCE.name, taxonomy_archive=str(ARCHIVE.relative_to(BASE)),
                       archive_sha256=_sha256_file(ARCHIVE),
                       traversal='post-order of selected genera plus all ancestors to NCBI root tax_id=1',
                       sibling_order='numeric tax_id ascending', tree_nodes=len(sequence),
                       selected_genera=len(ids), shape=list(saved.shape), nonzero_entries=saved.nnz,
                       verification='Every child precedes parent; all 4096 genera occur once; inverse permutation restores counts exactly; metadata and labels aligned.')
        (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    main()


def order_genera():
    """Publish the postorder matrix where the training code expects it."""
    import shutil
    base = Path(__file__).resolve().parent
    fixed = base / 'mgnify_matched_counts_genus_ncbi_top4096_min20'
    # Keep the original column mapping so the operation is repeatable.
    for name in ('counts.npz', 'features.csv', 'source_column_indices.npy'):
        path = fixed / name
        backup = path.with_name(path.stem + '_abundance_order' + path.suffix)
        if not backup.exists():
            shutil.copy2(path, backup)
    _order_genera()
    ordered = base / 'mgnify_matched_counts_genus_ncbi_top4096_min20_postorder'
    for name in ('counts.npz', 'features.csv', 'source_column_indices.npy', 'taxonomy_postorder.csv'):
        shutil.copy2(ordered / name, fixed / name)


def taxonomy_order(tax_ids, archive):
    """Return source-column indices in postorder, siblings by numeric tax_id."""
    import tarfile
    import numpy as np
    ids = np.asarray(tax_ids)
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu' or (ids <= 0).any() or len(set(ids.tolist())) != len(ids):
        raise ValueError('Expected unique positive integer tax IDs')
    with tarfile.open(archive, 'r:gz') as stream:
        matches = [m for m in stream.getmembers() if Path(m.name).name == 'nodes.dmp' and m.isfile()]
        if len(matches) != 1:
            raise ValueError('Expected exactly one nodes.dmp in taxonomy archive')
        with stream.extractfile(matches[0]) as handle:
            parent = {}
            for line in handle:
                fields = line.decode('utf-8').split('|')
                parent[int(fields[0])] = int(fields[1])
    children = {}; relevant = set()
    for taxid in ids.tolist():
        seen = set(); current = taxid
        while current not in relevant:
            if current not in parent or current in seen:
                raise ValueError(f'Missing or cyclic taxonomy lineage: {taxid}')
            seen.add(current)
            if current == 1:
                break
            current = parent[current]
        relevant.update(seen)
    if parent.get(1) != 1:
        raise ValueError('Invalid NCBI root')
    for node in relevant - {1}:
        children.setdefault(parent[node], []).append(node)
    columns = {int(t): i for i, t in enumerate(ids)}
    order = []; stack = [(1, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            if node in columns:
                order.append(columns[node])
        else:
            stack.append((node, True))
            stack.extend((child, False) for child in sorted(children.get(node, []), reverse=True))
    if len(order) != len(ids):
        raise ValueError('Taxonomy traversal lost an input tax ID')
    return np.asarray(order, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['extract', 'match', 'order'])
    args = parser.parse_args()
    {'extract': extract_taxonomy, 'match': match_genera, 'order': order_genera}[args.task]()


if __name__ == '__main__':
    main()
