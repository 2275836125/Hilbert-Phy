"""Read-only NCBI taxonomy, strict identity audit and deterministic tree layout."""
import csv
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXED = ROOT / 'data_processing/mgnify_matched_counts_genus_ncbi_top4096_min20'
SOURCE = ROOT / 'data_processing/mgnify_matched_counts_genus_ncbi/counts.npz'
TAXDUMP = ROOT / 'data_processing/taxdump.tar.gz'
RULES = {'root': 1, 'edge_length': 1, 'pruning': 'retain all ancestors; no contraction',
         'siblings': 'minimum fixed descendant column, then tax_id',
         'matrix': 'breadth-first rows; children in parent order packed left; zero padding'}
PCA_CONFIG = {'n_components': 32, 'svd_solver': 'randomized', 'random_state': 0,
              'iterated_power': 7, 'n_oversamples': 10, 'whiten': False,
              'adaptation_parameter': True}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def audit_tree(report_path):
    features = list(csv.DictReader((FIXED/'features.csv').open(newline='')))
    ids = [int(r['tax_id']) for r in features]
    issues = {'duplicates': [i for i,n in Counter(ids).items() if n != 1],
              'column_order': [], 'missing': [], 'merged': [], 'deleted': [],
              'not_genus': [], 'broken_paths': [], 'array_mapping': []}
    for j,r in enumerate(features):
        if int(r['column_index']) != j: issues['column_order'].append(r)
    parents = {}; ranks = {}; merged = {}; deleted = set()
    with tarfile.open(TAXDUMP, 'r:gz') as t:
        for line in t.extractfile('nodes.dmp'):
            v = line.decode().split('|'); parents[int(v[0])] = int(v[1]); ranks[int(v[0])] = v[2].strip()
        for line in t.extractfile('merged.dmp'):
            v = line.decode().split('|'); merged[int(v[0])] = int(v[1])
        for line in t.extractfile('delnodes.dmp'): deleted.add(int(line.split(b'|')[0]))
    paths = []
    for i in ids:
        if i not in parents: issues['missing'].append(i)
        if i in merged: issues['merged'].append({'tax_id':i,'merged_into':merged[i]})
        if i in deleted: issues['deleted'].append(i)
        if i in ranks and ranks[i] != 'genus': issues['not_genus'].append([i,ranks[i]])
        path = []; n = i
        while True:
            if n in path or n not in parents:
                issues['broken_paths'].append({'tax_id':i,'path':path,'failed_at':n}); break
            path.append(n)
            if n == 1: break
            n = parents[n]
        paths.append(path)
    postorder = list(csv.DictReader((FIXED/'taxonomy_postorder.csv').open(newline='')))
    ordered = [int(r['tax_id']) for r in postorder if r['is_selected_genus'].casefold() == 'true']
    if ordered != ids: issues['array_mapping'].append('taxonomy_postorder_selected_order')
    cols = np.load(FIXED/'source_column_indices.npy')
    rows = np.load(FIXED/'source_row_indices.npy')
    with np.load(SOURCE) as z, np.load(FIXED/'counts.npz') as f:
        for label, ok in {
            'fixed_tax_ids': np.array_equal(f['tax_ids'], ids),
            'source_tax_ids': np.array_equal(z['tax_ids'][cols], ids),
            'feature_source_columns': np.array_equal(cols,[int(r['source_column_index']) for r in features]),
            'source_sample_ids': np.array_equal(z['sample_ids'][rows],f['sample_ids']),
            'fixed_shape': np.array_equal(f['shape'],[238419,4096]),
            'source_genera': int(z['shape'][1]) == 10437,
        }.items():
            if not ok: issues['array_mapping'].append(label)
    if len(ids) != 4096: issues['array_mapping'].append('feature_count')
    report = {'columns':len(ids),'samples':len(rows),'issues':issues,
              'taxdump_sha256':sha(TAXDUMP),'rules':RULES,
              'column_mapping':[{'column':j,'tax_id':i,'path_to_root':p} for j,(i,p) in enumerate(zip(ids,paths))]}
    write_json(report_path, report)
    if any(issues.values()): raise ValueError(f'Taxonomy/mapping audit failed; inspect {report_path}')
    # A selected genus must remain a terminal node of the induced tree.
    selected = set(ids)
    ancestors = [(i,a) for i,p in zip(ids,paths) for a in p[1:] if a in selected]
    if ancestors:
        report['issues']['selected_ancestor_pairs'] = ancestors
        write_json(report_path,report); raise ValueError('Selected genus is ancestor of another selected genus')
    children = defaultdict(list); minimum = {}
    for j,p in enumerate(paths):
        for n in p: minimum[n] = min(j,minimum.get(n,j))
    for n in minimum:
        if n != 1: children[parents[n]].append(n)
    for n in children: children[n].sort(key=lambda x:(minimum[x],x))
    levels = [[1]]
    while True:
        row = [c for n in levels[-1] for c in children[n]]
        if not row: break
        levels.append(row)
    nodes = [n for row in levels for n in row]
    positions = {n:(r,c) for r,row in enumerate(levels) for c,n in enumerate(row)}
    report.update(nodes=len(nodes), matrix_shape=[len(levels),max(map(len,levels))],
                  positions=[{'tax_id':n,'row':positions[n][0],'column':positions[n][1]} for n in nodes])
    report['tree_signature'] = digest({'mapping':ids,'paths':paths,'rules':RULES,'taxdump':report['taxdump_sha256']})
    write_json(report_path,report)
    return ids, paths, nodes, positions, report


def path_distance(a, b):
    """Distance in original unit edges; paths are ordered leaf to root."""
    shared = 0
    for x,y in zip(reversed(a),reversed(b)):
        if x != y: break
        shared += 1
    return len(a)+len(b)-2*shared
