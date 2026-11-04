"""Minimal NCBI taxonomy postorder + Hilbert encoder."""
import argparse
from pathlib import Path

import numpy as np
from ete3 import NCBITaxa


def hilbert_positions(side):
    """Return row-major pixel indices in Hilbert traversal order."""
    positions = []
    for distance in range(side * side):
        x = y = 0
        remaining = distance
        scale = 1
        while scale < side:
            rx = (remaining // 2) & 1
            ry = (remaining ^ rx) & 1
            if ry == 0:
                if rx == 1:
                    x, y = scale - 1 - x, scale - 1 - y
                x, y = y, x
            x += scale * rx
            y += scale * ry
            remaining //= 4
            scale *= 2
        positions.append(y * side + x)
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tax-ids', type=Path, required=True)
    parser.add_argument('--abundance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dbfile', type=Path, help='Existing ete3 taxa.sqlite; default: ~/.etetoolkit/taxa.sqlite')
    args = parser.parse_args()
    ids = np.load(args.tax_ids, allow_pickle=False)
    values = np.load(args.abundance, allow_pickle=False)
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu':
        parser.error('tax IDs must be a nonempty 1D integer array')
    if len(set(ids.tolist())) != len(ids) or np.any(ids <= 0):
        parser.error('tax IDs must be unique and positive')
    if values.ndim != 2 or values.shape[1] != len(ids):
        parser.error('abundance must have shape (samples, number of tax IDs)')
    if values.dtype.kind not in 'iuf' or not np.isfinite(values).all() or np.any(values < 0):
        parser.error('abundance must contain finite, nonnegative numeric values')
    if args.dbfile is not None and not args.dbfile.is_file():
        parser.error('--dbfile must point to an existing ete3 database')
    ncbi = NCBITaxa(dbfile=str(args.dbfile) if args.dbfile else None)
    missing = set(ids.tolist()) - set(ncbi.get_taxid_translator(ids.tolist()))
    if missing:
        parser.error(f'tax IDs missing from database: {sorted(missing)}')
    tree = ncbi.get_topology(ids.tolist(), intermediate_nodes=True)
    # Stable sibling ordering within the selected database version.
    for node in tree.traverse():
        node.children.sort(key=lambda child: int(child.name))
    columns = {int(taxid): index for index, taxid in enumerate(ids)}
    order = [columns[int(node.name)] for node in tree.traverse('postorder')
             if int(node.name) in columns]
    if len(order) != len(ids):
        raise RuntimeError('Taxonomy traversal did not retain every input tax ID')
    side = 1
    while side * side < len(ids):
        side *= 2
    flat = np.zeros((len(values), side * side), dtype=values.dtype)
    flat[:, hilbert_positions(side)[:len(ids)]] = values[:, order]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, flat.reshape(len(values), side, side))
    print(f'Saved output with shape {(len(values), side, side)}')


if __name__ == '__main__':
    main()
