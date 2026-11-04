"""Hilbert-Phy: taxonomy-ordered abundance vectors to lossless 2D arrays."""
import argparse
from pathlib import Path
import numpy as np

def xy2d(n, x, y):
    """Standard Hilbert xy2d; n must be a positive power of two."""
    if n < 1 or n & (n - 1) or not (0 <= x < n and 0 <= y < n):
        raise ValueError('Expected power-of-two n and coordinates in [0, n).')
    distance = 0
    scale = n // 2
    while scale:
        rx, ry = int(bool(x & scale)), int(bool(y & scale))
        distance += scale * scale * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x, y = scale - 1 - x, scale - 1 - y
            x, y = y, x
        scale //= 2
    return distance


def hilbert_indices(side):
    if side < 1 or side & (side - 1):
        raise ValueError('Side must be a positive power of two')
    return np.array([xy2d(side, x, y) for y in range(side) for x in range(side)], dtype=np.int64)


def encode(values, order=None):
    values = np.asarray(values)
    if values.ndim != 2 or not values.shape[1] or values.dtype.kind not in 'iuf' or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected finite nonnegative samples-by-features array')
    if order is not None:
        order = np.asarray(order)
        if order.dtype.kind not in 'iu' or not np.array_equal(np.sort(order), np.arange(values.shape[1])):
            raise ValueError('Invalid taxonomy column permutation')
        values = values[:, order]
    side = 1
    while side * side < values.shape[1]:
        side *= 2
    padded = np.zeros((len(values), side * side), dtype=values.dtype)
    padded[:, :values.shape[1]] = values
    return padded[:, hilbert_indices(side)].reshape(-1, side, side)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task', required=True)
    index = sub.add_parser('indices', help='Generate the formal 64x64 lookup arrays')
    index.add_argument('--side', type=int, default=64)
    index.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'hilbert_encoding')
    image = sub.add_parser('encode', help='Encode a dense abundance array, optionally sorting tax IDs')
    image.add_argument('--input', type=Path, required=True)
    image.add_argument('--output', type=Path, required=True)
    image.add_argument('--tax-ids', type=Path, help='Without this option input columns must already follow taxonomy postorder')
    image.add_argument('--taxonomy', type=Path, default=Path(__file__).resolve().parent / 'taxdump.tar.gz')
    args = parser.parse_args()
    if args.task == 'indices':
        h = hilbert_indices(args.side); inverse = np.argsort(h)
        assert np.array_equal(np.sort(h), np.arange(len(h)))
        coordinates = np.column_stack(np.unravel_index(inverse, (args.side, args.side)))
        assert (np.abs(np.diff(coordinates, axis=0)).sum(axis=1) == 1).all()
        args.output.mkdir(parents=True, exist_ok=True)
        np.save(args.output / 'hilbert.npy', h)
        np.save(args.output / 'hilbert_inverse.npy', inverse)
        np.save(args.output / f'hilbert_{args.side}x{args.side}.npy', h.reshape(args.side, args.side))
    else:
        values = np.load(args.input, allow_pickle=False)
        order = None
        if args.tax_ids:
            from ncbi import taxonomy_order
            ids = np.load(args.tax_ids, allow_pickle=False)
            if values.ndim != 2 or values.shape[1] != len(ids):
                parser.error('tax ID count must equal the number of abundance columns')
            order = taxonomy_order(ids, args.taxonomy)
        result = encode(values, order)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, result)
        print(f'Saved {result.shape}')


if __name__ == '__main__':
    main()
