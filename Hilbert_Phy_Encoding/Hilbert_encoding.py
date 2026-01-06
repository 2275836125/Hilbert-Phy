import argparse
import numpy as np
import pandas as pd
from ete3 import NCBITaxa
import Hilbert
import os

def map_to_hilbert(abu_profile, temp, output_2d=True, verbose=True):
    """
    Map abundance data to Hilbert space
    
    Parameters:
    - abu_profile: Abundance data matrix (samples x tax_ids)
    - temp: List of tax_ids, should correspond to columns of abu_profile
    - output_2d: Whether to output 2D Hilbert images (True) or 1D sequences (False)
    - verbose: Whether to print detailed information
    
    Returns:
    - If output_2d=True: 3D array (samples, n, n) Hilbert images
    - If output_2d=False: 2D array (samples, n*n) 1D sequences
    - Tuple information: (hilbert_images, mapping_info)
    """
    
    if verbose:
        print("="*60)
        print("Starting Hilbert space mapping")
        print("="*60)
    
    # 1. Get leaf node order from taxonomic tree
    if verbose:
        print("1. Getting taxonomic tree leaf nodes...")
    
    ncbi = NCBITaxa()
    tax_tree = ncbi.get_topology(temp)
    leaf_nodes = tax_tree.get_leaves()
    leaf_list = [leaf.name for leaf in leaf_nodes]
    
    if verbose:
        print(f"   Original tax_id count: {len(temp)}")
        print(f"   Leaf node count: {len(leaf_list)}")
    
    # 2. Reorder abundance data according to leaf node order
    if verbose:
        print("\n2. Reordering data according to taxonomic tree leaf nodes...")
    
    # Ensure abu_profile is a 2D array
    if isinstance(abu_profile, pd.DataFrame):
        abu_array = abu_profile.values
    else:
        abu_array = abu_profile
    
    # Check dimensions
    if abu_array.shape[1] != len(temp):
        raise ValueError(f"Abundance data columns ({abu_array.shape[1]}) do not match tax_id count ({len(temp)})")
    
    # Create DataFrame and reorder columns
    leaf_list = np.array(leaf_list).astype(np.int32)
    df = pd.DataFrame(abu_array, columns=temp)
    df_reordered = df[leaf_list]  # Reorder columns according to leaf_list
    
    if verbose:
        print(f"   Reordered data shape: {df_reordered.shape}")
    
    # 3. Pad to nearest power of 4
    if verbose:
        print("\n3. Padding to nearest power of 4...")
    
    k_n = [np.power(4, i) for i in range(10)]
    current_len = len(leaf_list)
    
    # Find the smallest power of 4 that is greater than or equal to current length
    target_len = None
    target_index = None
    for i, k in enumerate(k_n):
        if k >= current_len:
            target_len = k
            target_index = i
            break
    
    if target_len is None:
        target_len = k_n[-1]
        target_index = len(k_n) - 1
        if verbose:
            print(f"   Warning: Length {current_len} exceeds maximum k_n ({target_len})")
    
    # Padding
    padding_size = target_len - current_len
    if padding_size > 0:
        padded_leaf_list = np.pad(leaf_list, [0, padding_size], mode='constant', constant_values=0)
        if verbose:
            print(f"   Padding {padding_size} zeros, from {current_len} to {target_len}")
    else:
        padded_leaf_list = leaf_list[:target_len]
        if verbose:
            print(f"   Truncating to length {target_len}")
    
    # 4. Generate Hilbert curve mapping
    if verbose:
        print("\n4. Generating Hilbert curve mapping...")
    
    hilbert_obj = Hilbert.Hilbert()
    n = 2 ** target_index
    
    if verbose:
        print(f"   Hilbert grid size: {n}x{n} = {n*n}")
    
    # Generate Hilbert distance matrix
    hilbert_map = []
    for i in range(n):
        for j in range(n):
            hilbert_dist = hilbert_obj.xy2d(n, Hilbert.Point(j, i))
            hilbert_map.append(hilbert_dist)
    
    hilbert_map = np.array(hilbert_map).reshape([n, n])
    
    if verbose:
        print(f"   Hilbert mapping matrix shape: {hilbert_map.shape}")
    
    # 5. Create mapping relationships
    if verbose:
        print("\n5. Creating mapping relationships...")
    
    # 5.1 Create mapping from tax_id to position in padded_leaf_list
    taxid_to_position = {}
    for idx, tax_id in enumerate(padded_leaf_list):
        if tax_id != 0:
            taxid_to_position[tax_id] = idx
    
    # 5.2 Create mapping from Hilbert distance to coordinates
    hilbert_to_coord = {}
    for i in range(n):
        for j in range(n):
            hilbert_dist = hilbert_map[i, j]
            hilbert_to_coord[hilbert_dist] = (i, j)
    
    # 5.3 Create mapping from position to Hilbert distance
    position_to_hilbert = {}
    for hilbert_dist, (i, j) in hilbert_to_coord.items():
        # Calculate position in 1D array
        position = i * n + j
        position_to_hilbert[position] = hilbert_dist
    
    # 6. Map data to Hilbert space
    if verbose:
        print("\n6. Mapping data to Hilbert space...")
    
    num_samples = df_reordered.shape[0]
    
    if output_2d:
        # Output 2D Hilbert images
        hilbert_images = np.zeros((num_samples, n, n))
        
        for sample_idx in range(num_samples):
            if verbose and sample_idx % 100 == 0 and sample_idx > 0:
                print(f"   Processing sample {sample_idx}/{num_samples}")
            
            # Iterate over each tax_id
            for col_idx, tax_id in enumerate(leaf_list):
                # Get position of tax_id in padded_leaf_list
                if tax_id in taxid_to_position:
                    position = taxid_to_position[tax_id]
                    
                    # Get Hilbert distance for this position
                    if position in position_to_hilbert:
                        hilbert_dist = position_to_hilbert[position]
                        
                        # Get coordinates corresponding to Hilbert distance
                        if hilbert_dist in hilbert_to_coord:
                            i, j = hilbert_to_coord[hilbert_dist]
                            
                            # Place abundance value in Hilbert image
                            hilbert_images[sample_idx, i, j] = df_reordered.iloc[sample_idx, col_idx]
    
        result = hilbert_images
        if verbose:
            print(f"   Generated Hilbert image shape: {hilbert_images.shape}")
    else:
        # Output 1D Hilbert sequences
        hilbert_sequences = np.zeros((num_samples, n * n))
        
        # Reorder according to Hilbert order
        # First get list of positions sorted by Hilbert distance
        hilbert_ordered_positions = []
        for hilbert_dist in sorted(position_to_hilbert.values()):
            # Find position corresponding to this hilbert_dist
            for pos, dist in position_to_hilbert.items():
                if dist == hilbert_dist:
                    hilbert_ordered_positions.append(pos)
                    break
        
        # Create mapping from padded_leaf_list position to 1D sequence position
        pos_to_seq_idx = {pos: idx for idx, pos in enumerate(hilbert_ordered_positions)}
        
        for sample_idx in range(num_samples):
            if verbose and sample_idx % 100 == 0 and sample_idx > 0:
                print(f"   Processing sample {sample_idx}/{num_samples}")
            
            # Iterate over each tax_id
            for col_idx, tax_id in enumerate(leaf_list):
                if tax_id in taxid_to_position:
                    position = taxid_to_position[tax_id]
                    
                    if position in pos_to_seq_idx:
                        seq_idx = pos_to_seq_idx[position]
                        hilbert_sequences[sample_idx, seq_idx] = df_reordered.iloc[sample_idx, col_idx]
        
        result = hilbert_sequences
        if verbose:
            print(f"   Generated Hilbert sequence shape: {hilbert_sequences.shape}")
    
    # 7. Prepare return information
    mapping_info = {
        'leaf_list': leaf_list,
        'padded_leaf_list': padded_leaf_list,
        'target_len': target_len,
        'n': n,
        'hilbert_map': hilbert_map,
        'taxid_to_position': taxid_to_position,
        'hilbert_to_coord': hilbert_to_coord,
        'df_reordered': df_reordered
    }
    
    if verbose:
        print("\n" + "="*60)
        print("Hilbert space mapping completed!")
        print("="*60)
        
        # Print statistics
        print("\nStatistics:")
        print(f"  Sample count: {num_samples}")
        print(f"  Original features: {len(temp)}")
        print(f"  Leaf nodes: {len(leaf_list)}")
        print(f"  Padded length: {len(padded_leaf_list)}")
        print(f"  Hilbert grid: {n}x{n}")
        
        if output_2d:
            print(f"  Output shape: {result.shape}")
            # Calculate sparsity
            non_zero = np.count_nonzero(result[0])
            total = result[0].size
            print(f"  Sparsity: {1 - non_zero/total:.2%} ({non_zero}/{total} non-zero values)")
    
    return result, mapping_info


def main():
    parser = argparse.ArgumentParser(description="Map abundance data to Hilbert space")
    parser.add_argument("--tax_id", type=str, required=True,
                        help="Path to tax_id list file (npy format)")
    parser.add_argument("--abu_profile", type=str, required=True,
                        help="Path to abundance data file (npy or csv format)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path (npy format)")
    parser.add_argument("--output_2d", action="store_true",
                        help="Whether to output 2D Hilbert images (default outputs 1D sequences)")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to print detailed information")

    args = parser.parse_args()

    # 1. Load tax_id
    temp = np.load(args.tax_id).astype(np.int32)

    # 2. Load abundance data
    if args.abu_profile.endswith(".npy"):
        abu_profile = np.load(args.abu_profile)
    elif args.abu_profile.endswith(".csv"):
        abu_profile = pd.read_csv(args.abu_profile).values
    else:
        raise ValueError("abu_profile file must be in .npy or .csv format")

    # 3. Map to Hilbert space
    result, mapping_info = map_to_hilbert(
        abu_profile,
        temp,
        output_2d=args.output_2d,
        verbose=args.verbose
    )

    # 4. Save results
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(args.output, result)

    if args.verbose:
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()