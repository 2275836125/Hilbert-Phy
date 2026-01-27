# Hilbert-Phy - A Two-Dimensional Encoding Method for Microbiome Data Analysis

## Overview
This tool maps taxonomic abundance data to a Hilbert-curve based representation, which can be output as either 2D images or 1D sequences. The mapping preserves the phylogenetic relationships between taxa by ordering them according to the taxonomic tree structure.

<img src="https://github.com/2275836125/Hilbert-Phy/blob/main/framework.png" style="zoom:150%;" />

## Dependencies

All required dependencies are listed in the `requirements.txt` file. You can install them with:

```bash
pip install -r requirements.txt
```

## Notes
1. The tool automatically downloads NCBI taxonomy database on first run (via ete3)
2. Output dimensions are determined by the number of taxa (padded to nearest power of 4)
3. Missing taxa in the abundance profile will be represented as zeros in the output
4. The Hilbert curve order preserves some spatial locality of phylogenetically related taxa


## CLI Usage Guide

### Examples

#### Basic usage (1D output):
```bash
python hilbert_mapping.py \
  --tax_id tax_ids.npy \
  --abu_profile abundance.csv \
  --output hilbert_sequences.npy
```

#### With 2D image output and verbose logging:
```bash
python hilbert_mapping.py \
  --tax_id tax_ids.npy \
  --abu_profile abundance.npy \
  --output hilbert_images.npy \
  --output_2d \
  --verbose
```

#### Using CSV abundance data:
```bash
python hilbert_mapping.py \
  --tax_id /path/to/tax_ids.npy \
  --abu_profile /path/to/data.csv \
  --output /path/to/output/hilbert.npy
```

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--tax_id` | string (file path) | **Required.** Path to the tax_id list file. Must be in `.npy` format containing an array of taxonomic IDs (integers). |
| `--abu_profile` | string (file path) | **Required.** Path to the abundance data file. Can be either `.npy` (NumPy array) or `.csv` (CSV file) format. Data should be in samples × tax_ids format. |
| `--output` | string (file path) | **Required.** Output file path where the Hilbert-mapped data will be saved. Output is saved in `.npy` format. |

### Optional Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--output_2d` | flag | When specified, outputs 2D Hilbert images. When omitted, outputs 1D Hilbert sequences. Default: False (outputs 1D sequences). |
| `--verbose` | flag | When specified, prints detailed progress information and statistics during processing. |

### File Format Specifications

#### Input Files
1. **tax_id file** (`.npy`):
   - NumPy array of taxonomic IDs (integers)
   - Shape: `(n_taxa,)` where n_taxa is the number of taxonomic features
   - Must correspond to the columns in the abundance profile

2. **abu_profile file** (`.npy` or `.csv`):
   - **`.npy`**: NumPy array with shape `(n_samples, n_taxa)`
   - **`.csv`**: CSV file where rows are samples and columns are taxa
   - Values should represent abundance/read counts for each taxon

#### Output File
- **`.npy`**: NumPy array containing Hilbert-mapped data
  - If `--output_2d`: Shape is `(n_samples, n, n)` where n is determined by padding to nearest power of 4
  - If not `--output_2d`: Shape is `(n_samples, n*n)` (flattened Hilbert order)

### Processing Steps
1. **Taxonomic tree extraction**: Uses NCBI taxonomy database to get phylogenetic relationships
2. **Leaf node ordering**: Reorders taxa according to taxonomic tree leaf traversal
3. **Padding**: Pads to nearest power of 4 for Hilbert curve generation
4. **Hilbert mapping**: Maps ordered taxa to Hilbert curve coordinates
5. **Data placement**: Places abundance values at corresponding Hilbert coordinates
6. **Output**: Saves resulting Hilbert representation
