import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Nucleotide mapping for encoding RNA sequences
nucleotide_map = {'A': 1, 'C': 2, 'G': 3, 'U': 4}

def load_data(seq_path, labels_path=None):
    """Load sequence and label data from CSV files."""
    try:
        sequences = pd.read_csv(seq_path)
        if labels_path:
            labels = pd.read_csv(labels_path)
            labels.fillna(0, inplace=True)  # Fill missing values with 0
            return sequences, labels
        return sequences
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find file: {e.filename}")

def encode_sequence(seq):
    """Encode an RNA sequence into integers using nucleotide_map."""
    if not isinstance(seq, str):
        raise TypeError("Sequence must be a string")
    return [nucleotide_map.get(ch, 0) for ch in seq]

def process_labels(labels_df):
    """
    Process labels into a dictionary mapping target_id to coordinate arrays.
    Uses more efficient vectorized operations where possible.
    """
    label_dict = {}
    
    for target_id, group in labels_df.groupby(labels_df['ID'].str.rsplit('_', n=1).str[0]):
        resids = group['ID'].str.rsplit('_', n=1).str[1].astype(int)
        coords = group[['x_1', 'y_1', 'z_1']].values.astype(np.float32)
        idx_sort = np.argsort(resids)
        label_dict[target_id] = coords[idx_sort]
    return label_dict

def create_dataset(sequences_df, labels_dict):
    """Create a dataset of encoded sequences and corresponding coordinates."""
    X, y, target_ids = [], [], []
    for idx, row in sequences_df.iterrows():
        tid = row['target_id']
        if tid in labels_dict:
            X.append(row['encoded'])
            y.append(labels_dict[tid])
            target_ids.append(tid)
    return X, y, target_ids

def augment_sequence(encoded_seq):
    """Create simple augmentations of RNA sequences"""
    # No augmentation for 50% of cases
    if np.random.random() > 0.5:
        return encoded_seq
        
    # Randomly replace a small percentage of nucleotides
    aug_seq = encoded_seq.copy()
    mask = np.random.random(len(aug_seq)) < 0.05  # 5% mutation rate
    for i in range(len(aug_seq)):
        if mask[i] and aug_seq[i] != 0:  # Don't replace padding
            aug_seq[i] = np.random.choice([1, 2, 3, 4])
    return aug_seq

def normalize_coordinates(coords):
    """Normalize coordinates to zero mean and unit variance."""
    # Only normalize non-zero coordinates
    mask = (coords != 0).any(axis=1)
    if mask.sum() > 0:
        mean = coords[mask].mean(axis=0)
        std = coords[mask].std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        normalized = coords.copy()
        normalized[mask] = (coords[mask] - mean) / std
        return normalized
    return coords

def pad_coordinates(coord_array, max_len):
    """Pad coordinate arrays to match the maximum sequence length."""
    L = coord_array.shape[0]
    if L < max_len:
        pad_width = ((0, max_len - L), (0, 0))
        return np.pad(coord_array, pad_width, mode='constant', constant_values=0)
    return coord_array

def pad_data(X, y=None, max_len=None):
    """Pad sequences and optionally coordinates to a specified length."""
    if max_len is None:
        max_len = max(len(seq) for seq in X)
    X_pad = pad_sequences(X, maxlen=max_len, padding='post', value=0)
    if y is not None:
        y_pad = np.array([pad_coordinates(arr, max_len) for arr in y])
        return X_pad, y_pad, max_len
    return X_pad

def split_dataset(X, y, target_ids, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split data into train/val/test sets."""
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    test_size = int(len(indices) * test_ratio)
    val_size = int(len(indices) * val_ratio)
    
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size+val_size]
    train_indices = indices[test_size+val_size:]
    
    splits = {
        'train': (np.array(X)[train_indices], np.array(y)[train_indices], np.array(target_ids)[train_indices]),
        'val': (np.array(X)[val_indices], np.array(y)[val_indices], np.array(target_ids)[val_indices]),
        'test': (np.array(X)[test_indices], np.array(y)[test_indices], np.array(target_ids)[test_indices])
    }
    return splits

def visualize_structure(coords, target_id=None, save_path=None):
    """Visualize RNA 3D structure using matplotlib."""
    # Filter out zero padding
    mask = (coords != 0).any(axis=1)
    valid_coords = coords[mask]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2], 'o-', linewidth=2)
    ax.set_title(f"RNA 3D Structure {target_id}" if target_id else "RNA 3D Structure")
    
    if save_path:
        plt.savefig(save_path)
    return fig, ax

def create_augmented_dataset(X, y, target_ids, augmentation_factor=2):
    """Create augmented dataset by applying augmentation to sequences."""
    X_aug, y_aug, target_ids_aug = [], [], []
    
    for i in range(len(X)):
        # Always include the original sequence
        X_aug.append(X[i])
        y_aug.append(y[i])
        target_ids_aug.append(target_ids[i])
        
        # Add augmented versions
        for j in range(augmentation_factor - 1):
            X_aug.append(augment_sequence(X[i]))
            y_aug.append(y[i])
            target_ids_aug.append(target_ids[i])
    
    return X_aug, y_aug, target_ids_aug

def evaluate_predictions(y_true, y_pred, target_ids=None):
    """
    Calculate RMSD (Root Mean Square Deviation) for each structure.
    Lower RMSD values indicate better predictions.
    """
    results = {}
    
    for i in range(len(y_true)):
        # Filter out padding (zeros)
        mask = (y_true[i] != 0).any(axis=1)
        true_coords = y_true[i][mask]
        pred_coords = y_pred[i][mask]
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((true_coords - pred_coords)**2, axis=1)))
        
        tid = target_ids[i] if target_ids is not None else i
        results[tid] = rmsd
    
    # Overall RMSD
    overall_rmsd = np.mean(list(results.values()))
    results['overall'] = overall_rmsd
    
    return results