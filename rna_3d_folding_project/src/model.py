import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, Dropout
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split

# Import our custom data processing module
from data_processing import (
    load_data, encode_sequence, process_labels, create_dataset, 
    pad_data, nucleotide_map, normalize_coordinates, visualize_structure,
    split_dataset, create_augmented_dataset, evaluate_predictions
)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Load training data


# Define the base directory and data paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gets the parent directory of src
data_dir = os.path.join(base_dir, 'data', 'raw')

# Load training data
print("Loading training data...")
train_sequences = pd.read_csv(os.path.join(data_dir, 'train_sequences.csv'))
train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))

# Load validation data
print("Loading validation data...")
valid_sequences = pd.read_csv(os.path.join(data_dir, 'validation_sequences.csv'))
valid_labels = pd.read_csv(os.path.join(data_dir, 'validation_labels.csv'))

# Load test data
print("Loading test data...")
test_sequences = pd.read_csv(os.path.join(data_dir, 'test_sequences.csv'))


print(f"Loaded {len(train_sequences)} training sequences, {len(valid_sequences)} validation sequences, and {len(test_sequences)} test sequences")

# Encode sequences
print("Encoding sequences...")
train_sequences['encoded'] = train_sequences['sequence'].apply(encode_sequence)
valid_sequences['encoded'] = valid_sequences['sequence'].apply(encode_sequence)
test_sequences['encoded'] = test_sequences['sequence'].apply(encode_sequence)

# Process labels
print("Processing labels...")
train_labels_dict = process_labels(train_labels)
valid_labels_dict = process_labels(valid_labels)

# Create datasets
print("Creating datasets...")
X_train, y_train, train_ids = create_dataset(train_sequences, train_labels_dict)
X_valid, y_valid, valid_ids = create_dataset(valid_sequences, valid_labels_dict)

# Optional: Create augmented dataset
print("Creating augmented training dataset...")
X_train_aug, y_train_aug, train_ids_aug = create_augmented_dataset(X_train, y_train, train_ids, augmentation_factor=1)  # No extra augmentation initially

# Pad data
print("Padding data...")
X_train_pad, y_train_pad, max_len = pad_data(X_train_aug, y_train_aug)
X_valid_pad, y_valid_pad, _ = pad_data(X_valid, y_valid, max_len=max_len)

# Optional: Normalize coordinates
print("Normalizing coordinates...")
y_train_norm = np.array([normalize_coordinates(coords) for coords in y_train_pad])
y_valid_norm = np.array([normalize_coordinates(coords) for coords in y_valid_pad])

# Check for NaN values
print("Any NaN in y_train_norm?", np.isnan(y_train_norm).any())
print("Any NaN in y_valid_norm?", np.isnan(y_valid_norm).any())

# Replace any NaN values if they exist
if np.isnan(y_train_norm).any():
    y_train_norm = np.nan_to_num(y_train_norm)
if np.isnan(y_valid_norm).any():
    y_valid_norm = np.nan_to_num(y_valid_norm)

# Define hyperparameters
vocab_size = max(nucleotide_map.values()) + 1  # +1 for padding token 0
embedding_dim = 32  # Increased from 16
num_filters = 128  # Increased from 64
kernel_size = 3
drop_rate = 0.3  # Increased from 0.2
lstm_units = 64

# Build an improved CNN-LSTM hybrid model
print("Building model...")
input_seq = Input(shape=(max_len,), name='input_seq')
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name='embedding')(input_seq)

# First Conv Block
x = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation='relu', name='conv1')(x)
x = BatchNormalization(name='bn1')(x)
x = Dropout(drop_rate, name='drop1')(x)

# Second Conv Block
x = Conv1D(filters=num_filters*2, kernel_size=kernel_size, padding='same', activation='relu', name='conv2')(x)
x = BatchNormalization(name='bn2')(x)
x = Dropout(drop_rate, name='drop2')(x)

# Bidirectional LSTM layer to capture sequence dependencies
x = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(x)
x = BatchNormalization(name='bn3')(x)
x = Dropout(drop_rate, name='drop3')(x)

# Third Conv Block for final prediction
output_coords = Conv1D(filters=3, kernel_size=1, padding='same', activation='linear', name='predicted_coords')(x)

# Create the model
rna_model = Model(inputs=input_seq, outputs=output_coords)

# Compile with a lower learning rate
optimizer = Adam(learning_rate=0.001)
rna_model.compile(optimizer=optimizer, loss='mse')
rna_model.summary()

# Define callbacks for training
print("Setting up training callbacks...")
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Increased from 5
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'models/rna_model_checkpoint.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train the model
print("Training model...")
history = rna_model.fit(
    X_train_pad, y_train_norm,
    validation_data=(X_valid_pad, y_valid_norm),
    epochs=100,  # Increased from 50
    batch_size=32,  # Increased from 16
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("RNA 3D Folding Model Training vs. Validation Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('plots/training_history.png')
plt.show()

# Save the model
print("Saving model...")
rna_model.save('models/rna_model.h5')
print("Model saved to 'models/rna_model.h5'")

# Generate predictions on validation set
print("Generating validation predictions...")
val_preds = rna_model.predict(X_valid_pad)

# Evaluate predictions
print("Evaluating predictions...")
eval_results = evaluate_predictions(y_valid_norm, val_preds, valid_ids)
print(f"Overall RMSD: {eval_results['overall']}")

# Visualize a few examples
print("Visualizing examples...")
for i in range(min(3, len(valid_ids))):
    fig, ax = visualize_structure(y_valid_norm[i], target_id=valid_ids[i])
    plt.title(f"True Structure: {valid_ids[i]}")
    plt.savefig(f"plots/true_structure_{valid_ids[i]}.png")
    plt.close()
    
    fig, ax = visualize_structure(val_preds[i], target_id=valid_ids[i])
    plt.title(f"Predicted Structure: {valid_ids[i]}")
    plt.savefig(f"plots/pred_structure_{valid_ids[i]}.png")
    plt.close()

# Prepare test data for submission
print("Preparing test data for submission...")
X_test_encoded = test_sequences['encoded'].tolist()
X_test_pad = pad_data(X_test_encoded, max_len=max_len)

# Generate predictions on test set
print("Generating test predictions...")
test_preds = rna_model.predict(X_test_pad)

# Create submission file
print("Creating submission file...")
# This part depends on the specific format required by the competition
# You might need to adjust this based on the competition requirements

print("Done! Your RNA 3D folding model has been trained and evaluated.")