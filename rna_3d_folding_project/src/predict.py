import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from data_processing import load_data, encode_sequence, pad_data

# Load test sequences
test_sequences = load_data('data/raw/test_sequences.csv')

# Encode test sequences
test_sequences['encoded'] = test_sequences['sequence'].apply(encode_sequence)

# Load the trained model
cnn_model = load_model('models/rna_model.h5')

# Get max_len from the model's input shape
max_len = cnn_model.input_shape[1]

# Pad test sequences
X_test_pad = pad_data(test_sequences['encoded'].tolist(), max_len=max_len)

# Make predictions
predictions = cnn_model.predict(X_test_pad)
print("Predictions shape:", predictions.shape)

# Create submission file
submission_rows = []
for idx, row in test_sequences.iterrows():
    target_id = row['target_id']
    pred_coords = predictions[idx][:len(row['encoded']), :]  # Trim to actual sequence length
    for i in range(len(row['encoded'])):
        coords = pred_coords[i]
        # Replicate coordinates 5 times per competition format
        submission_rows.append({
            'ID': f"{target_id}_{i+1}",
            'resname': row['sequence'][i],
            'resid': i+1,
            **{f"x_{j+1}": coords[0] for j in range(5)},
            **{f"y_{j+1}": coords[1] for j in range(5)},
            **{f"z_{j+1}": coords[2] for j in range(5)}
        })

submission_df = pd.DataFrame(submission_rows)
print("Submission DataFrame shape:", submission_df.shape)
print(submission_df.head(10))

# Save submission
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")