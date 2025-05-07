import pandas as pd
from sklearn.model_selection import train_test_split
SEED = 12

df = pd.read_csv("melanoma/train.csv")[['image_name', 'target']]

# Processing 
df['image_name'] = 'melanoma/data/' + df['image_name'].astype(str) + '.dcm'

# Split into benign and malignant
benign_df = df[df['target'] == 0]
malignant_df = df[df['target'] == 1]

# Determine the limiting factor (minority class)
# We want 2 benign for every 1 malignant in both sets

malignant_len =  len(malignant_df)
benign_len = min(len(benign_df), 2 * malignant_len)

print(f"Malignants: {malignant_len}")
print(f"Benigns: {benign_len}")

benign_df = benign_df.sample(n=benign_len, random_state=SEED)
malignant_df = malignant_df.sample(n=malignant_len, random_state=SEED)


# Split each class into train/test (based on splits used in original paper)
benign_train, benign_test = train_test_split(benign_df, test_size=0.162, random_state=SEED)
malignant_train, malignant_test = train_test_split(malignant_df, test_size=0.162, random_state=SEED)


# Combine train/test and shuffle
train_df = pd.concat([benign_train, malignant_train]).sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df = pd.concat([benign_test, malignant_test]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Update Column names
train_df.columns = ['file_path', 'label']
test_df.columns = ['file_path', 'label']

print(f"Train: {len(train_df)}")
print(f"Test: {len(test_df)}")

# Save to CSV
train_df.to_csv('melanoma/train_split.csv', index=True)
test_df.to_csv('melanoma/test_split.csv', index=True)
print("Saved train/test CSVs")