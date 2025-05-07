import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os
SEED = 12

# Get root of data
jpeg_root = 'jpeg'  # adjust if needed
jpg_files = glob.glob(os.path.join(jpeg_root, '**', '*.jpg'), recursive=True)

uid_to_path = {}
for path in jpg_files:
    parts = path.split(os.sep)
    if len(parts) >= 2:
        uid = parts[-2]  # UID is the folder name just before the file
        uid_to_path[uid] = path

jpg_names = glob.glob('jpeg/**/*.jpg', recursive=True)
file_names = pd.Series([f.split('\\')[1] for f in jpg_names])


def create_data_split(csv_path, name):
    # Read in data
    df = pd.read_csv(csv_path)[['image file path', 'pathology']]
    
    # Pull out relevant portion of image file path
    df['image file path'] = df['image file path'].str.split('/', n = 1).str[1]
    df['sop_uid'] = df['image file path'].str.split('/', n = 2).str[1]
    df['file_path'] = "cbis\\" + df['sop_uid'].map(uid_to_path)
    df['file_path'] = df['file_path'].str.replace('\\', '/')

    # Add label field to df based on pathology designation
    df['label'] = df['pathology'].map({"MALIGNANT" : 1, "BENIGN" : 0,  "BENIGN_WITHOUT_CALLBACK" : 0,})

    # Subsample based on labels
    count_0s = len(df) - sum(df['label'])
    count_1s = min(sum(df['label']), round(.5 * count_0s))


    # print("1s: ", count_1s)
    # print("0s: ", count_0s)
    # print("Len", len(df))

    # Grab data sampled at the right level
    z = df[df['label'] == 0].sample(n=count_0s, random_state=SEED)
    o = df[df['label'] == 1]

    df_samp = pd.concat([z, o]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # Output df
    df_samp.to_csv(name, index=True)
    return df_samp

train_df = create_data_split('cbis/csv/mass_case_description_train_set.csv', 'train_split.csv')
test_df = create_data_split('cbis/csv/mass_case_description_test_set.csv', 'test_split.csv')
