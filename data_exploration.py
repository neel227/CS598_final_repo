import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set plot style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
base_save_path = './results/characteristics'
os.makedirs(base_save_path, exist_ok=True)

ext = 'pdf'

cbis_path = 'cbis/csv/'
# === CBIS-DDSM ===
cbis_mass = pd.read_csv(cbis_path + 'mass_case_description_train_set.csv')
cbis_calc = pd.read_csv(cbis_path + 'calc_case_description_train_set.csv')


print(cbis_calc['pathology'].unique())

# Sync Schema
cbis_mass = cbis_mass.rename(columns={'breast_density': 'breast density'})

cbis_mass['type'] = 'mass'
cbis_calc['type'] = 'calcification'

cbis_mass['calc type'] = np.nan
cbis_mass['calc distribution'] = np.nan
cbis_calc['mass shape'] = np.nan
cbis_calc['mass margins'] = np.nan

cbis_calc = cbis_calc[['patient_id', 'breast density', 'left or right breast', 'image view',
       'abnormality id', 'abnormality type', 'mass shape', 'mass margins',
       'assessment', 'pathology', 'subtlety', 'image file path',
       'cropped image file path', 'ROI mask file path', 'type', 'calc type', 'calc distribution']]

print(cbis_mass.columns)
print(cbis_calc.columns)
print(cbis_mass.head())
print(cbis_calc.head())



# Combine for general overview
cbis_combined = pd.concat([cbis_mass, cbis_calc], join = 'outer', ignore_index = True, sort = False)

#print(cbis_combined.columns)

# Visualize pathology distribution
sns.countplot(data=cbis_combined, x='pathology', hue='type')
title = 'CBIS-DDSM Pathology Distribution by Type'
plt.title(title)
out_path = os.path.join(base_save_path, f'{title}.{ext}')
plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()


# Breast density
sns.countplot(data=cbis_combined, x='breast density')
title = 'Breast Density Distribution (CBIS-DDSM)'
plt.title(title)
out_path = os.path.join(base_save_path, f'{title}.{ext}')
plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()

cbis_combined.to_csv('cbis_combined.csv', index=True)

# === SIIM-ISIC ===
isic = pd.read_csv('melanoma/train.csv')

print("\nSIIM-ISIC Head:\n", isic.head())

# Diagnosis distribution
top_diagnoses = isic[isic['diagnosis'] != 'unknown']['diagnosis'].value_counts().nlargest(10)
top_diagnoses.plot(kind='bar')
title = 'Top 10 Diagnoses in SIIM-ISIC'
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.title(title)
out_path = os.path.join(base_save_path, f'{title}.{ext}')
plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()

# Age distribution
sns.histplot(data=isic, x='age_approx', bins=20, kde=True)
title = 'Age Distribution (SIIM-ISIC)'
plt.title(title)
out_path = os.path.join(base_save_path, f'{title}.{ext}')
plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()

# Anatomical site distribution
sns.countplot(data=isic, y='anatom_site_general_challenge', order=isic['anatom_site_general_challenge'].value_counts().index)
title = 'Anatomical Site Distribution'
plt.title(title)
out_path = os.path.join(base_save_path, f'{title}.{ext}')
plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()
