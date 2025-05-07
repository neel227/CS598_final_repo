from scipy.spatial import distance
import numpy as np
import pandas as pd
from PIL import Image
import torch
import clip
import pydicom
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import resample
import os
import imageio.v2 as imageio
import cv2

# Function for processing .dcm files
def process_dcm(file_path):
    ds = pydicom.dcmread(file_path)
    im = ds.pixel_array.astype(float)
    
    im = im / im.max()
    im = (255 * im).astype(np.uint8)

    # If grayscale, convert to RGB
    if im.ndim == 2:
        im_rgb = np.stack([im] * 3, axis=-1)  # shape: (H, W, 3)
    elif im.ndim == 3 and im.shape[2] == 3:
        im_rgb = im
    else:
        raise ValueError(f"Unexpected image shape after processing: {im.shape}")

    return im_rgb

# Function for creating the features of the image
def create_clip_feature_mat(file_list, clip_model, preprocess_fxn, device):
    X = np.zeros((len(file_list), 512)) # 512 is feature dimension
    for i, f in tqdm(enumerate(file_list), total=len(file_list)):
        if '.dcm' in f:
            im = Image.fromarray(process_dcm(f))
        else:
            im = Image.open(f)
        im = preprocess_fxn(im).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(im)
        X[i] = image_features[0].cpu().numpy()

    return X

# Function for fitting words using computervision model
def fit_words(train_df, test_df, device, words, save_dir, save_tag, model_type = "Linear"):
    clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
    X_train = create_clip_feature_mat(train_df.file_path.values, clip_model, preprocess_fxn, device)

    classifier = LogisticRegression(random_state=0, C=1, max_iter=1000, verbose=1, fit_intercept=False)
    classifier.fit(X_train, train_df.label.values)

    tokened_words = clip.tokenize(words).to(device)
    with torch.no_grad():
        word_features = clip_model.encode_text(tokened_words)

    # Choose which model type: Linear or Ridge
    weights_model = LinearRegression(fit_intercept=False)
    if model_type.capitalize == 'Ridge':
        alpha_list = [.01, .1] + list(range(10, 51, 10))
        weights_model = RidgeCV(alphas=alpha_list, fit_intercept=False)
    else:
        print("Error In model choice. Defaulting to Linear Regression.")


    weights_model.fit(word_features.cpu().T, classifier.coef_[0])
    word_df = pd.DataFrame({'word': words, 'weights': weights_model.coef_})
    word_df.sort_values('weights', inplace=True)
    word_df.set_index('word', inplace=True)
    word_df.to_csv(os.path.join(save_dir, f'word_weights-{save_tag}.csv'))

    X_test = create_clip_feature_mat(test_df.file_path.values, clip_model, preprocess_fxn, device)

    # Get predicted probabilities for the positive class (malignant = 1)
    yhat_proba = classifier.predict_proba(X_test)[:, 1]
    #yhat = classifier.predict_proba(X_test)
    y_true = test_df.label.values

    eval_results = evaluate_with_ci(X_train, train_df.label.values, classifier.predict_proba(X_train)[:, 1], word_features.cpu().numpy())

    for metric, (mean, (low, high)) in eval_results.items():
        print(f"{metric}: {mean:.3f} (CI {low:.3f}–{high:.3f})")


    # Calculate AUROC
    auroc = roc_auc_score(y_true, yhat_proba)

    # Print results
    print('Test Accuracy:', classifier.score(X_test, y_true))
    print('AUROC:', auroc)

    pred_coef = weights_model.predict(word_features.cpu().T)
    cos_sim = 1 - distance.cosine(pred_coef, classifier.coef_[0])
    print('cosine sim between weights', cos_sim)

# Get metrics to evaluate classifier weights (including confidence intervals)
def evaluate_with_ci(X, y, y_proba, word_features, n_bootstraps=1000, ci=0.95):
    accs, aucs, cos_sims = [], [], []
    n = len(y)
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, n, replace=True)
        y_boot = y[idx]
        X_boot = X[idx]
        y_proba_boot = y_proba[idx]

        try:
            # Accuracy
            accs.append(accuracy_score(y_boot, y_proba_boot >= 0.5))

            # AUROC
            aucs.append(roc_auc_score(y_boot, y_proba_boot))

            # Cosine similarity
            clf = LogisticRegression(fit_intercept=False, max_iter=1000)
            clf.fit(X_boot, y_boot)
            lr = LinearRegression(fit_intercept=False)
            lr.fit(word_features.T, clf.coef_[0])
            pred_coef = lr.predict(word_features.T)
            cos = 1 - distance.cosine(pred_coef, clf.coef_[0])
            cos_sims.append(cos)
        except:
            continue  # Skip bootstraps with class imbalance

    def ci_bounds(values):
        lower = np.percentile(values, (1 - ci) / 2 * 100)
        upper = np.percentile(values, (1 + ci) / 2 * 100)
        return np.mean(values), (lower, upper)

    return {
        'accuracy': ci_bounds(accs),
        'auroc': ci_bounds(aucs),
        'cosine_similarity': ci_bounds(cos_sims)
    }


# Get the scores for each word
def get_prototypes(df, words, device, save_dir, n_save=20):
    clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
    X = create_clip_feature_mat(df.file_path.values, clip_model, preprocess_fxn, device)

    tokened_words = clip.tokenize(words).to(device)
    with torch.no_grad():
        word_features = clip_model.encode_text(tokened_words)

    file_dot = np.zeros((len(df), len(words)))
    for i in range(len(df)):
        for j in range(len(words)):
            file_dot[i, j] = np.dot(X[i], word_features[j].cpu().numpy())


    file_dot_pred = np.zeros((len(df), len(words)))
    for j in range(len(words)):
        fit_j = [k for k in range(len(words)) if k != j]
        dot_regression = LinearRegression()
        dot_regression.fit(file_dot[:, fit_j], file_dot[:, j])
        file_dot_pred[:, j] = dot_regression.predict(file_dot[:, fit_j])

    dot_df_diff = pd.DataFrame(file_dot - file_dot_pred, columns=words)
    dot_df_diff['label'] = df['label'].values
    dot_df_diff.set_index(df.file_path, inplace=True)

    for w in words:
        print(w)
        for sort_dir in ['top']:
            this_df = dot_df_diff.sort_values(w, ascending=(sort_dir == 'bottom'))
            save_files = this_df.index.values[:n_save]
            these_labels = this_df.label.values[:n_save]
            this_out_dir = save_dir + w + '_' + sort_dir + '/'
            if not os.path.exists(this_out_dir):
                os.mkdir(this_out_dir)

            for i, f in enumerate(save_files):
                if '.dcm' in f:
                    im = process_dcm(f)
                else:
                    im = imageio.imread(f)

                # If grayscale, convert to RGB
                if im.ndim == 2:
                    im = np.stack([im] * 3, axis=-1)  # shape: (H, W, 3)
                elif not (im.ndim == 3 and im.shape[2] == 3):
                    raise ValueError(f"Unexpected image shape after processing: {im.shape}")
                    
                # make square and downsample for efficiency (CLIP also crops to square)
                min_dim = min(im.shape[:2])
                for dim in [0, 1]:
                    if im.shape[dim] > min_dim:
                        n_start = int((im.shape[dim] - min_dim) / 2)
                        n_stop = n_start + min_dim
                        if dim == 0:
                            im = im[n_start:n_stop, :, :]
                        else:
                            im = im[:, n_start:n_stop, :]
                if min_dim > 500:
                    im = cv2.resize(im, (500, 500))
                f_name = f'rank{i}_label{these_labels[i]}.png'
                imageio.imwrite(os.path.join(this_out_dir, f_name), im)


