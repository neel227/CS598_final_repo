import os
import pandas as pd
import time
from fit_words import fit_words, get_prototypes
from visualize_weights import make_word_weights_plot

def log_runtime(start, end):
    return round(end - start, 2)

if __name__ == '__main__':
    dataset_name = 'cbis' # 'melanoma' or 'cbis'
    device = 'cpu'

    # Two possible dictionaries
    dict1 = [
        'dark', 'light',
        'round', 'pointed',
        'large', 'small',
        'smooth', 'coarse',
        'transparent', 'opaque',
        'symmetric', 'asymmetric',
        'high contrast', 'low contrast'
    ]

    dict2 = [
        # Shape/Border Characteristics
        'irregular', 'well-defined',
        'spiculated', 'smooth',
        'lobulated', 'oval',
        'asymmetric', 'symmetric',

        # Internal Texture
        'heterogeneous', 'homogeneous',
        'dense', 'lucent',
        'calcified', 'non-calcified',

        # Growth Behavior / Appearance
        'invasive', 'contained',
        'rapid growing', 'slow growing',

        # Color and Contrast (esp. for dermoscopic images)
        'dark', 'light',
        'high contrast', 'low contrast',
        'multicolored', 'uniform color',

        # Surface / Pattern
        'scalloped', 'regular edges',
        'blurred margins', 'sharp margins'
    ]

    # Choose which data to use
    if dataset_name == 'cbis':
        train_path = './cbis/train_split.csv'
        test_path = './cbis/test_split.csv'
    elif dataset_name == 'melanoma':
        train_path = './melanoma/train_split.csv'
        test_path = './melanoma/test_split.csv'

    # Choose which dictionary to use
    dict_choice = dict1

    # Choose which model type to use
    model_type = 'Ridge' 

    # Print Datset Info
    print("Dataset:", dataset_name)
    print("Dictionary: ", dict_choice)
    print("Model Type: ", model_type)
    print("")

    base_out_dir = './results/'
    if not os.path.exists(base_out_dir):
        os.mkdir(base_out_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    save_tag = dataset_name
    save_dir = os.path.join(base_out_dir, save_tag)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    metrics = {
        'dataset': dataset_name,
        'n_train_samples': len(train_df),
        'n_test_samples': len(test_df)
    }

    start_fit = time.time()
    fit_words(train_df, test_df, device, dict_choice, save_dir=save_dir, save_tag=save_tag, model_type = model_type)
    end_fit = time.time()
    metrics['fit_words_time_sec'] = log_runtime(start_fit, end_fit)

    prot_save_dir = os.path.join(save_dir, save_tag + '_prototypes')
    if not os.path.exists(prot_save_dir):
        os.mkdir(prot_save_dir)

    start_proto = time.time()
    get_prototypes(train_df, dict_choice, device, prot_save_dir, n_save=5)
    end_proto = time.time()
    metrics['get_prototypes_time_sec'] = log_runtime(start_proto, end_proto)


    # Load word weights and plot
    word_weights_path = os.path.join(save_dir, f'word_weights-{save_tag}.csv')
    word_df = pd.read_csv(word_weights_path, index_col=0)

    start_plot = time.time()
    make_word_weights_plot(word_df, os.path.join(save_dir, f'{save_tag}_weights_plot'))
    end_plot = time.time()
    metrics['make_plot_time_sec'] = log_runtime(start_plot, end_plot)
    
    print("\n=== Runtime & Dataset Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Save as CSV log
    log_df = pd.DataFrame([metrics])
    log_df.to_csv(os.path.join(save_dir, f'{save_tag}_run_log.csv'), index=False)