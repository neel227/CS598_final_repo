# task_word_explainability
Code for CS 598: Deep Learning for Healthcare as part of UIUC masters in computer Science coursework. Working to replicate and extend results from S Agarwal, YR Semenov, W Lotter. Representing visual classification as a linear combination of words. _Proceedings of the 3rd Machine Learning for Health symposium, PMLR_ (2023).

## Set-up/Requirements
1. Install CLIP and its dependencies by following instructions at: https://github.com/openai/CLIP
2. Install packages in `requirements.txt` in order to run all scripts. The code should generally be version agnostic for these packages.

## Usage
`data_exploration.py`: Creates some general visualizations to understand distributions of CBIS and SIIM-ISIC datasets 

`main.py`: Illustrates fitting a linear classifier based on CLIP image embeddings and subquently estimating the classifier based on a combination of word embeddings. Enables user to vary dataset, underlying dictionary, and whether the word embedding combination will be linear or ridge. Also plots the regression word weights.
