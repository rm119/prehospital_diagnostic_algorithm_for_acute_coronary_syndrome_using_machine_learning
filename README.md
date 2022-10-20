# Prehospital diagnostic algorithm for acute coronary syndrome

## Overview
Python codes to re-produce results of acute coronary syndrome prediction (Takeda et al. 2022 [1,2])

## Prerequisites
- Python 3.7.13
- Scikit-learn 1.0.2
- XGBoost 1.4.0
- lightGBM 3.2.1
- Numpy 1.21.6
- Pandas 1.3.5
- Matplotlib 3.2.2
- Shap 0.41.0
- Optuna 2.10.1


## Tested environment
Linux (Ubuntu 18.04)
Google Colab
MacOSX 11.6

## Dataset
- `data/datasheet.xlsx`: Original datasets.
- `data/datasheet_external_cohort.xlsx`: External datasets used for validation of the models.
- `data/datasheet_info.xlsx`: Definitions of each data column (see also the publication[1]).

## code
- `utils/global_variables.py` : Definitions of global variables.
- `utils/imputation.py` : procedure for imputation.
- `utils/preprocessor.py` : Preprocessing procedure including scaling and One-hot encoding.
- `utils/importdata.py` : Load original table data, add new columns including target columns.
- `utils/get_feature_names_for_sklearn1_0_2.py`: A tool to extract feature names after column transformation. To be used in case of Sklearn 1.0.2.
- `utils/generate_model.py`: Generate classifiers.
- `utils/tuning.py` : Functions for hyperparameter tuning
- `utils/evaluation.py` : Functions for model evaluation.
- `utils/utils.py` : Functions for save and load files.
- `utils/iteration.py` : Functions for cross-validation.
- `scripts/evaluate_models.py`: a script to evaluate each ML model and generate final models.



## Publication
- [1] Takeda M, Oami T, Hayashi Y, Shimada T, Hattori N, Tateishi K, Miura RE, Yamao Y, Abe R, Kobayashi Y, Nakada T. Prehospital diagnostic algorithm for acute coronary syndrome using machine learning: a prospective observational study. [DOI](https://doi.org/10.1038/s41598-022-18650-6), Sci Rep 12, 14593 (2022).　  [Read the article at the publisher's site](https://www.nature.com/articles/s41598-022-18650-6)

### Preprint
- [2] Takeda M, Oami T, Hayashi Y, Shimada T, Hattori N, Tateishi K, Miura RE, Yamao Y, Abe R, Kobayashi Y, Nakada T. Prehospital Diagnostic Algorithm for Acute Coronary Syndrome Using Machine Learning: A Prospective Observational Study. Research Square. 2022. [Read the preprint version at ResearchSquare](https://www.researchsquare.com/article/rs-1360222/v1)

## Citation
[Download citation](https://citation-needed.springer.com/v2/references/10.1038/s41598-022-18650-6?format=refman&flavour=citation)

PMID: 36028534
