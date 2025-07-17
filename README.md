# oneClick-ml


---
# Robust ML Classifier Testing Pipeline

This repository provides a robust and usable pipeline for testing machine learning classifiers through a **randomized nested grid search cross-validation strategy**. The pipeline is designed to systematically evaluate classifier performance, identify optimal hyperparameters, and provide insights into feature importance through SHAP and permutation importance analysis.

---
## Features

* **Randomized Nested Cross-Validation:** Employs a robust nested cross-validation approach with randomized train-test splits for reliable performance estimation.
* **Hyperparameter Optimization:** Integrates randomized grid search to efficiently explore hyperparameter spaces and identify optimal configurations.
* **Multiple Classifier Support:** Designed to evaluate various machine learning classifiers.
* **Comprehensive Metric Collection:** Gathers a wide array of classification metrics for thorough performance assessment.
* **Feature Importance Analysis:** Includes modules for computing and visualizing SHAP (SHapley Additive exPlanations) values and permutation importance to understand feature contributions.
* **Data Imbalance Handling:** Option to incorporate strategies for dealing with imbalanced datasets.
* **Modular and Configurable:** Uses a configuration file (`config.yaml`) and parameter distribution file (`params.yaml`) for easy customization and reproducibility.
* **Organized Output:** Stores all results, metrics, and analysis outputs in a well-structured directory hierarchy.

---
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/endrwalter/oneClick-ML.git
    cd oneClick-ML
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
---
## Usage

The pipeline is executed via the `main.py` script, which takes a configuration file as an argument.

### Configuration

The project uses two primary configuration files:

* `config.ini`: Defines global parameters for the pipeline run, such as the target label, input/output paths, number of iterations, classifiers to use, and whether to perform SHAP or permutation importance analysis.
* `params.yaml`: Specifies the parameter distributions for the randomized grid search for each classifier.

**Example `config.ini`:**

```ini
; y label for this specific analysis, this label should be a column in the X.csv file present in analysis/Data/1_preprocessed_data
y_label =  FutureDyskinesia
; other col to drop. note that the current motor symptom at bl is always dropped - do not include it here
col_to_drop = FutureMotorFluctuations FutureFreezing FutureFalls max_status_longi CognitiveStatus
; perform feature selection using stat-test: True/False
include_feature_selector = False 
; how many features do you want to include?
n_of_features = 17
; path to folder that contains csv of feature types (one csv list per type)
path_to_feature_types = ../data/clusters/feature_list_types
; list of model to test (atm only: randomforestclassifier extratreesclassifier xgbclassifier logisticregression svc voting stacking)
classifiers_list = randomforestclassifier extratreesclassifier xgbclassifier logisticregression svc 
; shap analysis (True or False)
shap_analysis = True
; permutation importance analysis (True or False)
perm_importance = False
; path to input files
input_path = ../data/clusters/neuro_dyskinesia.csv
; path for output
output_path = ../results/no_active_at_bl/



[gridsearch params]
n_jobs = 30
; refit strategy for each randomized grid search cv
refit = mcc 
; number of repeated cross validations
cv_repeats = 5
; cross validation number of folds
cv_splits = 3
; randomized grid search cv number of iterations
n_iter = 20
; proportion of test/train data
train_test_split_size = 0.2
; how to handle imbalanced data: choose among all, SMOTE, UnderSampling, no
handle_imb_data = no
; number of overall iteration with differently generated train test splits (using different seed in train_test_split function)
nested_iterations = 30
