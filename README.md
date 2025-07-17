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
    (You will need to create a `requirements.txt` file based on the imported libraries. A minimal `requirements.txt` would include `numpy`, `pandas`, `scikit-learn`, `PyYAML`, and `shap`.)

---
## Usage

The pipeline is executed via the `main.py` script, which takes a configuration file as an argument.

### Configuration

The project uses two primary configuration files:

* `config.yaml`: Defines global parameters for the pipeline run, such as the target label, input/output paths, number of iterations, classifiers to use, and whether to perform SHAP or permutation importance analysis.
* `params.yaml`: Specifies the parameter distributions for the randomized grid search for each classifier.

**Example `config.yaml`:**

```yaml
y_label: 'target_column'
input_path: 'data/input_data.csv'
output_path: 'results/'
path_to_feature_types: 'data/feature_types.json' # Example, if you have a file mapping feature types
nested_iterations: 10
train_test_split_size: 0.2
classifiers_list: ['RandomForestClassifier', 'SVC'] # Example classifier names
refit: 'mcc' # Metric to refit the best estimator on
n_jobs: -1 # Number of jobs to run in parallel (-1 means all available processors)
cv_repeats: 3
cv_splits: 5
n_iter: 100 # Number of parameter settings that are sampled
shap_analysis: True
perm_importance: True
handle_imb_data: False # Set to true to handle imbalanced data
include_feature_selector: False # Set to true to include feature selection in the pipeline
n_of_features: 10 # Number of features to select if feature selector is included
col_to_drop: ['ID'] # Columns to drop from the input data
