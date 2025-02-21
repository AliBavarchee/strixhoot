# **striXhooT: Hybrid Generative Model for Regression Tasks**
=============================================<p align="Center">![StrixHoot](https://teal-broad-gecko-650.mypinata.cloud/ipfs/bafybeihncepnasy4kavp5aa62tjginydqzplx4bsvg6oiucc3t6osnpefy)</p>=============================================
=====

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# striXhooT

striXhooT is a hybrid machine learning package that integrates generative models (CVAE, CGAN) with LightGBM for enhanced predictive modeling. The package is designed for structured datasets where generating synthetic data can improve **regression** performance.

## Features
- Train Conditional Variational Autoencoder (CVAE) and Conditional Generative Adversarial Network (CGAN) models.
- Use generative models to augment training data for better **regression** performance.
- Implement an optimized LightGBM model with optional hyperparameter tuning.
- Supports PCA and SVD for dimensionality reduction.
- Provides end-to-end orchestration of model training and evaluation.

## Installation
To install striXhoot, clone the repository and install dependencies:
```sh
# Clone the repository
git clone https://github.com/AliBavarchee/strixhoot.git
cd strixhoot

# Install dependencies
pip install -r requirements.txt
```

## Usage
Below is an example script to train the hybrid model using striXhoot:
```python
import os
import argparse
from strixhoot.trainer import main as train_model

# Define paths
input_path = "input.csv"  # Path to the dataset
output_path = "output_results"  # Directory to save all models and results

# Define training configuration
config = {
    "input_path": input_path,
    "output_path": output_path,
    "seed": 42,  # Set a random seed for reproducibility
    "gen_model": "both",  # Train both CVAE and CGAN models
    "train_generative": True,  # Enable training of generative models
    "tune_lgbm": False,  # Set to True if hyperparameter tuning is needed
    "dim_reducer": "pca",  # Choose between "pca" or "svd"
    "n_iter": 20,  # Number of iterations for hyperparameter tuning (only if tune_lgbm=True)
}

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Convert dictionary to argparse namespace
args = argparse.Namespace(**config)

# Train the models using the defined configuration
train_model(args)
```

## Configuration Options
| Parameter        | Description |
|-----------------|-------------|
| `input_path`    | Path to the dataset (CSV format) |
| `output_path`   | Directory where results and models will be saved |
| `seed`          | Random seed for reproducibility |
| `gen_model`     | Choose between `cvae`, `cgan`, or `both` |
| `train_generative` | Boolean flag to enable generative model training |
| `tune_lgbm`     | Boolean flag to enable LightGBM hyperparameter tuning |
| `dim_reducer`   | Choose dimensionality reduction method: `pca` or `svd` |
| `n_iter`        | Number of iterations for LightGBM hyperparameter tuning (only if `tune_lgbm=True`) |

## Results and Outputs
After running the script, the following artifacts will be saved in the `output_results/` directory:
- Trained CVAE and CGAN models
- Trained LightGBM regressor
- Performance metrics and visualizations (such as `true_vs_pred.png`)


## Contributor(s)
- Ali Bavarchee (ali.bavarchee@gmail.com)



## Configuration

Default parameters can be modified through:
- Command-line arguments
- JSON configuration files
- Python API parameters

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

=============================================<p align="Center">![ALI BAVARCHIEE](https://teal-broad-gecko-650.mypinata.cloud/ipfs/bafkreif332ra4lrdjfzaiowc2ikhl65uflok37e7hmuxomwpccracarqpy)</p>=============================================
=====
| https://github.com/AliBavarchee/ |
----
| https://www.linkedin.com/in/ali-bavarchee-qip/ |

