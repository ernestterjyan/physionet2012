# physionet2012
# PhysioNet 2012 Challenge Solution

## Project Overview
This project is a solution to the PhysioNet 2012 Challenge, focusing on predicting in-hospital mortality based on clinical data. The solution includes a data preprocessing step, a machine learning model for predictions, and a pipeline to run the model in either training or testing mode.

### Team Members
- Ernest
- Karen G.
- Shavarsh

## Files in the Repository
- `preprocessor.py`: Contains the `Preprocessor` class for data preprocessing, including filling NaNs, scaling, and feature extraction.
- `model.py`: Contains the `Model` class with `fit` and `predict` methods. This class is used for training the machine learning model and making predictions.
- `run_pipeline.py`: Contains the `Pipeline` class with a `run` method. This script can be run with two arguments: `--data_path` for the dataset path and `--test` to specify the mode (testing or training).

## Usage
To train the model:

python run_pipeline.py --data_path [path_to_training_data] --save_model (optional)


To test the model:

python run_pipeline.py --data_path [path_to_test_data] --test


## Model and Preprocessing
The model used is a Gradient Boosting Classifier. The preprocessing steps include scaling and handling missing values.

### Validation Results
- Accuracy: 0.87
- Sensitivity: 0.379
- Specificity: 0.951
- AUC Score: 0.844
- MCC: 0.389

## Requirements
For required libraries, see `requirements.txt`.

- Model and preprocessor are saved after training.
- The `run_pipeline.py` script in testing mode loads the saved model and preprocessor to make predictions.
- Predictions are saved in `predictions.json` as described in the project description.

