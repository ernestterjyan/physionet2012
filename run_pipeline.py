import argparse
import pandas as pd
import json
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from preprocessor import Preprocessor
from model import Model
from imblearn.over_sampling import SMOTE
class Pipeline:
    def _evaluate_model(self, y_true, y_pred, y_proba):
        # Evaluate the model's performance and print key metrics
        accuracy = accuracy_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(y_true, y_proba)
        mcc = matthews_corrcoef(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        print(f'Accuracy: {accuracy}')
        print(f'Sensitivity: {sensitivity}')
        print(f'Specificity: {specificity}')
        print(f'AUC Score: {auc}')
        print(f'MCC: {mcc}')
        print('Classification Report:')
        print(report)

    def run(self, data_path, test=False, save_model=False):
        # Load data and run the pipeline in either test or training mode
        data = pd.read_csv(data_path)

        if test:
            # In test mode: load preprocessor and model, predict, and save results
            preprocessor = Preprocessor.load('preprocessor.joblib')
            model = Model.load('model.joblib')

            # Handle missing columns gracefully
            try:
                data_reduced = data.drop(columns=preprocessor.columns_to_drop)
            except KeyError as e:
                missing_columns = list(e.args[0])
                print(f"Warning: Columns {missing_columns} not found in the dataset. Continuing without dropping them.")
                data_reduced = data.copy()

            X_test = preprocessor.transform(data_reduced)
            predictions = model.predict_proba(X_test)[:, 1]
            threshold = 0.5  # Threshold for classification
            prediction_results = {'predict_probas': predictions.tolist(), 'threshold': threshold}
            with open('predictions.json', 'w') as f:
                json.dump(prediction_results, f)
        else:
            # In training mode: preprocess data, split, train model, evaluate, and optionally save
            preprocessor = Preprocessor()
            data_reduced = data.drop(columns=preprocessor.columns_to_drop)
            y = data['In-hospital_death']
            X = data_reduced.drop('In-hospital_death', axis=1)
            preprocessor.fit(X)
            X_transformed = preprocessor.transform(X)

            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

            # Apply SMOTE only on training data
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            model = Model()
            model.fit(X_train_resampled, y_train_resampled)

            # Evaluate the model on the validation set
            y_pred_val = model.predict(X_val)
            y_proba_val = model.predict_proba(X_val)[:, 1]
            self._evaluate_model(y_val, y_pred_val, y_proba_val)

            if save_model:
                # Save the preprocessor and model states for future use
                preprocessor.save('preprocessor.joblib')
                model.save('model.joblib')
if __name__ == "__main__":
    # Parse command line arguments for data path, test mode, and model saving
    parser = argparse.ArgumentParser(description='Run the pipeline')
    parser.add_argument('--data_path', required=True, help='Path to the dataset')
    parser.add_argument('--test', action='store_true', help='Run in testing mode')
    parser.add_argument('--save_model', action='store_true', help='Save the model and preprocessor')
    args = parser.parse_args()

    # Instantiate and run the pipeline
    pipeline = Pipeline()
    pipeline.run(args.data_path, test=args.test, save_model=args.save_model)



'''
Accuracy: 0.87
Sensitivity: 0.378698224852071
Specificity: 0.950533462657614
AUC Score: 0.843714667783906
MCC: 0.389054572342253
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.95      0.93      1031
           1       0.56      0.38      0.45       169

    accuracy                           0.87      1200
   macro avg       0.73      0.66      0.69      1200
weighted avg       0.85      0.87      0.86      1200

'''