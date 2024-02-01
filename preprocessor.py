import pandas as pd
from sklearn.impute import SimpleImputer
import joblib
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.columns_to_drop = ['SAPS-I', 'SOFA', 'Survival', 'Length_of_stay']
        self.thresholds = {
            'Age': {'min': 0, 'max': 120},
            'Height': {'min': 100, 'max': 250},  # Assuming a plausible height range in cm
            'pH_last': {'min': 6.8, 'max': 7.8},
            'Temp_last': {'min': 32, 'max': 43},  # Temperature in Celsius
            'TroponinI_last': {'min': 0, 'max': 50},
            'WBC_last': {'min': 0.1, 'max': 100},  # White blood cell count in 10^9/L
            'Weight_last': {'min': 3, 'max': 300},  # Weight in kg
            'PaCO2_first': {'min': 10, 'max': 100},  # Partial pressure of carbon dioxide in mmHg
            'PaCO2_last': {'min': 10, 'max': 100}
}

    def fit(self, X):
        self.imputer.fit(X)

    def transform(self, X):
        X = self._convert_height(X)
        X = self._mark_implausible_as_nan(X)
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    def _convert_height(self, X):
        X['Height'] = X['Height'].apply(lambda h: h * 2.54 if h < 60 else h)
        return X

    def _mark_implausible_as_nan(self, X):
        for col, limits in self.thresholds.items():
            if col in X.columns:
                X[col] = X[col].apply(lambda x: None if x < limits['min'] or x > limits['max'] else x)
        return X

    def save(self, filename):
        joblib.dump(self, filename)

    @staticmethod
    def load(filename):
        return joblib.load(filename)
