from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
import joblib

class Model:
    def __init__(self):
        # Initialize XGBoost and CatBoost
        self.xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.cat = CatBoostClassifier(random_state=42, silent=True)

        # Create ensemble model - VotingClassifier
        self.ensemble_model = VotingClassifier(
            estimators=[('xgb', self.xgb), ('cat', self.cat)],
            voting='soft'
        )

        # Define an expanded parameter grid for hyperparameter tuning
        self.param_grid = {
            'xgb__n_estimators': [100, 200],
            'xgb__max_depth': [3, 6],
            'xgb__learning_rate': [0.1, 0.2],
            'cat__iterations': [100, 200],
            'cat__depth': [4, 6],
            'cat__learning_rate': [0.1, 0.22]
        }

        # Setup GridSearchCV with 4-fold cross-validation
        self.kfold = KFold(n_splits=4, random_state=42, shuffle=True)
        self.grid_search = GridSearchCV(
            estimator=self.ensemble_model,
            param_grid=self.param_grid,
            cv=self.kfold,
            n_jobs=-1,
            scoring='roc_auc',
            verbose=2
        )

    def fit(self, X, y):
        # Perform hyperparameter tuning and fit the best model
        self.grid_search.fit(X, y)
        self.ensemble_model = self.grid_search.best_estimator_

    def predict(self, X):
        # Predict the class labels using the best fitted model
        return self.ensemble_model.predict(X)

    def predict_proba(self, X):
        # Predict class probabilities using the best fitted model
        return self.ensemble_model.predict_proba(X)

    def save(self, filename):
        # Save the trained model to a file for future use
        joblib.dump(self.ensemble_model, filename)

    @staticmethod
    def load(filename):
        # Load and return a model instance from a saved file
        model = Model()
        model.ensemble_model = joblib.load(filename)
        return model
