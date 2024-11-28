import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

class SupervisedAnalysis:
    def __init__(self, data, target_column):
        """
        Initialize the SupervisedAnalysis class with a dataset.

        Args:
            data (pd.DataFrame): The input dataset.
            target_column (str): The name of the target column to predict.
        """
        self.data = data
        self.target_column = target_column
        self.models = {}

        # Prepare data for training and testing
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_linear_regression(self):
        """
        Train a Linear Regression model and evaluate its performance.
        """
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = model

        predictions = model.predict(self.X_test)
        self.evaluate_model("Linear Regression", predictions)
        self.plot_linear_regression_coefficients(model)

    def train_random_forest(self, hyperparameter_tuning=False, param_grid=None):
        """
        Train a Random Forest model and evaluate its performance.

        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
            param_grid (dict): Parameter grid for hyperparameter tuning.
        """
        if hyperparameter_tuning:
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20, 50],
                    'min_samples_split': [2, 5, 10],
                }
            model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
            model.fit(self.X_train, self.y_train)
            print(f"Best Parameters for Random Forest: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
            model.fit(self.X_train, self.y_train)

        self.models['Random Forest'] = model
        predictions = model.predict(self.X_test)
        self.evaluate_model("Random Forest", predictions)
        self.plot_random_forest_feature_importance(model)

    def plot_linear_regression_coefficients(self, model):
        """
        Plot the coefficients of variables in the Linear Regression model and return them as a DataFrame.
        The exact coefficient values are also displayed on the bars.

        Args:
            model (statsmodels.OLS or sklearn.LinearRegression): Trained Linear Regression model.

        Returns:
            pd.DataFrame: A DataFrame with features and their corresponding coefficients.
        """
        if hasattr(model, "params"):  # For statsmodels.OLS
            coef = model.params
            features = coef.index
        elif hasattr(model, "coef_"):  # For sklearn.LinearRegression
            coef = model.coef_
            features = self.X.columns
        else:
            raise ValueError("Unsupported model type. Use statsmodels.OLS or sklearn.LinearRegression.")

        # Create DataFrame of coefficients
        coef_df = pd.DataFrame({"Feature": features, "Coefficient": coef}).sort_values(
            by="Coefficient", key=abs, ascending=False
        )

        # Plot coefficients
        plt.figure(figsize=(12, 8))
        bars = plt.barh(coef_df["Feature"], coef_df["Coefficient"], color="skyblue")

        # Add coefficient values on the bars
        for bar, coeff in zip(bars, coef_df["Coefficient"]):
            plt.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"{coeff:.2f}",
                va="center",
                ha="left" if coeff > 0 else "right",
                fontsize=10,
                color="black"
            )

        plt.title("Linear Regression Coefficients")
        plt.xlabel("Coefficient Value")
        plt.ylabel("Feature")
        plt.gca().invert_yaxis()  # Largest coefficients on top
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Return the DataFrame for further inspection or saving
        return coef_df

    def plot_random_forest_feature_importance(self, model):
        """
        Plot the feature importance for the Random Forest model.

        Args:
            model (RandomForestRegressor): Trained Random Forest model.
        """
        importances = model.feature_importances_
        features = self.X.columns

        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color="green")
        plt.title("Random Forest Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.gca().invert_yaxis()  # Invert y-axis to show the largest importance on top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def train_neural_network(self, hyperparameter_tuning=False, param_grid=None):
        """
        Train a Neural Network model and evaluate its performance.

        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
            param_grid (dict): Parameter grid for hyperparameter tuning.
        """
        if hyperparameter_tuning:
            if param_grid is None:
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                    'activation': ['relu', 'tanh'],
                    'learning_rate': ['constant', 'adaptive'],
                }
            model = GridSearchCV(MLPRegressor(max_iter=10000, random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
            model.fit(self.X_train, self.y_train)
            print(f"Best Parameters for Neural Network: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=10000, random_state=42)
            model.fit(self.X_train, self.y_train)

        self.models['Neural Network'] = model
        predictions = model.predict(self.X_test)
        self.evaluate_model("Neural Network", predictions)

    def train_xgboost(self, hyperparameter_tuning=False, param_grid=None):
        """
        Train an XGBoost model and evaluate its performance.

        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
            param_grid (dict): Parameter grid for hyperparameter tuning.
        """
        if hyperparameter_tuning:
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                }
            model = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
            model.fit(self.X_train, self.y_train)
            print(f"Best Parameters for XGBoost: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            model.fit(self.X_train, self.y_train)

        self.models['XGBoost'] = model
        predictions = model.predict(self.X_test)
        self.evaluate_model("XGBoost", predictions)

    def evaluate_model(self, model_name, predictions):
        """
        Evaluate a model's performance using multiple metrics.

        Args:
            model_name (str): The name of the model.
            predictions (np.ndarray): The predicted values.
        """
        mae = mean_absolute_error(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, predictions)

        n = len(self.y_test)
        p = self.X_test.shape[1]
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        print(f"Model: {model_name}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Adjusted R-squared: {adjusted_r2:.4f}")
        print("-" * 50)

    def run_all_models(self, hyperparameter_tuning=False):
        """
        Train and evaluate all models.

        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
        """
        print("Training Linear Regression...")
        self.train_linear_regression()

        print("\nTraining Random Forest...")
        self.train_random_forest(hyperparameter_tuning=hyperparameter_tuning)

        print("\nTraining Neural Network...")
        self.train_neural_network(hyperparameter_tuning=hyperparameter_tuning)

        print("\nTraining XGBoost...")
        self.train_xgboost(hyperparameter_tuning=hyperparameter_tuning)
