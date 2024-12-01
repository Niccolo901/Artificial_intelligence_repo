import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

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

    def train_neural_network_with_tf(self, hyperparameter_tuning=False, param_grid=None, epochs=50, batch_size=32):
        """
        Train a Neural Network model using TensorFlow and evaluate its performance.

        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
            param_grid (dict): Parameter grid for hyperparameter tuning.
            epochs (int): Number of training epochs.
            batch_size (int): Size of the training batch.

        """

        def build_model(hidden_layer_sizes=(100,), activation='relu', learning_rate=0.01):
            """
            Build a Sequential Neural Network model using TensorFlow.

            Args:
                hidden_layer_sizes (tuple): Sizes of hidden layers.
                activation (str): Activation function for hidden layers.
                learning_rate (float): Learning rate for the optimizer.

            Returns:
                tf.keras.Model: Compiled Neural Network model.
            """
            model = Sequential()
            model.add(Input(shape=(self.X_train.shape[1],)))  # Use Input layer for better compatibility
            for layer_size in hidden_layer_sizes:
                model.add(Dense(layer_size, activation=activation))
            model.add(Dense(1, activation='linear'))

            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            return model

        # Hyperparameter tuning
        if hyperparameter_tuning:
            if param_grid is None:
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                    'activation': ['relu', 'tanh'],
                    'learning_rate': [0.001, 0.01, 0.1],
                }
            best_loss = float('inf')
            best_params = None
            best_model = None

            for hidden_layer_sizes in param_grid['hidden_layer_sizes']:
                for activation in param_grid['activation']:
                    for learning_rate in param_grid['learning_rate']:
                        print(
                            f"Testing parameters: hidden_layer_sizes={hidden_layer_sizes}, activation={activation}, learning_rate={learning_rate}")
                        model = build_model(hidden_layer_sizes, activation, learning_rate)
                        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                                            epochs=epochs, batch_size=batch_size, verbose=0)
                        val_loss = min(history.history['val_loss'])
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_params = {
                                'hidden_layer_sizes': hidden_layer_sizes,
                                'activation': activation,
                                'learning_rate': learning_rate,
                            }
                            best_model = model

            print(f"Best Parameters for Neural Network: {best_params}")
            model = best_model
        else:
            # Default configuration
            model = build_model(hidden_layer_sizes=(100,), activation='relu', learning_rate=0.01)
            history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                                epochs=epochs, batch_size=batch_size, verbose=1)

        # Evaluate the model
        predictions = model.predict(self.X_test).flatten()
        self.evaluate_model("Neural Network", predictions)

        # Plot training and validation loss
        self.plot_training_validation_loss(history)

        # Save the trained model
        self.models['Neural Network'] = model

    def plot_training_validation_loss(self, history):
        """
        Plot training and validation loss over epochs.

        Args:
            history (History): Training history object from TensorFlow.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss', marker='o')
        plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def train_xgboost(self, hyperparameter_tuning=False, param_grid=None):
        """
        Train an XGBoost model and evaluate its performance.

        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
            param_grid (dict): Parameter grid for hyperparameter tuning.
        """
        best_params = None  # Store best parameters
        if hyperparameter_tuning:
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                }
            model = GridSearchCV(
                XGBRegressor(random_state=42),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error'
            )
            model.fit(self.X_train, self.y_train)
            best_params = model.best_params_
            print(f"Best Parameters for XGBoost: {best_params}")
            model = model.best_estimator_
        else:
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=1.0,
                colsample_bytree=1.0,
                random_state=42
            )
            model.fit(self.X_train, self.y_train)

        self.models['XGBoost'] = model
        predictions = model.predict(self.X_test)
        self.evaluate_model("XGBoost", predictions)
        self.plot_xgboost_feature_importance(model, best_params)

    def plot_xgboost_feature_importance(self, model, best_params=None):
        """
        Plot the feature importance for the XGBoost model with the best hyperparameters displayed.

        Args:
            model (XGBRegressor): Trained XGBoost model.
            best_params (dict, optional): Best hyperparameters from GridSearchCV.
        """
        importances = model.feature_importances_
        features = self.X.columns

        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Create the plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color="orange")

        # Add importance values to the bars
        for bar, importance in zip(bars, importance_df['Importance']):
            plt.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.4f}",
                va="center",
                ha="left",
                fontsize=10,
                color="black"
            )

        plt.title("XGBoost Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.gca().invert_yaxis()  # Largest importance on top
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Display best hyperparameters, if available
        if best_params:
            param_text = "\n".join([f"{k}: {v}" for k, v in best_params.items()])
            plt.figtext(0.95, 0.5, f"Best Parameters:\n{param_text}", fontsize=10, ha="left", va="center")

        plt.tight_layout()
        plt.show()

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
        Train and evaluate all models sequentially and display a comparative summary.

        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
        """
        results = []

        # Train Linear Regression
        print("Training Linear Regression...")
        self.train_linear_regression()
        results.append(self.evaluate_model_for_summary("Linear Regression"))

        # Train Random Forest
        print("\nTraining Random Forest...")
        self.train_random_forest(hyperparameter_tuning=hyperparameter_tuning)
        results.append(self.evaluate_model_for_summary("Random Forest"))

        # Train Neural Network
        print("\nTraining Neural Network...")
        self.train_neural_network_with_tf(hyperparameter_tuning=hyperparameter_tuning)
        results.append(self.evaluate_model_for_summary("Neural Network"))

        # Train XGBoost
        print("\nTraining XGBoost...")
        self.train_xgboost(hyperparameter_tuning=hyperparameter_tuning)
        results.append(self.evaluate_model_for_summary("XGBoost"))

        # Summarize results
        results_df = pd.DataFrame(results)
        self.plot_model_comparisons(results_df)
        print("\nModel Evaluation Summary:")
        print(results_df)

    def evaluate_model_for_summary(self, model_name):
        """
        Evaluate a model and return the metrics as a dictionary for summary.

        Args:
            model_name (str): Name of the model to evaluate.

        Returns:
            dict: Dictionary containing evaluation metrics for the model.
        """
        predictions = self.models[model_name].predict(self.X_test)

        mae = mean_absolute_error(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, predictions)

        n = len(self.y_test)
        p = self.X_test.shape[1]
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        return {
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2,
            "Adjusted R²": adjusted_r2,
        }

    def plot_model_comparisons(self, results_df):
        """
        Plot a bar chart comparing R² scores for different models.

        Args:
            results_df (pd.DataFrame): DataFrame containing model evaluation metrics.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(results_df["Model"], results_df["R²"], color="skyblue")
        plt.title("Model Comparison (R² Scores)")
        plt.ylabel("R² Score")
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
