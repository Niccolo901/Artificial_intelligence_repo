import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import plotly.express as px

class DataProcessor:
    def __init__(self, file_path):
        """
        Initialize the DataProcessor with the dataset file path.

        Args:
            file_path (str): Path to the housing dataset CSV file.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Load the dataset from the CSV file.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully with shape: {self.data.shape}")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def check_missing_values(self):
        """
        Check and report missing values in the dataset.

        Returns:
            pd.Series: Series with counts of missing values per column.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        missing_values = self.data.isnull().sum()
        print("Missing values per column:")
        print(missing_values[missing_values > 0])
        return missing_values

    def plot_missing_values(self):
        """
        Plot a heatmap of missing values in the dataset with improved readability.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        missing = self.data.isnull()
        if missing.sum().sum() == 0:
            print("No missing values found in the dataset.")
        else:
            plt.figure(figsize=(12, 8))  # Increased figure size for better readability
            sns.heatmap(missing, cbar=False, cmap="viridis", yticklabels=False, xticklabels=True)
            plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate and align column names
            plt.title("Missing Values Heatmap", fontsize=14)
            plt.tight_layout()  # Adjust layout to prevent clipping
            plt.show()

    def drop_missing_values(self):
        """
        Drop rows with missing values from the dataset.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        initial_shape = self.data.shape
        self.data.dropna(inplace=True)
        print(f"Dropped rows with missing values. Shape changed from {initial_shape} to {self.data.shape}.")

    def normalize_columns(self, columns_to_normalize):
        """
        Normalize specified columns in the dataset.

        Args:
            columns_to_normalize (list): List of column names to normalize.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        missing_cols = [col for col in columns_to_normalize if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found for normalization: {missing_cols}")

        scaler = StandardScaler()
        self.data[columns_to_normalize] = scaler.fit_transform(self.data[columns_to_normalize])
        print("Normalization complete for columns:", columns_to_normalize)

    def transform_categorical_to_numeric(self):
        """
        Transform categorical variables in the dataset into one-hot encoded numeric variables.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        categorical_columns = self.data.select_dtypes(include=['object']).columns.tolist()
        if not categorical_columns:
            print("No categorical columns found to transform.")
            return
        self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)
        print("Categorical variables transformed into numeric (one-hot encoded) variables.")

    def get_numeric_columns(self):
        """
        Retrieve numeric columns from the dataset.

        Returns:
            list: List of numeric column names.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        numeric_columns = self.data.select_dtypes(include=['number']).columns.tolist()
        return numeric_columns

    def get_data(self):
        """
        Retrieve the processed DataFrame.

        Returns:
            pd.DataFrame: The processed dataset.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data

    def save_processed_data(self, output_file):
        """
        Save the processed dataset to a CSV file.

        Args:
            output_file (str): The file path to save the processed dataset.
        """
        if self.data is None:
            raise ValueError("Data not loaded or processed. Nothing to save.")
        self.data.to_csv(output_file, index=False)
        print(f"Processed dataset saved to '{output_file}'.")


class EDA:
    def __init__(self, data):
        """
        Initialize the EDA class with a dataset.

        Args:
            data (pd.DataFrame): The input dataset for analysis.
        """
        self.data = data

    def display_basic_info(self):
        """
        Display basic information about the dataset, including shape and data types.
        """
        print("Dataset Information:")
        print(self.data.info())
        print("\nFirst five rows of the dataset:")
        print(self.data.head())
        print("\nDescriptive statistics:")
        print(self.data.describe())

    def plot_lat_long(self, city_name="Map of Houses"):
        """
        Plot the locations on an interactive map using longitude and latitude.

        Args:
            city_name (str): Title of the plot.
        """
        if "longitude" not in self.data.columns or "latitude" not in self.data.columns:
            print("Longitude or latitude column not found in the dataset.")
            return

        fig = px.scatter_mapbox(
            self.data,
            lat="latitude",
            lon="longitude",
            hover_data=["median_house_value", "median_income"],
            zoom=8,
            height=600,
            width=800,
            title=city_name
        )

        # Use an open-source map style
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
        fig.show()

    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix of numeric features in the dataset.
        """
        # Filter numeric features only
        numeric_data = self.data.select_dtypes(include=['number'])

        # Compute correlation matrix
        correlation = numeric_data.corr()

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix of Numeric Features")
        plt.show()

    def compare_distribution_and_boxplot(self, column):
        """
        Plot the distribution and boxplot of a specified column stacked vertically for comparison.

        Args:
            column (str): Column name to plot.
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found in the dataset.")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])

        # Distribution plot
        sns.histplot(self.data[column], kde=True, bins=30, color="blue", ax=axes[0])
        axes[0].set_title(f"Distribution of {column}", fontsize=14)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Frequency", fontsize=12)

        # Boxplot
        sns.boxplot(x=self.data[column], ax=axes[1], color="skyblue")
        axes[1].set_title(f"Boxplot of {column}", fontsize=14)
        axes[1].set_xlabel(column, fontsize=12)

        # Adjust layout and show
        plt.tight_layout()
        plt.show()

    def check_outliers(self, column):
        """
        Plot a boxplot to identify potential outliers in a specific column.

        Args:
            column (str): Column name to check for outliers.
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found in the dataset.")
            return

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.data[column])
        plt.title(f"Boxplot of {column} (Outliers Detection)")
        plt.xlabel(column)
        plt.show()

    def plot_distribution(self, column):
        """
        Plot the distribution of a specified column.

        Args:
            column (str): Column name to plot.
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found in the dataset.")
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def plot_relationship(self, feature_x, feature_y):
        """
        Plot a scatterplot to visualize the relationship between two features.

        Args:
            feature_x (str): Name of the feature for the x-axis.
            feature_y (str): Name of the feature for the y-axis.
        """
        if feature_x not in self.data.columns or feature_y not in self.data.columns:
            print(f"One or both features '{feature_x}' and '{feature_y}' not found in the dataset.")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data[feature_x], y=self.data[feature_y], alpha=0.6)
        plt.title(f"Scatterplot: {feature_x} vs {feature_y}")
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.grid(True)
        plt.show()



