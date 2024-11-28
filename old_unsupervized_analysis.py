import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px


class UnsupervisedAnalysis:
    def __init__(self, data):
        """
        Initialize the UnsupervisedAnalysis class with a dataset.

        Args:
            data (pd.DataFrame): The input dataset for analysis.
        """
        self.data = self.ensure_numeric(data)
        self.numeric_data = self.data.select_dtypes(include=["number"])
        self.pca_data = None
        self.pca_model = None

    @staticmethod
    def ensure_numeric(data):
        """
        Ensure all columns in the dataset are numeric.

        Args:
            data (pd.DataFrame): Input dataset.

        Returns:
            pd.DataFrame: Dataset with non-numeric columns converted where possible.
        """
        numeric_data = data.copy()
        for col in numeric_data.columns:
            if not pd.api.types.is_numeric_dtype(numeric_data[col]):
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors="coerce")
        numeric_data.dropna(axis=1, how="all", inplace=True)  # Drop columns that became entirely NaN
        print("Non-numeric columns have been converted where possible.")
        return numeric_data

    def perform_pca(self, n_components=None):
        """
        Perform Principal Component Analysis (PCA) on the dataset.

        If n_components is not specified, the Kaiser Rule is used to select the number of components.

        Args:
            n_components (int, optional): Number of principal components to compute.

        Returns:
            pd.DataFrame: Transformed dataset with principal components.
            pd.DataFrame: PCA loadings for each component.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.numeric_data)

        # Perform PCA
        pca = PCA()
        pca.fit(scaled_data)

        # Apply Kaiser Rule if n_components is not specified
        if n_components is None:
            eigenvalues = pca.explained_variance_
            n_components = sum(eigenvalues > 1)
            print(f"Using Kaiser Rule: Retaining {n_components} components with eigenvalues > 1")

        self.pca_data = pca.transform(scaled_data)[:, :n_components]
        self.pca_model = pca

        # Create a DataFrame for the PCA results
        pca_df = pd.DataFrame(
            self.pca_data, columns=[f"PC{i+1}" for i in range(n_components)]
        )

        # Compute PCA loadings
        loadings = pd.DataFrame(
            pca.components_[:n_components],
            columns=self.numeric_data.columns,
            index=[f"PC{i+1}" for i in range(n_components)]
        )

        print(f"Explained Variance Ratio (Selected Components): {pca.explained_variance_ratio_[:n_components]}")
        return pca_df, loadings

    def plot_screeplot(self):
        """
        Plot a scree plot showing the explained variance ratio of each principal component.
        """
        if self.pca_model is None:
            print("PCA has not been performed yet. Call perform_pca() first.")
            return

        eigenvalues = self.pca_model.explained_variance_
        plt.figure(figsize=(10, 6))
        plt.plot(
            np.arange(1, len(eigenvalues) + 1),
            eigenvalues,
            marker="o",
            linestyle="--",
        )
        plt.axhline(y=1, color="red", linestyle="--", label="Kaiser Rule Threshold")
        plt.title("Scree Plot (Eigenvalues)")
        plt.xlabel("Principal Component")
        plt.ylabel("Eigenvalue")
        plt.xticks(np.arange(1, len(eigenvalues) + 1))
        plt.legend()
        plt.grid()
        plt.show()

    def plot_biplot(self):
        """
        Plot a biplot showing PCA results with variable loadings.
        """
        if self.pca_model is None or self.pca_data is None:
            print("PCA has not been performed yet. Call perform_pca() first.")
            return

        components = self.pca_model.components_
        features = self.numeric_data.columns

        plt.figure(figsize=(10, 6))
        plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1], alpha=0.5, c="blue")
        for i, feature in enumerate(features):
            plt.arrow(0, 0, components[0, i] * max(self.pca_data[:, 0]),
                      components[1, i] * max(self.pca_data[:, 1]),
                      color="red", alpha=0.7)
            plt.text(components[0, i] * max(self.pca_data[:, 0]) * 1.1,
                     components[1, i] * max(self.pca_data[:, 1]) * 1.1,
                     feature, color="green", ha="center", va="center")
        plt.title("PCA Biplot")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid()
        plt.show()

    def perform_clustering(self, n_clusters=3, method="kmeans", use_pca=False, **kwargs):
        """
        Perform clustering (KMeans or DBSCAN) on the dataset.

        Args:
            n_clusters (int): Number of clusters for KMeans.
            method (str): Clustering method ("kmeans" or "dbscan").
            use_pca (bool): Whether to use PCA-reduced data for clustering.
            **kwargs: Additional arguments for DBSCAN.

        Returns:
            pd.Series: Cluster labels for each data point.
        """
        data_to_cluster = self.pca_data if use_pca and self.pca_data is not None else self.numeric_data

        if data_to_cluster is None:
            raise ValueError("No data available for clustering. Perform PCA or ensure numeric data is loaded.")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_cluster)

        if method == "kmeans":
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.cluster_labels = kmeans.fit_predict(scaled_data)
            print(f"Cluster Centers (KMeans):\n{kmeans.cluster_centers_}")
        elif method == "dbscan":
            # Perform DBSCAN clustering
            dbscan = DBSCAN(**kwargs)
            self.cluster_labels = dbscan.fit_predict(scaled_data)
            print(f"Number of clusters (DBSCAN): {len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)}")
        else:
            raise ValueError("Invalid clustering method. Choose 'kmeans' or 'dbscan'.")

        # Add cluster labels to the dataset
        self.data["Cluster"] = self.cluster_labels
        return self.cluster_labels

    def plot_silhouette_scores(data, max_clusters=10, use_pca=False):
        """
        Plot silhouette scores for different numbers of clusters.

        Args:
            data (pd.DataFrame): The dataset to cluster.
            max_clusters (int): The maximum number of clusters to evaluate.
            use_pca (bool): Whether to use PCA-reduced data for clustering.
        """
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)

        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            score = silhouette_score(data, cluster_labels)
            silhouette_scores.append(score)
            print(f"Silhouette score for {n_clusters} clusters: {score:.3f}")

        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='--')
        plt.title("Silhouette Scores for Different Numbers of Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.xticks(cluster_range)
        plt.grid(True)
        plt.show()

    def compute_silhouette_score(self, use_pca=False):
        """
        Compute the silhouette score for the current clustering results.

        Args:
            use_pca (bool): Whether to use PCA-reduced data for silhouette score calculation.

        Returns:
            float: Silhouette score of the clustering.
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering has not been performed. Call perform_clustering() first.")
        if len(set(self.cluster_labels)) < 2:
            raise ValueError("Silhouette score cannot be computed for less than 2 clusters.")

        data_for_score = self.pca_data if use_pca and self.pca_data is not None else self.numeric_data

        if data_for_score is None:
            raise ValueError("No data available for silhouette score. Perform PCA or ensure numeric data is loaded.")

        score = silhouette_score(data_for_score, self.cluster_labels)
        print(f"Silhouette Score: {score:.3f}")
        return score

    def plot_clusters(self, use_pca=False, feature_x=None, feature_y=None):
        """
        Plot the clustering results in a 2D scatterplot.

        Args:
            use_pca (bool): Whether to use PCA-reduced data for plotting.
            feature_x (str): The name of the feature for the x-axis (ignored if use_pca is True).
            feature_y (str): The name of the feature for the y-axis (ignored if use_pca is True).
        """
        if "Cluster" not in self.data.columns:
            print("Clustering has not been performed yet. Call perform_clustering() first.")
            return

        if use_pca:
            if self.pca_data is None:
                print("PCA has not been performed. Call perform_pca() first.")
                return
            data_to_plot = self.pca_data
            x_label = "Principal Component 1"
            y_label = "Principal Component 2"
        else:
            # Validate feature selection
            if feature_x is None or feature_y is None:
                print("Please specify feature_x and feature_y for plotting.")
                return
            if feature_x not in self.data.columns or feature_y not in self.data.columns:
                print(f"Features '{feature_x}' and/or '{feature_y}' are not in the dataset.")
                return

            data_to_plot = self.data[[feature_x, feature_y]].values
            x_label = feature_x
            y_label = feature_y

        # Plot the clustering results
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=data_to_plot[:, 0],
            y=data_to_plot[:, 1],
            hue=self.data["Cluster"],
            palette="viridis",
            alpha=0.7
        )
        plt.title(
            "Clustering Results (PCA Reduced Data)" if use_pca else f"Clustering Results ({feature_x} vs {feature_y})")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(title="Cluster")
        plt.grid(True)
        plt.show()

    def plot_lat_long(self, city_name="Map of Houses with Clusters"):
        """
        Plot the locations on an interactive map using longitude and latitude, with cluster visualization.

        Args:
            city_name (str): Title of the plot.
        """
        if "longitude" not in self.data.columns or "latitude" not in self.data.columns:
            print("Longitude or latitude column not found in the dataset.")
            return

        if "Cluster" not in self.data.columns:
            print("Clustering has not been performed yet. Call perform_clustering() first.")
            return

        fig = px.scatter_mapbox(
            self.data,
            lat="latitude",
            lon="longitude",
            color="Cluster",  # Color points by cluster
            hover_data=["median_house_value", "median_income", "Cluster"],  # Add cluster info in hover data
            zoom=8,
            height=600,
            width=800,
            title=city_name
        )

        # Use an open-source map style
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
        fig.show()

