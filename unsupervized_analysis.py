import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'notebook'  # Use 'browser' if not using a notebook


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

    def optimize_kmeans_with_pca(self, max_clusters=10):
        """
        Optimize KMeans clustering using PCA-reduced data by finding the optimal number of clusters.

        Args:
            max_clusters (int): Maximum number of clusters to evaluate.

        Returns:
            int: Optimal number of clusters.
        """
        if self.pca_data is None:
            print("Performing PCA first...")
            self.perform_pca(n_components=3)

        # Use PCA-reduced data
        scaled_data = self.pca_data

        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)

        print("Evaluating silhouette scores for different cluster numbers...")
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(score)
            print(f"Number of clusters: {n_clusters}, Silhouette score: {score:.3f}")

        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, silhouette_scores, marker="o", linestyle="--")
        plt.title("Silhouette Scores for Different Cluster Numbers")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.xticks(cluster_range)
        plt.grid(True)
        plt.show()

        # Determine the optimal number of clusters
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")
        return optimal_clusters

    def perform_kmeans_with_pca(self, n_clusters=None):
        """
        Perform KMeans clustering on PCA-reduced data with an optimized number of clusters.

        Args:
            n_clusters (int): Number of clusters. If None, the optimal number will be determined.
        """
        if self.pca_data is None:
            print("Performing PCA first...")
            self.perform_pca(n_components=3)

        # Optimize number of clusters if not provided
        if n_clusters is None:
            n_clusters = self.optimize_kmeans_with_pca()

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10, max_iter=300)
        self.cluster_labels = kmeans.fit_predict(self.pca_data)
        self.data["Cluster"] = self.cluster_labels

        print(f"Cluster Centers:\n{kmeans.cluster_centers_}")
        print(f"Silhouette Score: {silhouette_score(self.pca_data, self.cluster_labels):.3f}")

        return self.cluster_labels

    def plot_pca_clusters(self):
        """
        Plot the KMeans clusters on the PCA-reduced data.

        """
        if self.pca_data is None:
            print("PCA has not been performed. Call perform_pca() first.")
            return
        if "Cluster" not in self.data.columns:
            print("Clustering has not been performed. Call perform_kmeans_with_pca() first.")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=self.pca_data[:, 0],
            y=self.pca_data[:, 1],
            hue=self.data["Cluster"],
            palette="viridis",
            alpha=0.7
        )
        plt.title("KMeans Clusters on PCA-Reduced Data")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Cluster")
        plt.grid(True)
        plt.show()


    def plot_lat_long(self, city_name="Map of Houses with Clusters"):
        print("Starting plot_lat_long...")
        if "longitude" not in self.data.columns or "latitude" not in self.data.columns:
            print("Longitude or latitude column not found in the dataset.")
            return

        if "Cluster" not in self.data.columns:
            print("Clustering has not been performed yet. Call perform_clustering() first.")
            return

        print("Preparing map...")
        fig = px.scatter_mapbox(
            self.data,
            lat="latitude",
            lon="longitude",
            color="Cluster",
            hover_data=["median_house_value", "median_income", "Cluster"],
            zoom=8,
            height=600,
            width=800,
            title=city_name,
            color_discrete_map={
                0: "red",  # Cluster 0 will be red
                1: "blue",  # Cluster 1 will be blue
                2: "green",  # Cluster 2 will be green
                # Add more mappings as needed
            }
        )

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
        print("Displaying map...")
        fig.show()

    def compare_clusters(self):
        """
        Compare the original features across clusters to understand their differences.
        """
        if "Cluster" not in self.data.columns:
            print("Clustering has not been performed. Call perform_kmeans_with_pca() first.")
            return

        # Group data by clusters and calculate mean for each feature
        cluster_summary = self.data.groupby("Cluster").mean()

        print("Cluster Summary Statistics:")
        display(cluster_summary)

        # Plot feature distributions for each cluster
        for column in self.data.columns:
            if column not in ["Cluster"]:
                plt.figure(figsize=(8, 5))
                sns.boxplot(data=self.data, x="Cluster", y=column, palette="viridis")
                plt.title(f"Distribution of {column} Across Clusters")
                plt.xlabel("Cluster")
                plt.ylabel(column)
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.show()






