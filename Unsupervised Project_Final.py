from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.metrics import confusion_matrix
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import MinMaxScaler
import ruptures as rpt
import matplotlib.dates as mdates


AllCountiesData = pd.read_csv("MTA_Daily_Ridership_Data__2020_-_2025_20250414.csv")
print(AllCountiesData.head())
X_with_dates = AllCountiesData.copy()
AllCountiesData = AllCountiesData.drop(columns=["Date"])
AllCountiesData = AllCountiesData.dropna()

df = AllCountiesData.copy()
X = AllCountiesData.copy()
sns.set_palette("Purples")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def compare_clustering_visualizations(
        df,
        kmeans_n_clusters=4,
        hierarchical_n_clusters=5,
        dbscan_eps=0.6,
        dbscan_min_samples=4,
        gmm_n_components=3,
        spectral_n_clusters=4,
        perplexity=40,
        random_state=42
):
    """
    Compare multiple clustering algorithms across different dimensionality reductions.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input data with features (will be automatically scaled)
    kmeans_n_clusters : int
        Number of clusters for K-means
    hierarchical_n_clusters : int
        Number of clusters for Hierarchical clustering
    dbscan_eps : float
        Epsilon parameter for DBSCAN
    dbscan_min_samples : int
        Minimum samples parameter for DBSCAN
    gmm_n_components : int
        Number of components for Gaussian Mixture Model
    spectral_n_clusters : int
        Number of clusters for Spectral Clustering
    perplexity : int
        Perplexity parameter for t-SNE
    random_state : int
        Random seed for reproducibility
    """

    # === Define clustering algorithms ===
    cluster_algorithms = {
        'K-means': KMeans(n_clusters=kmeans_n_clusters, random_state=random_state),
        'Hierarchical': AgglomerativeClustering(n_clusters=hierarchical_n_clusters,
                                                linkage='ward'),
        'DBSCAN': DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples),
        'GMM': GaussianMixture(n_components=gmm_n_components, random_state=random_state),
        'Spectral': SpectralClustering(n_clusters=spectral_n_clusters,
                                       affinity='nearest_neighbors',
                                       random_state=random_state)
    }

    # === Fit clusterers ===
    cluster_results = {}
    for name, model in cluster_algorithms.items():
        if name == 'GMM':
            cluster_results[name] = model.fit_predict(X_scaled)
        else:
            cluster_results[name] = model.fit_predict(X_scaled)

    # === Dimensionality reduction ===
    reduction_methods = {
        'PCA': PCA(n_components=2).fit_transform(X_scaled),
        't-SNE': TSNE(n_components=2, perplexity=perplexity,
                      random_state=random_state).fit_transform(X_scaled),
        'UMAP': umap.UMAP(random_state=random_state).fit_transform(X_scaled)
    }

    # === Visualization ===
    for reduction_name, X_reduced in reduction_methods.items():
        plt.figure(figsize=(20, 18))
        plt.suptitle(f'Cluster Visualizations on {reduction_name}', fontsize=20)

        for i, (cluster_name, labels) in enumerate(cluster_results.items(), 1):
            plt.subplot(3, 2, i)
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                                  c=labels, cmap='viridis', alpha=0.6)
            plt.title(cluster_name)
            plt.xlabel(f'{reduction_name} 1')
            plt.ylabel(f'{reduction_name} 2')
            if cluster_name == 'DBSCAN':
                plt.colorbar(scatter, ticks=np.unique(labels))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return cluster_results, reduction_methods


cluster_results, reductions = compare_clustering_visualizations(X_scaled)

from sklearn import metrics


def evaluate_clustering_performance(X_scaled, cluster_results, verbose=True):
    """
    Evaluate clustering results using multiple metrics and return a formatted DataFrame.

    Parameters:
    -----------
    X_scaled : array-like
        Scaled feature matrix (n_samples, n_features)
    cluster_results : dict
        Dictionary of clustering results {method_name: cluster_labels}
    verbose : bool
        Whether to print the results table

    Returns:
    --------
    pd.DataFrame
        DataFrame with evaluation metrics for each method
    """

    def _evaluate_single_cluster(X, labels):
        """Helper function for individual cluster evaluation"""
        if len(np.unique(labels)) > 1 and np.all(labels != -1):  # Ignore noise-only/single-cluster
            return {
                'Silhouette': metrics.silhouette_score(X, labels),
                'Davies-Bouldin': metrics.davies_bouldin_score(X, labels),
                'Calinski-Harabasz': metrics.calinski_harabasz_score(X, labels),
                'n_clusters': len(np.unique(labels))
            }
        return {
            'Silhouette': np.nan,
            'Davies-Bouldin': np.nan,
            'Calinski-Harabasz': np.nan,
            'n_clusters': len(np.unique(labels[labels != -1]))  # Exclude noise if present
        }

    # Evaluate all methods
    evaluation_results = []
    for method, labels in cluster_results.items():
        metrics_dict = _evaluate_single_cluster(X_scaled, labels)
        evaluation_results.append({
            'Method': method,
            **metrics_dict
        })

    # Create DataFrame
    eval_df = pd.DataFrame(evaluation_results).set_index('Method')

    # Formatting
    if verbose:
        print("=== Clustering Performance Comparison ===")
        display(eval_df.style
                .format('{:.3f}', subset=['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz'])
                .highlight_max(subset=['Silhouette', 'Calinski-Harabasz'], color='lightgreen')
                .highlight_min(subset=['Davies-Bouldin'], color='lightgreen')
                .set_caption(f"Evaluation Metrics (Higher is better for all except Davies-Bouldin)"))

    return eval_df


metrics_df = evaluate_clustering_performance(X_scaled, cluster_results)

total_cols = [col for col in df.columns if 'Total' in col]
df_totals = df[total_cols].copy()

pct_cols = [col for col in df.columns if '% of Comparable Pre-Pandemic Day' in col]
df_pct = df[pct_cols].copy()


scaler = StandardScaler()
df_totals_scaled = scaler.fit_transform(df_totals)
df_pct_scaled = scaler.fit_transform(df_pct)

total_cluster_results, total_reductions = compare_clustering_visualizations(
    df_totals_scaled,
    kmeans_n_clusters=3,
    hierarchical_n_clusters=5,
    dbscan_eps=0.7,
    dbscan_min_samples=4,
    gmm_n_components=6,
    spectral_n_clusters=4,
    perplexity=40,
    random_state=42
)

total_metrics_df = evaluate_clustering_performance(df_totals_scaled, total_cluster_results)


pct_cluster_results, pct_reductions = compare_clustering_visualizations(
    df_pct_scaled,
    kmeans_n_clusters=4,
    hierarchical_n_clusters=6,
    dbscan_eps=0.9,
    dbscan_min_samples=4,
    gmm_n_components=4,
    spectral_n_clusters=4,
    perplexity=40,
    random_state=42
)

pct_metrics_df = evaluate_clustering_performance(df_pct_scaled, pct_cluster_results)


df_totals['Spectral_Cluster'] = total_cluster_results['Spectral']
#create_radar_charts(df_totals,cluster_col='Spectral_Cluster', features=None)

df_pct['Spectral_Cluster'] = pct_cluster_results['Spectral']
#create_radar_charts(df_pct,cluster_col='Spectral_Cluster', features=None)


from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import table
import matplotlib.colors as mcolors


# Use this colormap for both centroid and contributing difference styling
def get_custom_style(val, vmin, vmax):
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    try:
        rgb = tuple(int(c * 255) for c in custom_cmap(norm(val))[:3])
        return f"background-color: rgb{rgb}; font-weight: bold"
    except:
        return "font-weight: bold"


def plot_kmeans_clusters(data, numerical_columns=None, n_components=2, n_clusters=4, random_state=42):
    """
    Plot K-means clusters with PCA-reduced numerical data and calculate evaluation metrics.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset containing numerical features.
    numerical_columns : list, optional
        List of column names of numerical features. If None, all columns are treated as numerical.
    n_components : int, optional
        Number of dimensions to keep in the PCA output. Default is 2.
    n_clusters : int, optional
        Number of clusters for K-means. Default is 3.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns:
    --------
    tuple
        Reduced data, cluster labels, k-means model, and evaluation metrics.
    """
    # Create a copy of the data to avoid modifying the original
    df = data.copy()

    # If numerical_columns is not specified, use all columns
    if numerical_columns is None:
        numerical_columns = df.columns.tolist()

    # Filter out only the numerical columns that exist in the dataframe
    valid_num_columns = [col for col in numerical_columns if col in df.columns]

    # Check if we have valid numerical columns to process
    if not valid_num_columns:
        raise ValueError("No valid numerical columns found in the data.")

    # Extract numerical features
    X = df[valid_num_columns].values

    # Scale the numerical data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_reduced)

    # Calculate evaluation metrics
    metrics = {}
    metrics['silhouette'] = silhouette_score(X_reduced, labels)
    metrics['davies_bouldin'] = davies_bouldin_score(X_reduced, labels)
    metrics['calinski_harabasz'] = calinski_harabasz_score(X_reduced, labels)

    # Print evaluation metrics
    print("=== Clustering Evaluation Metrics ===")
    print(f"Silhouette Score: {metrics['silhouette']:.4f} (higher is better, range: -1 to 1)")
    print(f"Davies-Bouldin Score: {metrics['davies_bouldin']:.4f} (lower is better)")
    print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.4f} (higher is better)")
    print("")

    # Get the cluster centers
    centers = kmeans.cluster_centers_

    # Create a DataFrame with the reduced data and cluster labels
    result_df = pd.DataFrame(X_reduced, columns=[f'PC{i + 1}' for i in range(n_components)])
    result_df['Cluster'] = labels

    # Calculate explained variance for axis labels
    explained_variance = pca.explained_variance_ratio_ * 100

    # Plot the clusters
    plt.figure(figsize=(10, 8))

    # Create scatter plot with cluster coloring
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=result_df, palette='viridis', s=100, alpha=0.7)

    # Plot cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='X', c='red', label='Cluster Centers')

    # Add labels and title
    plt.title(f'K-means Clustering with {n_clusters} Clusters', fontsize=15)
    plt.xlabel(f'PC1 ({explained_variance[0]:.2f}% explained variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_variance[1]:.2f}% explained variance)', fontsize=12)
    plt.legend(title='Cluster', fontsize=10)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add metrics to the plot
    metrics_text = (f"Silhouette: {metrics['silhouette']:.4f}\n"
                    f"Davies-Bouldin: {metrics['davies_bouldin']:.4f}\n"
                    f"Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")
    plt.annotate(metrics_text, xy=(0.05, 0.05), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Create a 3D plot if n_components >= 3
    if n_components >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create scatter plot with cluster coloring
        scatter = ax.scatter(
            result_df['PC1'],
            result_df['PC2'],
            result_df['PC3'],
            c=labels,
            cmap='viridis',
            s=70,
            alpha=0.8
        )

        # Plot cluster centers
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            s=200,
            marker='X',
            c='red',
            label='Cluster Centers'
        )

        # Add labels and title
        ax.set_title(f'3D K-means Clustering with {n_clusters} Clusters', fontsize=15)
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2f}%)', fontsize=12)
        ax.set_zlabel(f'PC3 ({explained_variance[2]:.2f}%)', fontsize=12)

        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        plt.tight_layout()
        plt.show()

    return X_reduced, labels, kmeans, metrics


# Function to identify most contributing numerical features for each cluster
def identify_contributing_features(data, labels, numerical_columns=None):
    """
    Identify the most contributing numerical features to each cluster.

    Parameters:
    -----------
    data : pandas.DataFrame
        The original dataset.
    labels : numpy.ndarray
        Cluster labels from K-means.
    numerical_columns : list, optional
        List of numerical column names to analyze.

    Returns:
    --------
    dict
        Dictionary containing the top contributing features for each cluster.
    """
    # Create a copy of the data with cluster labels
    df = data.copy()
    df['Cluster'] = labels

    # If numerical_columns is not specified, use all columns
    if numerical_columns is None:
        numerical_columns = [col for col in df.columns if col != 'Cluster']

    # Filter to only include columns that exist in the dataframe
    valid_num_columns = [col for col in numerical_columns if col in df.columns]

    # Get the number of clusters
    n_clusters = len(np.unique(labels))

    # Dictionary to store the results
    contributing_features = {}

    # For each cluster, calculate the feature importance based on:
    # 1. The difference between cluster mean and overall mean
    # 2. Normalized by standard deviation (like a z-score)
    for cluster in range(n_clusters):
        # Initialize a dictionary to store the importance scores for each feature
        feature_importance = {}

        # Get cluster data and overall data
        cluster_data = df[df['Cluster'] == cluster][valid_num_columns]
        overall_data = df[valid_num_columns]

        for feature in valid_num_columns:
            # Calculate cluster mean and overall mean
            cluster_mean = cluster_data[feature].mean()
            overall_mean = overall_data[feature].mean()
            overall_std = overall_data[feature].std()

            # Calculate the absolute difference, normalized by standard deviation
            # This gives a measure of how distinctive this feature is for this cluster
            if overall_std > 0:  # Avoid division by zero
                importance_score = abs(cluster_mean - overall_mean) / overall_std
            else:
                importance_score = 0

            # Store additional statistics for this feature
            feature_importance[feature] = {
                'importance_score': importance_score,
                'cluster_mean': cluster_mean,
                'overall_mean': overall_mean,
                'overall_std': overall_std,
                'difference': cluster_mean - overall_mean,
                'percent_diff': ((cluster_mean - overall_mean) / overall_mean * 100) if overall_mean != 0 else float(
                    'inf')
            }

        # Sort features by importance score (higher means more important)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['importance_score'], reverse=True)

        # Store the sorted features for this cluster
        contributing_features[cluster] = sorted_features

    return contributing_features


# Fix for the centroid heatmap visualization that was causing the error
def analyze_clusters(data, labels, numerical_columns=None, top_n_features=3):
    from matplotlib import cm
    from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
    import seaborn as sns
    import pandas as pd
    import numpy as np

    df = data.copy()
    df['Cluster'] = labels

    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Cluster' in numerical_columns:
            numerical_columns.remove('Cluster')

    valid_num_columns = [col for col in numerical_columns if
                         col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    n_clusters = len(np.unique(labels))

    # --- Cluster size summary ---
    cluster_counts = df['Cluster'].value_counts().sort_index()
    size_df = pd.DataFrame({
        'Cluster': cluster_counts.index,
        'Count': cluster_counts.values,
        'Percentage': [f"{v / len(df) * 100:.1f}%" for v in cluster_counts.values]
    }).set_index('Cluster')
    display(size_df.style.set_caption("Cluster Size Summary").background_gradient(cmap='Purples').set_properties(
        **{'font-weight': 'bold'}))

    # --- Top contributing features ---
    contributing_features = identify_contributing_features(data, labels, valid_num_columns)
    contrib_records = []
    all_diffs = []

    for cluster in range(n_clusters):
        top_features = contributing_features[cluster][:top_n_features]
        for i, (feature, stats) in enumerate(top_features):
            diff_val = stats["difference"]
            all_diffs.append(diff_val)
            contrib_records.append({
                "Cluster": cluster,
                "Rank": i + 1,
                "Feature": feature,
                "Cluster Mean": round(stats["cluster_mean"], 2),
                "Overall Mean": round(stats["overall_mean"], 2),
                "Difference": round(diff_val, 2),
                "Percent Diff": stats['percent_diff']
            })

    contrib_df = pd.DataFrame(contrib_records)

    # Custom colormap: soft red → white → purple
    soft_red = "#f4cccc"
    white = "#ffffff"
    purple = "#7e57c2"
    custom_cmap = LinearSegmentedColormap.from_list("custom_red_purple", [soft_red, white, purple])

    min_diff, max_diff = min(all_diffs), max(all_diffs)
    norm = TwoSlopeNorm(vmin=min_diff, vcenter=0, vmax=max_diff)

    def style_contrib(val):
        try:
            rgb = tuple(int(c * 255) for c in custom_cmap(norm(val))[:3])
            return f"background-color: rgb{rgb}; font-weight: bold"
        except:
            return "font-weight: bold"

    display(
        contrib_df.drop(columns=["Percent Diff"], errors="ignore")  # Optional: drop if not needed
        .style
        .applymap(style_contrib, subset=["Difference"])
        .set_caption("Top Contributing Features by Cluster")
        .format({
            "Cluster Mean": "{:.2f}",
            "Overall Mean": "{:.2f}",
            "Difference": "{:+.2f}"
        })
        .set_properties(**{'font-weight': 'bold'})
    )

    # --- Cluster centroids ---
    centroid_records = []
    all_diffs = []

    for col in valid_num_columns:
        overall_mean = df[col].mean()
        for cluster in range(n_clusters):
            cluster_mean = df[df['Cluster'] == cluster][col].mean()
            diff = cluster_mean - overall_mean
            pct_diff = 100 * diff / overall_mean if overall_mean != 0 else 0
            formatted_val = f"{diff:+.2f} ({pct_diff:+.0f}%)"
            centroid_records.append({
                "Feature": col,
                "Cluster": f"Cluster {cluster}",
                "Value": formatted_val,
                "RawDiff": diff
            })
            all_diffs.append(diff)

    centroid_df = pd.DataFrame(centroid_records)
    pivot_df = centroid_df.pivot(index="Feature", columns="Cluster", values="Value")
    raw_diff = centroid_df.pivot(index="Feature", columns="Cluster", values="RawDiff")

    min_cd, max_cd = min(all_diffs), max(all_diffs)
    norm_cd = TwoSlopeNorm(vmin=min_cd, vcenter=0, vmax=max_cd)

    def centroid_style(val):
        try:
            diff = float(val.split()[0])
            rgb = tuple(int(c * 255) for c in custom_cmap(norm_cd(diff))[:3])
            return f"background-color: rgb{rgb}; font-weight: bold"
        except:
            return "font-weight: bold"

    display(
        pivot_df.style
        .applymap(centroid_style)
        .set_caption("Cluster Centroids (Deviation from Overall Mean)")
        .set_properties(**{'font-weight': 'bold'})
    )

    # --- Summary table: std & count per cluster-feature ---
    summary_records = []
    for col in valid_num_columns:
        overall_std = df[col].std()
        for cluster in range(n_clusters):
            cluster_data = df[df['Cluster'] == cluster][col]
            summary_records.append({
                "Feature": col,
                "Cluster": cluster,
                "Count": cluster_data.shape[0],
                "Cluster Std": round(cluster_data.std(), 2),
                "Overall Std": round(overall_std, 2)
            })

    summary_df = pd.DataFrame(summary_records)
    summary_pivot = summary_df.pivot(index="Feature", columns="Cluster", values=["Count", "Cluster Std"])
    display(
        summary_pivot.style
        .background_gradient(cmap="Purples")
        .set_caption("Feature Summary by Cluster (Count & Std Deviation)")
        .format("{:.2f}")
        .set_properties(**{'font-weight': 'bold'})
    )
    return pivot_df, raw_diff


import matplotlib.colors as mcolors


def export_colored_centroid_table_to_pdf(pivot_df, raw_diff, filename="centroids_table.pdf",
                                         title="Cluster Centroids (Deviation from Overall Mean)"):
    """
    Exports a pivot_df DataFrame to a color-coded PDF table with minimal white space.
    """
    # Calculate tighter figure dimensions
    n_cols = len(pivot_df.columns)
    n_rows = len(pivot_df.index)

    # Adjust these multipliers to control spacing
    col_width = 1.5  # Width per column
    row_height = 0.5  # Height per row
    margin = 0.5  # Small margin

    fig, ax = plt.subplots(figsize=(n_cols * col_width + margin, n_rows * row_height + margin))
    ax.axis('off')

    # Extract short feature names
    short_index = [idx.split(':')[0].strip() for idx in pivot_df.index]

    # Normalize the raw differences for color mapping
    raw_vals = raw_diff.values.astype(float)
    clip_limit = np.percentile(np.abs(raw_vals), 95)
    norm = mcolors.TwoSlopeNorm(vmin=-clip_limit, vcenter=0, vmax=clip_limit)
    cmap = plt.get_cmap("Purples")

    # Build cell text and colors
    cell_text = []
    cell_colors = []
    for i, row in enumerate(pivot_df.index):
        row_text = []
        row_colors = []
        for j, col in enumerate(pivot_df.columns):
            val = pivot_df.loc[row, col]
            raw = np.clip(raw_diff.loc[row, col], -clip_limit, clip_limit)
            row_text.append(str(val))
            rgba = cmap(norm(raw))
            rgba = (rgba[0], rgba[1], rgba[2], 0.6)
            row_colors.append(rgba)
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    # Draw the table with tight layout
    table = ax.table(
        cellText=cell_text,
        rowLabels=short_index,
        colLabels=pivot_df.columns,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Scale the table to fill the figure
    table.scale(1, 1.2)  # Adjust the second parameter if row heights need tweaking

    for key, cell in table.get_celld().items():
        cell.set_text_props(weight='bold')
        if key[0] == 0 or key[1] < 0:  # Header and row label cells
            cell.set_text_props(weight='bold', color='black')

    # Save with minimal padding
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


pct_X_reduced, pct_labels, pkmeans, pmetrics = plot_kmeans_clusters(df_pct, n_components=2, n_clusters=2)
pct_pivot_df, pct_raw_diff = analyze_clusters(df_pct.drop("Spectral_Cluster", axis='columns'), pct_labels)
export_colored_centroid_table_to_pdf(pct_pivot_df, pct_raw_diff, filename="pct_centroids_table.pdf")

totals_X_reduced, totals_labels, tkmeans, tmetrics = plot_kmeans_clusters(df_pct, n_components=2, n_clusters=2)
totals_pivot_df, totals_raw_diff = analyze_clusters(df_totals.drop("Spectral_Cluster", axis='columns'), totals_labels)
export_colored_centroid_table_to_pdf(totals_pivot_df, totals_raw_diff, filename="totals_centroids_table.pdf")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from matplotlib.backends.backend_pdf import PdfPages
import contextlib


def grid_search_clustering(data, numerical_columns=None,
                           use_pca=True,
                           dim_range=range(2, 6),
                           cluster_range=range(2, 8),
                           random_state=42,
                           clustering_method='kmeans'):
    """
    Perform a grid search to find the optimal number of PCA dimensions and clusters
    for numerical data.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset containing numerical features.
    numerical_columns : list, optional
        List of column names of numerical features. If None, all columns are treated as numerical.
    use_pca : bool, optional
        Whether to use PCA dimensionality reduction or not.
    dim_range : range, optional
        Range of dimensions to try if using PCA.
    cluster_range : range, optional
        Range of clusters to try.
    random_state : int, optional
        Random seed for reproducibility.
    clustering_method : str, optional
        Clustering method to use ('kmeans', 'hierarchical', or 'spectral').

    Returns:
    --------
    tuple
        Grid of silhouette scores, Davies-Bouldin scores, and Calinski-Harabasz scores.
    """
    # Process numerical data
    df = data.copy()

    if numerical_columns is None:
        numerical_columns = df.columns.tolist()

    valid_num_columns = [col for col in numerical_columns if col in df.columns]

    if not valid_num_columns:
        raise ValueError("No valid numerical columns found in the data.")

    # Extract and scale numerical features
    X = df[valid_num_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_pca:
        # Initialize result grids based on PCA dimensions
        silhouette_grid = np.zeros((len(dim_range), len(cluster_range)))
        db_grid = np.zeros((len(dim_range), len(cluster_range)))
        ch_grid = np.zeros((len(dim_range), len(cluster_range)))

        # Perform grid search with PCA
        for i, n_components in enumerate(dim_range):
            # Apply PCA with n_components
            pca = PCA(n_components=n_components, random_state=random_state)
            X_reduced = pca.fit_transform(X_scaled)

            for j, n_clusters in enumerate(cluster_range):
                # Apply clustering
                if clustering_method == 'kmeans':
                    model = KMeans(n_clusters=n_clusters, random_state=random_state)
                elif clustering_method == 'hierarchical':
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                elif clustering_method == 'spectral':
                    model = SpectralClustering(n_clusters=n_clusters, random_state=random_state,
                                               affinity='nearest_neighbors')
                else:
                    raise ValueError("Invalid clustering method. Use 'kmeans', 'hierarchical', or 'spectral'.")

                labels = model.fit_predict(X_reduced)

                # Calculate clustering quality metrics if there are at least 2 different clusters
                n_unique_labels = len(np.unique(labels))
                if n_unique_labels >= 2:
                    silhouette_grid[i, j] = silhouette_score(X_reduced, labels)
                    db_grid[i, j] = davies_bouldin_score(X_reduced, labels)
                    ch_grid[i, j] = calinski_harabasz_score(X_reduced, labels)
                else:
                    silhouette_grid[i, j] = float('nan')
                    db_grid[i, j] = float('nan')
                    ch_grid[i, j] = float('nan')
    else:
        # If not using PCA, we'll have a 1D grid (only varying cluster count)
        silhouette_grid = np.zeros(len(cluster_range))
        db_grid = np.zeros(len(cluster_range))
        ch_grid = np.zeros(len(cluster_range))

        # Perform grid search without PCA
        for j, n_clusters in enumerate(cluster_range):
            # Apply clustering
            if clustering_method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=random_state)
            elif clustering_method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            elif clustering_method == 'spectral':
                model = SpectralClustering(n_clusters=n_clusters, random_state=random_state,
                                           affinity='nearest_neighbors')
            else:
                raise ValueError("Invalid clustering method. Use 'kmeans', 'hierarchical', or 'spectral'.")

            labels = model.fit_predict(X_scaled)

            # Calculate clustering quality metrics
            n_unique_labels = len(np.unique(labels))
            if n_unique_labels >= 2:
                silhouette_grid[j] = silhouette_score(X_scaled, labels)
                db_grid[j] = davies_bouldin_score(X_scaled, labels)
                ch_grid[j] = calinski_harabasz_score(X_scaled, labels)
            else:
                silhouette_grid[j] = float('nan')
                db_grid[j] = float('nan')
                ch_grid[j] = float('nan')

    return silhouette_grid, db_grid, ch_grid


def plot_grid_search_results(silhouette_grid, db_grid, ch_grid, dim_range=None, cluster_range=None,
                             clustering_method=None, use_pca=True, pdf=None):
    """
    Plot results of the grid search.

    Parameters:
    -----------
    silhouette_grid : numpy.ndarray
        Grid of silhouette scores.
    db_grid : numpy.ndarray
        Grid of Davies-Bouldin scores.
    ch_grid : numpy.ndarray
        Grid of Calinski-Harabasz scores.
    dim_range : range, optional
        Range of dimensions tried.
    cluster_range : range
        Range of clusters tried.
    clustering_method : str
        Clustering method used ('kmeans', 'hierarchical', or 'spectral').
    use_pca : bool
        Whether PCA was used or not.
    """
    if use_pca:
        # Plot heatmaps for PCA-based results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot silhouette score (higher is better)
        sns.heatmap(silhouette_grid, annot=True, fmt=".3f", cmap="Purples",
                    xticklabels=cluster_range, yticklabels=dim_range, ax=axes[0])
        axes[0].set_title(f'Silhouette Score\n({clustering_method}, higher is better)')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('PCA Dimensions')

        # Find and mark the best silhouette score
        best_silhouette_idx = np.unravel_index(np.nanargmax(silhouette_grid), silhouette_grid.shape)
        axes[0].add_patch(plt.Rectangle((best_silhouette_idx[1], best_silhouette_idx[0]), 1, 1,
                                        fill=False, edgecolor='red', lw=3))

        # Plot Davies-Bouldin score (lower is better)
        sns.heatmap(db_grid, annot=True, fmt=".3f", cmap="Purples_r",
                    xticklabels=cluster_range, yticklabels=dim_range, ax=axes[1])
        axes[1].set_title(f'Davies-Bouldin Score\n({clustering_method}, lower is better)')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('PCA Dimensions')

        # Find and mark the best Davies-Bouldin score
        best_db_idx = np.unravel_index(np.nanargmin(db_grid), db_grid.shape)
        axes[1].add_patch(plt.Rectangle((best_db_idx[1], best_db_idx[0]), 1, 1,
                                        fill=False, edgecolor='red', lw=3))

        # Plot Calinski-Harabasz score (higher is better)
        sns.heatmap(ch_grid, annot=True, fmt=".1f", cmap="Purples",
                    xticklabels=cluster_range, yticklabels=dim_range, ax=axes[2])
        axes[2].set_title(f'Calinski-Harabasz Score\n({clustering_method}, higher is better)')
        axes[2].set_xlabel('Number of Clusters')
        axes[2].set_ylabel('PCA Dimensions')

        # Find and mark the best Calinski-Harabasz score
        best_ch_idx = np.unravel_index(np.nanargmax(ch_grid), ch_grid.shape)
        axes[2].add_patch(plt.Rectangle((best_ch_idx[1], best_ch_idx[0]), 1, 1,
                                        fill=False, edgecolor='red', lw=3))

        plt.tight_layout()
        if pdf is not None:
            pdf.savefig(plt.gcf())
        plt.show()

        # Print the best combinations
        best_dim_sil, best_clust_sil = dim_range[best_silhouette_idx[0]], cluster_range[best_silhouette_idx[1]]
        best_dim_db, best_clust_db = dim_range[best_db_idx[0]], cluster_range[best_db_idx[1]]
        best_dim_ch, best_clust_ch = dim_range[best_ch_idx[0]], cluster_range[best_ch_idx[1]]

        print(f"Best combination by Silhouette score: {best_dim_sil} dimensions, {best_clust_sil} clusters")
        print(f"Best combination by Davies-Bouldin score: {best_dim_db} dimensions, {best_clust_db} clusters")
        print(f"Best combination by Calinski-Harabasz score: {best_dim_ch} dimensions, {best_clust_ch} clusters")

    else:
        # Plot line charts for non-PCA results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot silhouette score (higher is better)
        axes[0].plot(cluster_range, silhouette_grid, marker='o', linestyle='-')
        axes[0].set_title(f'Silhouette Score\n({clustering_method}, higher is better)')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Score')
        axes[0].grid(True)

        # Find and mark the best silhouette score
        best_silhouette_idx = np.nanargmax(silhouette_grid)
        axes[0].axvline(x=cluster_range[best_silhouette_idx], color='#4B0082', linestyle='--')

        # Plot Davies-Bouldin score (lower is better)
        axes[1].plot(cluster_range, db_grid, marker='o', linestyle='-')
        axes[1].set_title(f'Davies-Bouldin Score\n({clustering_method}, lower is better)')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('Score')
        axes[1].grid(True)

        # Find and mark the best Davies-Bouldin score
        best_db_idx = np.nanargmin(db_grid)
        axes[1].axvline(x=cluster_range[best_db_idx], color='#4B0082', linestyle='--')

        # Plot Calinski-Harabasz score (higher is better)
        axes[2].plot(cluster_range, ch_grid, marker='o', linestyle='-')
        axes[2].set_title(f'Calinski-Harabasz Score\n({clustering_method}, higher is better)')
        axes[2].set_xlabel('Number of Clusters')
        axes[2].set_ylabel('Score')
        axes[2].grid(True)

        # Find and mark the best Calinski-Harabasz score
        best_ch_idx = np.nanargmax(ch_grid)
        axes[2].axvline(x=cluster_range[best_ch_idx], color='#4B0082', linestyle='--')

        plt.tight_layout()
        if pdf is not None:
            pdf.savefig(plt.gcf())
        plt.show()

        # Print the best number of clusters
        best_clust_sil = cluster_range[best_silhouette_idx]
        best_clust_db = cluster_range[best_db_idx]
        best_clust_ch = cluster_range[best_ch_idx]

        print(f"Best number of clusters by Silhouette score: {best_clust_sil}")
        print(f"Best number of clusters by Davies-Bouldin score: {best_clust_db}")
        print(f"Best number of clusters by Calinski-Harabasz score: {best_clust_ch}")


def run_grid_search(X, numerical_columns=None, pdf_filename="grid_search_plots.pdf",
                    log_filename="grid_search_output.txt"):
    dim_range = range(2, 6)
    cluster_range = range(2, 11)

    with PdfPages(pdf_filename) as pdf, open(log_filename, 'w') as log_file, contextlib.redirect_stdout(log_file):
        # K-means with PCA
        print("\n=== K-means Clustering Grid Search with PCA ===")
        sil, db, ch = grid_search_clustering(X, numerical_columns, True, dim_range, cluster_range,
                                             clustering_method='kmeans')
        plot_grid_search_results(sil, db, ch, dim_range, cluster_range, 'K-means', True, pdf)

        print("\n=== K-means Clustering Grid Search without PCA ===")
        sil, db, ch = grid_search_clustering(X, numerical_columns, False, cluster_range=cluster_range,
                                             clustering_method='kmeans')
        plot_grid_search_results(sil, db, ch, cluster_range=cluster_range, clustering_method='K-means', use_pca=False,
                                 pdf=pdf)

        # Hierarchical
        print("\n=== Hierarchical Clustering Grid Search with PCA ===")
        sil, db, ch = grid_search_clustering(X, numerical_columns, True, dim_range, cluster_range,
                                             clustering_method='hierarchical')
        plot_grid_search_results(sil, db, ch, dim_range, cluster_range, 'Hierarchical', True, pdf)

        print("\n=== Hierarchical Clustering Grid Search without PCA ===")
        sil, db, ch = grid_search_clustering(X, numerical_columns, False, cluster_range=cluster_range,
                                             clustering_method='hierarchical')
        plot_grid_search_results(sil, db, ch, cluster_range=cluster_range, clustering_method='Hierarchical',
                                 use_pca=False, pdf=pdf)

        # Spectral
        print("\n=== Spectral Clustering Grid Search with PCA ===")
        sil, db, ch = grid_search_clustering(X, numerical_columns, True, dim_range, cluster_range,
                                             clustering_method='spectral')
        plot_grid_search_results(sil, db, ch, dim_range, cluster_range, 'Spectral', True, pdf)

        print("\n=== Spectral Clustering Grid Search without PCA ===")
        sil, db, ch = grid_search_clustering(X, numerical_columns, False, cluster_range=cluster_range,
                                             clustering_method='spectral')
        plot_grid_search_results(sil, db, ch, cluster_range=cluster_range, clustering_method='Spectral', use_pca=False,
                                 pdf=pdf)


# Example usage:
# Assuming X is your numerical dataframe
run_grid_search(df_pct.drop("Spectral_Cluster", axis='columns'), pdf_filename="pct_grid_search_plots.pdf")
run_grid_search(df_totals.drop("Spectral_Cluster", axis='columns'), pdf_filename="totals_grid_search_plots.pdf")

from scipy.spatial.distance import cdist
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages


def kmeans_anomaly_detection_with_analysis(data, numerical_columns=None, n_components=2, n_clusters=2,
                                           threshold_percentile=95, random_state=42, agglo_clusters=2, pdf_path=None):
    """
    K-means anomaly detection with feature ranking, hierarchical clustering, and PCA explanation.
    """
    df = data.copy()

    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    valid_columns = [col for col in numerical_columns if col in df.columns]
    if not valid_columns:
        raise ValueError("No valid numerical columns found.")

    # Standardize
    X = df[valid_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_reduced)

    # Distances and Anomalies
    distances = np.min(cdist(X_reduced, kmeans.cluster_centers_), axis=1)
    threshold = np.percentile(distances, threshold_percentile)
    anomalies = distances > threshold
    df['is_anomaly'] = anomalies

    # Agglomerative clustering on anomalies
    aglo_labels = np.full(len(df), fill_value=-1)  # default label for non-anomalies
    if anomalies.sum() > 1:
        agglo = AgglomerativeClustering(n_clusters=agglo_clusters)
        aglo_labels[anomalies] = agglo.fit_predict(X_reduced[anomalies])

    df['anomaly_group'] = aglo_labels

    # PCA Component Interpretation
    pca_components_df = pd.DataFrame(pca.components_, columns=valid_columns,
                                     index=[f'PC{i + 1}' for i in range(n_components)])
    print("\n=== PCA Component Weights ===")
    display(pca_components_df.T)

    # Visualize PCA contributions
    plt.figure(figsize=(10, 5))
    sns.heatmap(pca_components_df.T, annot=True, cmap='Purples', center=0)
    plt.title("PCA Component Contributions by Feature")
    plt.show()

    # Create PDF if path provided
    if pdf_path:
        pdf = PdfPages(pdf_path)
    else:
        pdf = None

    # Clusters & Anomalies Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scatter = axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='Purples', s=50, alpha=0.6,
                              edgecolors='k')
    axes[0].scatter(X_reduced[anomalies, 0], X_reduced[anomalies, 1], s=200, facecolors='none', edgecolors='#5e3c99',
                    linewidths=2)
    axes[0].set_title(f'K-means Clustering & Anomalies (Threshold: {threshold_percentile}th)')
    axes[0].legend(handles=[
        *scatter.legend_elements()[0],
        Line2D([0], [0], marker='o', color='w', label='Anomaly',
               markerfacecolor='none', markeredgecolor='#5e3c99', markersize=10)
    ])
    axes[0].set_xlabel('PCA 1')
    axes[0].set_ylabel('PCA 2')

    # Histogram of distances
    axes[1].hist(distances, bins=30, alpha=0.7, color='#a9a9f5')
    axes[1].axvline(x=threshold, color='#5e3c99', linestyle='--', label=f'Threshold ({threshold:.3f})')
    axes[1].set_title('Distance to Cluster Center')
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # === Summary Stats ===
    print(f"Total points: {len(df)}")
    print(f"Anomalies detected: {anomalies.sum()} ({anomalies.mean() * 100:.2f}%)")
    print(f"Distance threshold: {threshold:.3f}")

    print("\nAnomalies per original cluster:")
    for cluster in range(n_clusters):
        mask = cluster_labels == cluster
        anomaly_count = (anomalies & mask).sum()
        total = mask.sum()
        print(
            f"  Cluster {cluster}: {anomaly_count}/{total} ({(anomaly_count / total * 100) if total else 0:.2f}%) anomalies")

    # === Feature Ranking ===
    feature_ranking = []
    for col in valid_columns:
        anomaly_mean = df[df['is_anomaly']][col].mean()
        normal_mean = df[~df['is_anomaly']][col].mean()
        diff = anomaly_mean - normal_mean
        std = df[col].std()
        std_diff = diff / std if std else 0
        feature_ranking.append((col, anomaly_mean, normal_mean, diff, std_diff))

    feature_df = pd.DataFrame(feature_ranking,
                              columns=['Feature', 'Anomaly Mean', 'Normal Mean', 'Mean Difference', 'Std. Diff'])
    feature_df = feature_df.sort_values(by='Std. Diff', key=abs, ascending=False)
    print("\n=== Ranked Features (By Std. Difference) ===")
    display(feature_df)

    # KDE Plots of Features
    n_cols = 3
    n_rows = (len(valid_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(valid_columns):
        sns.kdeplot(df[~df['is_anomaly']][col], label='Normal', ax=axes[i], fill=True, color='#8da0cb')
        sns.kdeplot(df[df['is_anomaly']][col], label='Anomaly', ax=axes[i], fill=True, color='#5e3c99')
        axes[i].set_title(col)
        axes[i].legend()

    for j in range(len(valid_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if pdf:
        pdf.savefig(fig)
    plt.show()

    # Grouped anomaly plots
    if agglo_clusters > 1 and anomalies.sum() > 0:
        fig_group = plt.figure(figsize=(8, 6))
        palette = ['#5e3c99', '#a9a9f5']
        sns.scatterplot(
            x=X_reduced[anomalies][:, 0],
            y=X_reduced[anomalies][:, 1],
            hue=df.loc[anomalies, 'anomaly_group'],
            palette=palette,
            s=100)
        plt.title("Agglomerative Clustering of Anomalies")
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend(title='Anomaly Group')

        if pdf:
            pdf.savefig(fig_group)
        plt.show()

    if pdf:
        pdf.close()

    return {
        'X_reduced': X_reduced,
        'cluster_labels': cluster_labels,
        'distances': distances,
        'anomalies': anomalies,
        'threshold': threshold,
        'feature_ranking': feature_df,
        'pca_components': pca_components_df,
        'agglomerative_labels': aglo_labels
    }


results_pct = kmeans_anomaly_detection_with_analysis(df_pct.drop("Spectral_Cluster", axis='columns'),
                                                     threshold_percentile=97,
                                                     pdf_path="pct_normal_vs_anomaly_plots.pdf")
results_totals = kmeans_anomaly_detection_with_analysis(df_totals.drop("Spectral_Cluster", axis='columns'),
                                                        threshold_percentile=97,
                                                        pdf_path="totals_normal_vs_anomaly_plots.pdf")


def plot_feature_distributions(df, results=None, title="Feature Distributions",
                               n_cols=3, figsize=(15, 4), pdf_output=None):
    """
    Plot KDEs for each feature using ALL data (both normal and anomalous).

    Parameters:
    - df: Original DataFrame.
    - results: Optional dictionary (unused here, kept for backward compatibility).
    - title: Title for the figure.
    - n_cols: Number of columns in the subplot grid.
    - figsize: Figure size per row (width, height).
    - pdf_output: Path to save the plot as PDF (optional).
    """
    # Use all columns if 'results' is not provided, otherwise use ranked features
    if results is not None:
        valid_columns = results['feature_ranking']['Feature'].tolist()
    else:
        valid_columns = df.columns.tolist()

    n_rows = (len(valid_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(valid_columns):
        sns.kdeplot(df[col], fill=True, ax=axes[i], color='#8da0cb')  # Plot ALL data
        axes[i].set_title(col)
        axes[i].set_ylabel('Density')
        axes[i].set_xlabel(col)

    # Remove empty subplots
    for j in range(len(valid_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if pdf_output:
        plt.savefig(pdf_output, format='pdf', bbox_inches='tight')
    plt.show()


plot_feature_distributions(df_pct.drop("Spectral_Cluster", axis='columns'), title="Distributions (Percentage Features)",
                           pdf_output="pct_distribution.pdf")
plot_feature_distributions(df_totals.drop("Spectral_Cluster", axis='columns'), title="Distributions (Total Features)",
                           pdf_output="totals_distribution.pdf")


def analyze_anomaly_clusters_with_normals(results, data, pdf_path=None):
    """
    Analyze anomaly clusters identified by Agglomerative Clustering,
    including comparison to normal (non-anomalous) behavior.
    Parameters:
    -----------
    results : dict
        Output dictionary from kmeans_anomaly_detection_with_analysis function.
    data : pandas.DataFrame
        Original input DataFrame before PCA or clustering.
    pdf_path : str, optional
        Path to save plots as PDF. If None, plots are only displayed.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    # === Custom Colors ===
    normal_color = '#4879cf'  # Light blue
    purple_palette = ['#5e3c99', '#a05195', '#9e9ac8', '#dadaeb', '#c2a5cf']

    # Initialize PDF if path is provided
    if pdf_path:
        pdf = PdfPages(pdf_path)
    else:
        pdf = None

    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    # Prepare dataframe with anomaly info
    df_all = data.copy()
    df_all['is_anomaly'] = results['anomalies']
    df_all['anomaly_group'] = results['agglomerative_labels']
    df_all['anomaly_group'] = df_all['anomaly_group'].where(df_all['is_anomaly'], -1)
    anomalies_df = df_all[df_all['is_anomaly']]
    normal_df = df_all[~df_all['is_anomaly']]

    print("=== Data Breakdown ===")
    print(df_all['anomaly_group'].value_counts().rename_axis("Group").to_frame("Count"), "\n")

    # Ranked features
    numerical_columns = results['feature_ranking']['Feature'].tolist()

    # Mean comparison: group-wise + normal
    group_summary = df_all.groupby('anomaly_group')[numerical_columns].mean().T.round(4)
    group_diff = group_summary.sub(group_summary[-1], axis=0).drop(columns=-1).round(4)

    print("=== Mean Feature Values (Grouped by Anomaly Group + Normal) ===")
    display(group_summary)

    print("\n=== Difference from Normal Behavior ===")
    display(group_diff)

    # Heatmap of means
    fig1 = plt.figure(figsize=(10, 6))
    sns.heatmap(group_summary, annot=True, cmap='Purples', center=0, fmt=".4f")
    plt.title("Mean Feature Values per Anomaly Group (Including Normal)")
    plt.ylabel("Feature")
    plt.xlabel("Group (-1 = Normal)")
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig1)
    plt.show()

    # Heatmap of differences
    fig2 = plt.figure(figsize=(10, 6))
    sns.heatmap(group_diff, annot=True, cmap='BuPu', center=0, fmt=".4f")
    plt.title("Feature Deviations from Normal (Normal = 0)")
    plt.ylabel("Feature")
    plt.xlabel("Anomaly Group")
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig2)
    plt.show()

    # KDE plots comparing groups with normal
    print("\n=== Feature Distribution per Group (with Normal) ===")
    n_cols = 3
    n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    unique_groups = sorted(anomalies_df['anomaly_group'].unique())
    palette = {group: purple_palette[i % len(purple_palette)] for i, group in enumerate(unique_groups)}

    for i, col in enumerate(numerical_columns):
        sns.kdeplot(normal_df[col], label='Normal', ax=axes[i], color=normal_color, linestyle='--', fill=True)
        for group in unique_groups:
            sns.kdeplot(anomalies_df[anomalies_df['anomaly_group'] == group][col],
                        label=f'Anomaly Group {group}',
                        ax=axes[i],
                        color=palette[group],
                        fill=True)
        axes[i].set_title(col)
        axes[i].legend()

    for j in range(len(numerical_columns), len(axes)):
        fig3.delaxes(axes[j])

    plt.tight_layout()
    if pdf:
        pdf.savefig(fig3)
    plt.show()

    if pdf:
        pdf.close()


# Example usage:
analyze_anomaly_clusters_with_normals(
    results_pct,
    data=df_pct.drop("Spectral_Cluster", axis='columns'),
    pdf_path="pct_anomaly_clusters_analysis.pdf"
)

analyze_anomaly_clusters_with_normals(
    results_totals,
    data=df_totals.drop("Spectral_Cluster", axis='columns'),
    pdf_path="totals_anomaly_clusters_analysis.pdf"
)


from matplotlib.backends.backend_pdf import PdfPages

def plot_tsne_with_clusters_and_anomalies(data, results, numerical_columns=None, perplexity=40, random_state=42,
                                          figsize=(10, 7), title="t-SNE on Original Data with Clusters & Anomalies",
                                          pdf_path=None):
    """
    Plot t-SNE with cluster boundaries and anomalies overlaid. Optionally save to PDF.
    """
    df = data.copy()

    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Standardize original data
    X = df[numerical_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run t-SNE on standardized original data
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)

    # Combine with results
    anomalies = results['anomalies']
    groups = results['agglomerative_labels']
    clusters = results['cluster_labels']

    tsne_df = pd.DataFrame(X_tsne, columns=["TSNE 1", "TSNE 2"])
    tsne_df["Anomaly"] = anomalies
    tsne_df["Anomaly Group"] = groups
    tsne_df["Cluster"] = clusters

    # Plotting
    fig = plt.figure(figsize=figsize)

    # Plot background cluster areas using KDE
    for clust in np.unique(clusters):
        sns.kdeplot(
            data=tsne_df[tsne_df["Cluster"] == clust],
            x="TSNE 1", y="TSNE 2",
            fill=True, alpha=0.12, linewidth=0,
            cmap=sns.light_palette("purple", as_cmap=True)
        )

    # Plot normal points
    sns.scatterplot(
        data=tsne_df[~tsne_df["Anomaly"]],
        x="TSNE 1", y="TSNE 2",
        hue="Cluster",
        palette="Purples",
        style=None,
        s=40,
        alpha=0.5,
        edgecolor='none',
        legend=False
    )

    # Plot anomaly points overlaid
    palette_anom = sns.color_palette("dark:#5e3c99", n_colors=len(set(groups)) - (1 if -1 in groups else 0))
    sns.scatterplot(
        data=tsne_df[tsne_df["Anomaly"]],
        x="TSNE 1", y="TSNE 2",
        hue="Anomaly Group",
        palette=palette_anom,
        s=100,
        edgecolor='black',
        linewidth=0.6,
        alpha=0.95,
        marker='X',
        legend=True
    )

    plt.title(title)
    plt.legend(title="Anomaly Group", loc="best")
    plt.tight_layout()

    # Save to PDF if path provided
    if pdf_path:
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        print(f"t-SNE plot saved to: {pdf_path}")

    plt.show()

plot_tsne_with_clusters_and_anomalies(df_pct.drop("Spectral_Cluster", axis=1), results_pct, title="PCT - t-SNE (Original Features)")
plot_tsne_with_clusters_and_anomalies(df_totals.drop("Spectral_Cluster", axis=1), results_totals, title="Totals - t-SNE (Original Features)")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime


def multivariate_changepoint_detection(df, date_col, feature_cols, n_bkps=3, model="l2"):
    """
    Perform multivariate changepoint detection on multiple columns together.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    date_col (str): Name of the date column
    feature_cols (list): List of column names to analyze together
    n_bkps (int): Number of breakpoints to detect
    model (str): Cost model for ruptures (l1, l2, rbf, etc.)

    Returns:
    dict: Dictionary containing changepoint information
    """
    print(f"Running multivariate changepoint detection on {len(feature_cols)} features")

    # Check that we have at least one feature
    if not feature_cols:
        print("Error: No features provided for multivariate analysis")
        return {'indices': [], 'dates': [], 'values': {}}

    # Extract features and handle missing values
    X = df[feature_cols].copy()

    # Fill missing values
    X = X.fillna(method='ffill').fillna(method='bfill')

    # Standardize the features so they have equal weight
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply changepoint detection to multivariate data
    print("Applying changepoint detection algorithm...")
    algo = rpt.Dynp(model=model, min_size=10).fit(X_scaled)
    changepoints = algo.predict(n_bkps=n_bkps)

    # Remove the last index if it's just the length of the series
    if changepoints and changepoints[-1] == len(X_scaled):
        changepoints = changepoints[:-1]

    # Make sure we have valid changepoints
    if not changepoints:
        print("No valid changepoints found in multivariate analysis")
        return {'indices': [], 'dates': [], 'values': {}}

    # Convert indices to dates
    changepoint_dates = df[date_col].iloc[changepoints].tolist()

    # Create a dictionary of values at changepoints for each feature
    values_dict = {}
    for col in feature_cols:
        values_dict[col] = [df[col].iloc[idx] for idx in changepoints]

    result = {
        'indices': changepoints,
        'dates': changepoint_dates,
        'values': values_dict
    }

    print(f"Found {len(changepoints)} multivariate changepoints")
    return result


def visualize_multivariate_changepoints(df, date_col, feature_cols, changepoints, title, cmap='Purples',
                                        max_features=None):
    """
    Visualize multivariate changepoints across multiple features.

    Parameters:
    df (pd.DataFrame): DataFrame with date column and features
    date_col (str): Name of the date column
    feature_cols (list): List of column names to analyze together
    changepoints (dict): Dictionary with changepoint indices and dates
    title (str): Title for the plot
    cmap (str): Colormap to use
    max_features (int): Maximum number of features to display (None = all features)
    """
    # If max_features is set, limit the number of features to display
    if max_features and len(feature_cols) > max_features:
        print(f"Limiting visualization to first {max_features} features (out of {len(feature_cols)})")
        feature_cols = feature_cols[:max_features]
    else:
        print(f"Visualizing all {len(feature_cols)} features")

    # Get colormap with enough colors
    colors = sns.color_palette(cmap, len(feature_cols))

    # Create plot with subplots for each feature
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(15, 3 * len(feature_cols)), sharex=True)

    # Handle case of single feature
    if len(feature_cols) == 1:
        axes = [axes]

    # Plot each feature in its own subplot
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        ax.plot(df[date_col], df[col], color=colors[i], label=col, linewidth=2)
        ax.set_title(col, fontsize=12)
        ax.legend(loc='upper right')

        # Add vertical lines for changepoints
        for idx in changepoints['indices']:
            if 0 <= idx < len(df):
                date_val = df[date_col].iloc[idx]
                ax.axvline(x=date_val, color='red', linestyle='--', alpha=0.7)

                # Add annotation with date on the first subplot only to avoid clutter
                if i == 0:
                    ax.annotate(date_val.strftime('%Y-%m-%d'),
                                xy=(date_val, df[col].iloc[idx]),
                                xytext=(10, 0), textcoords='offset points',
                                rotation=45, ha='left', va='bottom',
                                color='red', fontsize=10)

    # Format x-axis dates on the bottom subplot
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].set_xlabel('Date', fontsize=12)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)

    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle

    return fig


def summarize_multivariate_changepoints(df, date_col, feature_cols, changepoints):
    """
    Print a summary of detected multivariate changepoints with dates and values.
    """
    summary_text = []

    if not changepoints['indices']:
        print("\nNo multivariate changepoints detected")
        summary_text.append("No multivariate changepoints detected")
        return summary_text

    summary_text.append("\nMultivariate Changepoints:")
    summary_text.append("=" * 60)

    print("\nMultivariate Changepoints:")
    print("=" * 60)

    for i, (idx, date) in enumerate(zip(changepoints['indices'], changepoints['dates'])):
        print(f"\nChangepoint {i + 1}: {date.strftime('%Y-%m-%d')}")
        summary_text.append(f"\nChangepoint {i + 1}: {date.strftime('%Y-%m-%d')}")

        # Calculate the segment ranges for before and after
        if i == 0:
            start_idx = 0
        else:
            start_idx = changepoints['indices'][i - 1]

        end_idx = idx
        next_idx = changepoints['indices'][i + 1] if i + 1 < len(changepoints['indices']) else len(df)

        print("-" * 40)
        print("Feature | Value | Avg Before | Avg After | % Change")
        print("-" * 40)

        summary_text.append("-" * 40)
        summary_text.append("Feature | Value | Avg Before | Avg After | % Change")
        summary_text.append("-" * 40)

        # Show the impact on each feature
        for col in feature_cols:
            segment_before = df[col].iloc[start_idx:end_idx].mean()
            segment_after = df[col].iloc[end_idx:next_idx].mean()

            # Calculate percent change if possible
            if segment_before != 0:
                pct_change = ((segment_after - segment_before) / abs(segment_before)) * 100
                pct_change_str = f"{pct_change:.2f}%"
            else:
                pct_change_str = "N/A"

            # Get value at the changepoint
            value = df[col].iloc[idx]

            # Print the summary for this feature
            feature_summary = f"{col[:15]:15} | {value:.2f} | {segment_before:.2f} | {segment_after:.2f} | {pct_change_str}"
            print(feature_summary)
            summary_text.append(feature_summary)

    return summary_text


def create_summary_figure(summary_text, title):
    """
    Create a figure with text summary for inclusion in PDF
    """
    fig = plt.figure(figsize=(15, len(summary_text) * 0.4))
    plt.axis('off')
    plt.title(title, fontsize=16)

    summary_str = '\n'.join(summary_text)
    plt.text(0.05, 0.95, summary_str, transform=fig.transFigure,
             fontsize=10, va='top', family='monospace')

    return fig


def run_multivariate_analysis_with_pdf(X_with_dates, date_col, n_bkps=3, output_dir='.'):
    """
    Run multivariate changepoint detection with 7-day rolling average and save plots to PDF.
    """
    print("\n" + "=" * 80)
    print("STARTING MULTIVARIATE CHANGEPOINT ANALYSIS")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = os.path.join(output_dir, f"mta_changepoint_analysis_{timestamp}.pdf")
    pdf = PdfPages(pdf_filename)

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(X_with_dates[date_col]):
        print(f"Converting dates from {type(X_with_dates[date_col].iloc[0])} to datetime")
        X_with_dates[date_col] = pd.to_datetime(X_with_dates[date_col])

    # Sort and apply 7-day rolling average (exclude date column)
    print("\nApplying 7-day rolling average to numeric columns")
    X_sorted = X_with_dates.sort_values(by=date_col).reset_index(drop=True)
    X_smoothed = X_sorted.copy()
    numeric_cols = X_smoothed.select_dtypes(include=[np.number]).columns.tolist()

    X_smoothed[numeric_cols] = X_smoothed[numeric_cols].rolling(window=7, center=True).mean()

    # Drop rows with NaNs from rolling window
    X_smoothed.dropna(inplace=True)
    print(f"Remaining rows after rolling average: {X_smoothed.shape[0]}")

    # ========== Splitting data ==========
    print("\nStep 1: Splitting data into totals and percentages")
    total_cols = [col for col in X_smoothed.columns if
                  ('total' in col.lower() or 'ridership' in col.lower()) and 'percent' not in col.lower()]
    pct_cols = [col for col in X_smoothed.columns if 'percent' in col.lower() or 'pct' in col.lower()]

    if not pct_cols:
        print("No 'percent' columns found. Using remaining numeric columns.")
        pct_cols = [col for col in numeric_cols if col not in total_cols and col != date_col]

    totals_with_dates = X_smoothed[[date_col] + total_cols].copy()
    pct_with_dates = X_smoothed[[date_col] + pct_cols].copy()

    print(f"Totals dataframe: {totals_with_dates.shape[0]} rows, {totals_with_dates.shape[1]} columns")
    print(f"Percentage dataframe: {pct_with_dates.shape[0]} rows, {pct_with_dates.shape[1]} columns")

    # Add title page to PDF
    fig_title = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.5, 0.7, "MTA Ridership Changepoint Analysis",
             ha='center', fontsize=24, fontweight='bold')
    plt.text(0.5, 0.6, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}",
             ha='center', fontsize=16)
    plt.text(0.5, 0.5,
             f"Data Period: {X_with_dates[date_col].min().strftime('%Y-%m-%d')} to {X_with_dates[date_col].max().strftime('%Y-%m-%d')}",
             ha='center', fontsize=16)
    plt.text(0.5, 0.4, f"Total Features: {len(total_cols)}", ha='center', fontsize=16)
    plt.text(0.5, 0.35, f"Percentage Features: {len(pct_cols)}", ha='center', fontsize=16)
    plt.text(0.5, 0.25, f"Number of Changepoints Detected: {n_bkps}", ha='center', fontsize=16)
    pdf.savefig(fig_title)
    plt.close(fig_title)

    # ===========================================================================
    # MULTIVARIATE ANALYSIS OF TOTAL RIDERSHIP
    # ===========================================================================
    print("\n" + "=" * 80)
    print("MULTIVARIATE ANALYSIS OF TOTAL RIDERSHIP")
    print("=" * 80)

    print("\nStep 2: Selecting columns for total ridership analysis")
    total_cols_to_analyze = [col for col in totals_with_dates.columns if col != date_col]
    print(f"Columns to analyze: {total_cols_to_analyze}")

    print("\nStep 3: Running multivariate changepoint detection on total ridership")
    total_changepoints = multivariate_changepoint_detection(totals_with_dates, date_col, total_cols_to_analyze,
                                                            n_bkps=n_bkps)

    print("\nStep 4: Visualizing total ridership multivariate changepoints")
    # Use all columns for the multivariate plot - NO LIMIT
    fig_total = visualize_multivariate_changepoints(totals_with_dates, date_col, total_cols_to_analyze,
                                                    total_changepoints,
                                                    "Multivariate Changepoints in Total Ridership",
                                                    cmap='Purples',
                                                    max_features=None)  # Explicitly set to None to plot all features
    pdf.savefig(fig_total)
    plt.close(fig_total)

    print("\nStep 5: Summarizing total ridership multivariate changepoints")
    total_summary = summarize_multivariate_changepoints(totals_with_dates, date_col, total_cols_to_analyze,
                                                        total_changepoints)

    # Add summary to PDF
    fig_total_summary = create_summary_figure(total_summary, "Total Ridership Changepoint Summary")
    pdf.savefig(fig_total_summary)
    plt.close(fig_total_summary)

    # ===========================================================================
    # MULTIVARIATE ANALYSIS OF PERCENTAGE RIDERSHIP
    # ===========================================================================
    print("\n" + "=" * 80)
    print("MULTIVARIATE ANALYSIS OF PERCENTAGE RIDERSHIP")
    print("=" * 80)

    print("\nStep 6: Selecting columns for percentage ridership analysis")
    pct_cols_to_analyze = [col for col in pct_with_dates.columns if col != date_col]
    print(f"Columns to analyze: {pct_cols_to_analyze}")

    print("\nStep 7: Running multivariate changepoint detection on percentage ridership")
    pct_changepoints = multivariate_changepoint_detection(pct_with_dates, date_col, pct_cols_to_analyze, n_bkps=n_bkps)

    print("\nStep 8: Visualizing percentage ridership multivariate changepoints")
    # Use all columns for the multivariate plot - NO LIMIT
    fig_pct = visualize_multivariate_changepoints(pct_with_dates, date_col, pct_cols_to_analyze,
                                                  pct_changepoints,
                                                  "Multivariate Changepoints in Percent of Pre-COVID Ridership",
                                                  cmap='Purples',
                                                  max_features=None)  # Explicitly set to None to plot all features
    pdf.savefig(fig_pct)
    plt.close(fig_pct)

    print("\nStep 9: Summarizing percentage ridership multivariate changepoints")
    pct_summary = summarize_multivariate_changepoints(pct_with_dates, date_col, pct_cols_to_analyze, pct_changepoints)

    # Add summary to PDF
    fig_pct_summary = create_summary_figure(pct_summary, "Percentage Ridership Changepoint Summary")
    pdf.savefig(fig_pct_summary)
    plt.close(fig_pct_summary)

    # Create a combined visualization to compare the two datasets' changepoints
    print("\nStep 10: Creating combined visualization of multivariate changepoints")

    # Select a few representative columns from each dataset
    # For combined plot, we'll still limit to avoid overcrowding
    total_rep_cols = total_cols_to_analyze[:3] if len(total_cols_to_analyze) > 3 else total_cols_to_analyze
    pct_rep_cols = pct_cols_to_analyze[:3] if len(pct_cols_to_analyze) > 3 else pct_cols_to_analyze

    # Create the combined plot
    fig_combined = plt.figure(figsize=(15, 10))
    gs = fig_combined.add_gridspec(2, 1)
    ax1 = fig_combined.add_subplot(gs[0, 0])
    ax2 = fig_combined.add_subplot(gs[1, 0], sharex=ax1)

    # Plot total ridership
    for i, col in enumerate(total_rep_cols):
        color = sns.color_palette("Purples", len(total_rep_cols))[i]
        ax1.plot(totals_with_dates[date_col], totals_with_dates[col], color=color, label=col)

    # Add changepoints for total ridership
    for idx in total_changepoints['indices']:
        if 0 <= idx < len(totals_with_dates):
            date_val = totals_with_dates[date_col].iloc[idx]
            ax1.axvline(x=date_val, color='red', linestyle='--', alpha=0.7)
            ax1.annotate(date_val.strftime('%Y-%m-%d'),
                         xy=(date_val, ax1.get_ylim()[1] * 0.9),
                         rotation=45, ha='left', color='red')

    ax1.set_title("Total Ridership Multivariate Changepoints", fontsize=14)
    ax1.legend(loc='best')

    # Plot percentage ridership
    for i, col in enumerate(pct_rep_cols):
        color = sns.color_palette("Purples", len(pct_rep_cols))[i]
        ax2.plot(pct_with_dates[date_col], pct_with_dates[col], color=color, label=col)

    # Add changepoints for percentage ridership
    for idx in pct_changepoints['indices']:
        if 0 <= idx < len(pct_with_dates):
            date_val = pct_with_dates[date_col].iloc[idx]
            ax2.axvline(x=date_val, color='red', linestyle='--', alpha=0.7)
            ax2.annotate(date_val.strftime('%Y-%m-%d'),
                         xy=(date_val, ax2.get_ylim()[1] * 0.9),
                         rotation=45, ha='left', color='red')

    ax2.set_title("Percentage Ridership Multivariate Changepoints", fontsize=14)
    ax2.legend(loc='best')

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # Main title
    fig_combined.suptitle("Comparison of Multivariate Changepoints: Total vs. Percentage Ridership", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to PDF
    pdf.savefig(fig_combined)
    plt.close(fig_combined)

    # Add individual feature plots with changepoints to PDF
    print("\nAdding individual feature plots to PDF...")

    # Total ridership individual features - plot all features
    for col in total_cols_to_analyze:
        fig_ind = plt.figure(figsize=(15, 6))
        plt.scatter(totals_with_dates[date_col], totals_with_dates[col], color='purple', s=20)
        plt.title(f"Total Ridership - {col}", fontsize=16)

        # Add changepoints
        for idx in total_changepoints['indices']:
            if 0 <= idx < len(totals_with_dates):
                date_val = totals_with_dates[date_col].iloc[idx]
                plt.axvline(x=date_val, color='red', linestyle='--', alpha=0.7)
                plt.annotate(date_val.strftime('%Y-%m-%d'),
                             xy=(date_val, totals_with_dates[col].iloc[idx]),
                             xytext=(10, 0), textcoords='offset points',
                             rotation=45, ha='left', color='red')

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()

        pdf.savefig(fig_ind)
        plt.close(fig_ind)

    # Percentage ridership individual features - plot all features
    for col in pct_cols_to_analyze:
        fig_ind = plt.figure(figsize=(15, 6))
        plt.scatter(pct_with_dates[date_col], pct_with_dates[col], color='purple', s=20)
        plt.title(f"Percentage Ridership - {col}", fontsize=16)

        # Add changepoints
        for idx in pct_changepoints['indices']:
            if 0 <= idx < len(pct_with_dates):
                date_val = pct_with_dates[date_col].iloc[idx]
                plt.axvline(x=date_val, color='red', linestyle='--', alpha=0.7)
                plt.annotate(date_val.strftime('%Y-%m-%d'),
                             xy=(date_val, pct_with_dates[col].iloc[idx]),
                             xytext=(10, 0), textcoords='offset points',
                             rotation=45, ha='left', color='red')

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()

        pdf.savefig(fig_ind)
        plt.close(fig_ind)

    # Close the PDF file
    pdf.close()
    print(f"\nPDF saved to: {pdf_filename}")

    print("\nMultivariate analysis complete!")
    return {
        'totals_df': totals_with_dates,
        'pct_df': pct_with_dates,
        'total_changepoints': total_changepoints,
        'pct_changepoints': pct_changepoints
    }, pdf_filename


mresults, mpdf_path = run_multivariate_analysis_with_pdf(X_with_dates, 'Date', n_bkps=4)


def filter_sundays(df, date_column='Date'):
    """
    Filters a dataframe to include only rows where the date falls on a Sunday.

    Parameters:
    df (pd.DataFrame): Input dataframe containing a date column
    date_column (str): Name of the column containing dates (default 'date')

    Returns:
    pd.DataFrame: A new dataframe containing only Sunday data
    """
    # Make sure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Filter for Sundays (dayofweek = 6 in pandas)
    sundays_df = df[df[date_column].dt.dayofweek == 2].copy()

    return sundays_df


# Assume df_raw is your full raw dataset
df_raw = X_with_dates.copy()

# Ensure the Date column is datetime
df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors='coerce')

# Extract column names
all_cols = df_raw.columns.tolist()

# Identify totals (contain 'Total' or 'Scheduled', and do NOT contain '%')
totals_cols_d = [col for col in all_cols if
                 ("Total" in col or "Scheduled" in col) and "%" not in col]

# Identify percentage columns (contain '%')
pct_cols_d = [col for col in all_cols if "%" in col]

# Optionally print them out
print("🟢 Total columns:")
print(total_cols)
print("\n🔵 Percentage columns:")
print(pct_cols)

# Now create separate DataFrames
totals_df_d = df_raw[["Date"] + totals_cols_d].copy()
pct_df_d = df_raw[["Date"] + pct_cols_d].copy()

# For totals_df_d
sundays_totals_d = filter_sundays(totals_df_d)
sundays_totals = filter_sundays(totals_df_d).drop('Date', axis='columns')

# For pct_df_d
sundays_pct_d = filter_sundays(pct_df_d)
sundays_pct = filter_sundays(pct_df_d).drop('Date', axis='columns')

sundays_X_d = filter_sundays(X_with_dates)



run_grid_search(sundays_totals, pdf_filename="sundays_totals_grid_search_plots.pdf")
run_grid_search(sundays_pct, pdf_filename="sundays_pct_grid_search_plots.pdf")


sundays_results_totals = kmeans_anomaly_detection_with_analysis(sundays_totals, threshold_percentile=90, pdf_path="sundays_totals_normal_vs_anomaly_plots.pdf")
sundays_results_pct = kmeans_anomaly_detection_with_analysis(sundays_pct, threshold_percentile=90, pdf_path="sundays_pct_normal_vs_anomaly_plots.pdf")

analyze_anomaly_clusters_with_normals(
    sundays_results_pct,
    data=sundays_pct,
    pdf_path="sundays_pct_anomaly_clusters_analysis.pdf"
)

analyze_anomaly_clusters_with_normals(
    sundays_results_totals,
    data=sundays_totals,
    pdf_path="sundays_totals_anomaly_clusters_analysis.pdf"
)

plot_feature_distributions(sundays_pct, title="Day-Specific Distributions (Percentage Features)", pdf_output = "day_pct_distribution.pdf" )
plot_feature_distributions(sundays_totals, title="Day-Specific Distributions (Total Features)", pdf_output = "day_totals_distribution.pdf")



results_pct_no_bus= kmeans_anomaly_detection_with_analysis(df_pct.drop(["Spectral_Cluster", "Buses: % of Comparable Pre-Pandemic Day"], axis='columns'), threshold_percentile=97, pdf_path="pct_normal_vs_anomaly_plots_no_bus.pdf")
analyze_anomaly_clusters_with_normals(
    results_pct_no_bus,
    data=df_pct.drop(["Spectral_Cluster", "Buses: % of Comparable Pre-Pandemic Day"], axis='columns'),
    pdf_path="pct_anomaly_clusters_analysis_no_bus.pdf"
)

results_totals_no_bus= kmeans_anomaly_detection_with_analysis(df_totals.drop(["Spectral_Cluster", "Buses: Total Estimated Ridership"], axis='columns'), threshold_percentile=97, pdf_path="totals_normal_vs_anomaly_plots_no_bus.pdf")
analyze_anomaly_clusters_with_normals(
    results_totals_no_bus,
    data=df_totals.drop(["Spectral_Cluster", "Buses: Total Estimated Ridership"], axis='columns'),
    pdf_path="totals_anomaly_clusters_analysis_no_bus.pdf"
)


import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_pca_with_date_gradient(data, numerical_columns=None, date_column=None,
                                random_state=42, figsize=(10, 7),
                                title="PCA Colored by Date", pdf_path=None):
    """
    Plot PCA (2D) with points colored by actual date values using a continuous gradient.
    """
    if date_column is None or date_column not in data.columns:
        raise ValueError("Please provide a valid `date_column` that exists in the data.")

    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)

    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Standardize numerical data
    X = df[numerical_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Prepare dataframe for plotting
    pca_df = pd.DataFrame(X_pca, columns=["PCA 1", "PCA 2"])
    pca_df["Date"] = df[date_column]
    pca_df["DateNum"] = mdates.date2num(df[date_column])

    # Normalize dates for color mapping
    norm = Normalize(vmin=pca_df["DateNum"].min(), vmax=pca_df["DateNum"].max())
    cmap = plt.cm.Purples  # <- Changed to Purples
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        pca_df["PCA 1"], pca_df["PCA 2"],
        c=pca_df["DateNum"],
        cmap=cmap,
        norm=norm,
        s=60,
        alpha=0.85,
        edgecolor='k',
        linewidth=0.3
    )

    # Colorbar with formatted date ticks
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Date")
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())

    plt.title(title)
    plt.tight_layout()

    # Save to PDF if path provided
    if pdf_path:
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        print(f"PCA plot with date gradient saved to: {pdf_path}")

    plt.show()


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def plot_tsne_weekend_kmeans(data, date_column, numerical_columns=None,
                             pca_components=2, perplexity=40, random_state=42,
                             figsize=(10, 7), title="t-SNE: Weekend + KMeans Clusters",
                             pdf_path=None):
    """
    Plot t-SNE with KMeans(2) clusters, using darker shades for weekends.
    Performs PCA dimensionality reduction before t-SNE and clustering.
    """
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df["Is_Weekend"] = df[date_column].dt.weekday >= 5

    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Standardize features
    X = df[numerical_columns].values
    X_scaled = StandardScaler().fit_transform(X)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: {explained_variance}")
    print(f"Total variance explained: {sum(explained_variance):.4f}")

    # t-SNE on PCA result
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X_pca)

    # KMeans clustering on t-SNE result
    kmeans = KMeans(n_clusters=2, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_tsne)

    # Combine results into DataFrame
    tsne_df = pd.DataFrame(X_tsne, columns=["TSNE 1", "TSNE 2"])
    tsne_df["Is_Weekend"] = df["Is_Weekend"]
    tsne_df["Cluster"] = cluster_labels

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for each combination of cluster and weekend status
    color_mapping = {
        (0, False): 'darkgray',  # Cluster 0, Weekday (lighter gray)
        (0, True): 'dimgray',  # Cluster 0, Weekend (darker gray)
        (1, False): '#C9B6DB',  # Cluster 1, Weekday (lighter purple)
        (1, True): '#6A329F',  # Cluster 1, Weekend (darker purple)
    }

    # Create markers based on cluster
    markers = ['o', 's']  # circle for cluster 0, square for cluster 1

    # Plot each group
    for (cluster, is_weekend), group_df in tsne_df.groupby(['Cluster', 'Is_Weekend']):
        weekend_label = "Weekend" if is_weekend else "Weekday"
        ax.scatter(
            group_df["TSNE 1"],
            group_df["TSNE 2"],
            marker=markers[cluster],
            c=color_mapping[(cluster, is_weekend)],
            s=70,
            alpha=0.85,
            edgecolor='black',
            linewidth=0.4,
            label=f"Cluster {cluster} - {weekend_label}"
        )

    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best")

    plt.tight_layout()

    if pdf_path:
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        print(f"t-SNE with KMeans plot saved to: {pdf_path}")

    plt.show()


plot_tsne_weekend_kmeans(
    data=pct_df_d,
    date_column="Date",
    title="PCT - t-SNE with KMeans(2) and Weekend Highlight"
)

plot_tsne_weekend_kmeans(
    data=totals_df_d,
    date_column="Date",
    title="Totals - t-SNE with KMeans(2) and Weekend Highlight"
)


def plot_pca_weekend_kmeans(data, date_column, numerical_columns=None,
                            random_state=42, figsize=(10, 7),
                            title="PCA: Weekend + KMeans Clusters",
                            pdf_path=None):
    """
    Plot PCA with KMeans(2) clusters, using darker shades for weekends.
    """
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df["Is_Weekend"] = df[date_column].dt.weekday >= 5

    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Standardize features
    X = df[numerical_columns].values
    X_scaled = StandardScaler().fit_transform(X)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: {explained_variance}")
    print(f"Total variance explained: {sum(explained_variance):.4f}")

    # KMeans clustering on PCA result
    kmeans = KMeans(n_clusters=2, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_pca)

    # Combine results into DataFrame
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Is_Weekend"] = df["Is_Weekend"]
    pca_df["Cluster"] = cluster_labels

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for each combination of cluster and weekend status
    color_mapping = {
        (0, False): 'darkgray',  # Cluster 0, Weekday (lighter gray)
        (0, True): 'dimgray',  # Cluster 0, Weekend (darker gray)
        (1, False): '#C9B6DB',  # Cluster 1, Weekday (lighter purple)
        (1, True): '#6A329F',  # Cluster 1, Weekend (darker purple)
    }

    # Create markers based on cluster
    markers = ['o', 's']  # circle for cluster 0, square for cluster 1

    # Plot each group
    for (cluster, is_weekend), group_df in pca_df.groupby(['Cluster', 'Is_Weekend']):
        weekend_label = "Weekend" if is_weekend else "Weekday"
        ax.scatter(
            group_df["PC1"],
            group_df["PC2"],
            marker=markers[cluster],
            c=color_mapping[(cluster, is_weekend)],
            s=70,
            alpha=0.85,
            edgecolor='black',
            linewidth=0.4,
            label=f"Cluster {cluster} - {weekend_label}"
        )

    # Add axis labels with variance explained
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")

    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best")

    plt.tight_layout()

    if pdf_path:
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        print(f"PCA with KMeans plot saved to: {pdf_path}")

    plt.show()


plot_pca_weekend_kmeans(
    data=totals_df_d,
    date_column="Date",
    title="Totals - PCA with KMeans(2) and Weekend Highlight",
    pdf_path="weekend_totals_pca_analysis.pdf"
)
print(totals_df_d.head)

from scipy import stats
from scipy.stats import wilcoxon


def analyze_weekend_weekday_ratio(df, feature_type='percentage', window_size=4,
                                  pdf_filename='weekend_weekday_ratio.pdf'):
    """
    Analyze and plot the weekend to weekday ratio for selected features,
    including statistical tests comparing ratios to 1, and save the plot to PDF.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing date column and features to analyze
    feature_type : str
        'percentage' for % of Pre-Pandemic Day, 'total' for raw totals
    window_size : int
        Number of weeks to include in moving average (not used in plotting)
    pdf_filename : str
        Output filename for the PDF report

    Returns:
    --------
    Tuple: (DataFrame with calculated ratios, DataFrame with statistical test results)
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])
    df['YearWeek'] = df['Date'].dt.strftime('%Y-%U')

    if feature_type.lower() == 'percentage':
        selected_columns = [col for col in df.columns if '%' in col]
    elif feature_type.lower() == 'total':
        selected_columns = [col for col in df.columns if 'Total' in col]
    else:
        raise ValueError("feature_type must be either 'percentage' or 'total'")

    for col in selected_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    ratio_results = []
    for week in df['YearWeek'].unique():
        week_data = df[df['YearWeek'] == week]
        weekend_data = week_data[week_data['IsWeekend']]
        weekday_data = week_data[~week_data['IsWeekend']]

        if len(weekend_data) == 0 or len(weekday_data) == 0:
            continue

        for col in selected_columns:
            weekend_value = weekend_data[col].mean()
            weekday_value = weekday_data[col].mean()

            if weekday_value > 0:
                ratio = weekend_value / weekday_value
                weekend_date = weekend_data['Date'].iloc[0]  # Take first weekend day as reference
                ratio_results.append({
                    'Date': weekend_date,
                    'YearWeek': week,
                    'Feature': col,
                    'WeekendValue': weekend_value,
                    'WeekdayAvgValue': weekday_value,
                    'Ratio': ratio
                })

    ratio_df = pd.DataFrame(ratio_results)

    if len(ratio_df) == 0:
        print("Not enough data to calculate ratios.")
        return None

    stats_results = []
    for feature in ratio_df['Feature'].unique():
        feature_ratios = ratio_df[ratio_df['Feature'] == feature]['Ratio'].values
        if len(feature_ratios) > 0:
            try:
                w_stat, p_value = wilcoxon(feature_ratios - 1)
            except ValueError:
                w_stat, p_value = np.nan, np.nan

            effect_size = (np.median(feature_ratios) - 1) / np.std(feature_ratios) if np.std(feature_ratios) > 0 else 0
            direction = "higher than" if np.median(feature_ratios) > 1 else "lower than"
            if pd.isna(p_value) or p_value >= 0.05:
                direction = "not significantly different from"
            median_diff = np.median(feature_ratios) - 1
            median_diff_percent = median_diff * 100
            stats_results.append({
                'Feature': feature,
                'Median Ratio': np.median(feature_ratios),
                'W-statistic': w_stat,
                'P-value': p_value,
                'P-value (sci)': f"{p_value:.1e}" if not pd.isna(p_value) else "NaN",
                'Significant (p < 0.05)': p_value < 0.05 if not pd.isna(p_value) else False,
                'Effect Size (approx)': effect_size,
                'Direction': direction,
                'Median Difference': median_diff,
                'Median Difference (%)': median_diff_percent
            })

    stats_df = pd.DataFrame(stats_results)

    plot_ratios_with_stats(ratio_df, stats_df, feature_type, pdf_filename)
    return ratio_df, stats_df


def plot_ratios_with_stats(ratio_df, stats_df, feature_type, pdf_filename):
    """
    Plot Weekend-Weekday ratios over time for each feature and save to PDF.
    """
    if len(ratio_df) == 0:
        print("No data to plot.")
        return

    purple_main = '#663399'
    purple_grid = '#D8BFD8'
    purple_ref = '#483D8B'
    purple_shade = '#8da0cb'

    features = ratio_df['Feature'].unique()
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(len(features), 1, figsize=(14, len(features) * 3.5), sharex=True)

    if len(features) == 1:
        axes = [axes]

    for i, feature in enumerate(features):
        feature_data = ratio_df[ratio_df['Feature'] == feature].sort_values('Date')
        ratios = feature_data['Ratio'].dropna().values
        axes[i].plot(feature_data['Date'], feature_data['Ratio'], 'o-', color=purple_main,
                     linewidth=2.5, markersize=7, label='Weekly Ratio')
        axes[i].axhline(y=1, color=purple_ref, linewidth=1.5, alpha=0.7)
        axes[i].grid(True, linestyle='--', color=purple_grid, alpha=0.6)
        axes[i].set_facecolor('#FCFAFF')
        axes[i].set_ylabel('Weekend / Weekday Ratio', fontsize=12, fontweight='bold')

        stats_row = stats_df[stats_df['Feature'] == feature]
        if not stats_row.empty:
            p_display = stats_row['P-value (sci)'].iloc[0]
            median_ratio = stats_row['Median Ratio'].iloc[0]
            direction = stats_row['Direction'].iloc[0]
            median_diff = stats_row['Median Difference (%)'].iloc[0]
            sig_stars = "***" if stats_row['Significant (p < 0.05)'].iloc[0] else ""
            title = (
                f"{feature}\nMedian Ratio: {median_ratio:.3f} ({median_diff:+.1f}%) {sig_stars}\n"
                f"Weekend is {direction} weekday (p={p_display})"
            )
        else:
            title = f"{feature}"

        axes[i].set_title(title, fontsize=13, fontweight='bold')
        axes[i].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    fig.patch.set_facecolor('#F8F4FF')
    fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    plt.close(fig)


# Example usage:
ratio_df, stats_df = analyze_weekend_weekday_ratio(
    X_with_dates,
    feature_type='percentage',  # or 'total'
    window_size=4,  # not currently used for plotting, but kept for compatibility
    pdf_filename='sunday_weekday_analysis.pdf'  # name of the PDF file to generate
)
# Print the statistical results table
print(stats_df)


df_2022_onwards = X_with_dates[X_with_dates['Date'] >= '2022-01-01']
ratio_df_2022, stats_df_2022 = analyze_weekend_weekday_ratio(
    df_2022_onwards,
    feature_type='percentage',           # or 'total'
    window_size=4,                       # not currently used for plotting, but kept for compatibility
    pdf_filename='sunday_weekday_analysis_2022.pdf'  # name of the PDF file to generate
)

# Print the statistical results table
print(stats_df_2022)

df_2022_previous = X_with_dates[X_with_dates['Date'] <= '2022-01-01']

ratio_dfp_2022, stats_dfp_2022 = analyze_weekend_weekday_ratio(
    df_2022_previous,
    feature_type='percentage',           # or 'total'
    window_size=4,                       # not currently used for plotting, but kept for compatibility
    pdf_filename='sunday_weekday_analysis_2022_previous.pdf'  # name of the PDF file to generate
)

# Print the statistical results table
print(stats_dfp_2022)

from scipy.stats import mannwhitneyu


def weekend_weekday_comparison(df, feature_type='total'):
    """
    Compare weekend vs. weekday values for all features with non-parametric tests

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing date column and features to analyze
    feature_type : str
        'percentage' or 'total' to select which columns to analyze

    Returns:
    --------
    DataFrame with comparison results
    """
    # Make copy and ensure datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Create weekend/weekday flag
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])

    # Select appropriate columns
    if feature_type.lower() == 'percentage':
        feature_cols = [col for col in df.columns if '%' in col]
    elif feature_type.lower() == 'total':
        feature_cols = [col for col in df.columns if 'Total' in col]
    else:
        raise ValueError("feature_type must be 'percentage' or 'total'")

    # Initialize results storage
    results = []

    for col in feature_cols:
        # Get values for each group
        weekend_vals = df[df['IsWeekend']][col].dropna()
        weekday_vals = df[~df['IsWeekend']][col].dropna()

        # Skip if not enough data
        if len(weekend_vals) < 3 or len(weekday_vals) < 3:
            continue

        # Calculate statistics
        weekend_mean = np.mean(weekend_vals)
        weekday_mean = np.mean(weekday_vals)
        mean_diff_pct = ((weekend_mean - weekday_mean) / weekday_mean) * 100 if weekday_mean != 0 else np.nan

        # Non-parametric test
        try:
            stat, p = mannwhitneyu(weekend_vals, weekday_vals, alternative='two-sided')
        except:
            stat, p = np.nan, np.nan

        # Determine significance and format p-value
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        # Format p-value in scientific notation
        p_formatted = f"{p:.2e}" if not np.isnan(p) else "NaN"

        results.append({
            'Feature': col,
            'Weekend_Mean': weekend_mean,
            'Weekday_Mean': weekday_mean,
            'Percent_Difference': mean_diff_pct,
            'MannWhitneyU_Stat': stat,
            'P_Value': p_formatted,
            'Significance': sig
        })

    # Create and format results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('P_Value')

    # Format numeric columns (excluding the formatted p-value)
    float_cols = ['Weekend_Mean', 'Weekday_Mean', 'Percent_Difference']
    results_df[float_cols] = results_df[float_cols].round(3)

    return results_df


# Example usage:
# For percentage features
percentage_results = weekend_weekday_comparison(
    df=X_with_dates,
    feature_type='percentage'
)

# For total features
total_results = weekend_weekday_comparison(
    df=X_with_dates,
    feature_type='total'
)

# Display results
print("Weekend vs Weekday Comparison (Percentage Features):")
display(percentage_results)

print("\nWeekend vs Weekday Comparison (Total Features):")
display(total_results)

def compare_percent_to_one(df, date_col='Date', alpha=0.05):
    """
    For each percentage feature in the dataframe, test whether its values
    are significantly different from 1 (i.e., 100% of pre-pandemic baseline).

    Parameters:
    -----------
    df : pandas DataFrame
        A DataFrame containing a date column and multiple percentage columns.
    date_col : str
        Name of the column containing dates.
    alpha : float
        Significance threshold for hypothesis testing.

    Returns:
    --------
    results_df : pandas DataFrame
        A DataFrame summarizing the test results for each feature.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Identify percentage columns (contains '%' in name)
    pct_cols = [col for col in df.columns if '%' in col and col != date_col]

    results = []

    for col in pct_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()

        if len(values) < 3:
            continue  # not enough data for test

        try:
            stat, p_val = wilcoxon(values - 1)
            median_val = np.median(values)
            median_diff = median_val - 1
            direction = (
                "significantly higher than 1"
                if median_diff > 0 and p_val < alpha else
                "significantly lower than 1"
                if median_diff < 0 and p_val < alpha else
                "not significantly different from 1"
            )

            results.append({
                'Feature': col,
                'Median': median_val,
                'Median Diff': median_diff,
                'P-value': p_val,
                'P-value (sci)': f"{p_val:.1e}",
                'Direction': direction,
                'Significant': p_val < alpha
            })

        except ValueError:
            results.append({
                'Feature': col,
                'Median': np.nan,
                'Median Diff': np.nan,
                'P-value': np.nan,
                'P-value (sci)': "NaN",
                'Direction': "Error",
                'Significant': False
            })

    return pd.DataFrame(results).sort_values('P-value')
df_2023 = X_with_dates[X_with_dates['Date'] >= '2023-03-01']

percent_stats = compare_percent_to_one(df_2023)
display(percent_stats)

def plot_rolling_average_to_pdf(df, date_col='Date', feature='Buses: % of Comparable Pre-Pandemic Day', window=7, output_dir='.'):
    """
    Plot the rolling average of the selected percentage feature and save the plot to a PDF.
    """

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(by=date_col).reset_index(drop=True)
    df['Smoothed'] = df[feature].rolling(window=window, center=True).mean()
    df.dropna(subset=['Smoothed'], inplace=True)

    # Prepare output
    timestamp = pd.to_datetime('now').strftime("%Y%m%d_%H%M%S")
    pdf_path = f"{output_dir}/rolling_average_{timestamp}.pdf"
    pdf = PdfPages(pdf_path)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df[date_col], df['Smoothed'], color='purple', linewidth=2.5, label='7-day Smoothed')

    ax.set_title(f"7-day Rolling Average of {feature}", fontsize=14)
    ax.set_ylabel('% of Pre-Pandemic Day', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to the PDF
    pdf.savefig(fig)
    plt.close(fig)

    # Close the PDF
    pdf.close()
    print(f"PDF saved to: {pdf_path}")

# Call the function to plot the rolling average and save to PDF
plot_rolling_average_to_pdf(df=X_with_dates, date_col='Date', feature='Buses: % of Comparable Pre-Pandemic Day', window=7, output_dir='.')