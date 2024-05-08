"""Functions for implementing Dimensionality reduction."""

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from multianndata import MultiAnnData as mad
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def getNeighborhoodAbundanceMat(X, meta, covariates, y_variable):
    """Use CNA to get the neighborhood abundance matrix.

    Args:
        X : DataFrame
            The latent dimensions.
        meta (DataFrame): Matrix of covariates + target to be used in the model.
        covariates (list): List of column names for covariates to include in the model.
        y_variable (str): The name of the column to be used as the target.

    Returns:
        matrix: Neighborhood abundance matrix.
    """
    # create MultiAnnData object
    d = mad(X=X.T, obs=meta, sampleid=covariates)
    print("**created the MultiAnnData object")
    # compute the UMAP cell-cell similarity graph
    sc.pp.neighbors(d, use_rep="X")
    print("**computed the cell-cell similarity graph**")
    # compute UMAP coordinates for plotting
    sc.tl.umap(d)
    # res = cna.tl.association(d, d.obs.disease)
    nam = d.uns["NAM.T"]
    return nam


def reduceDim(reducMethod, reducMethodParams):
    """Call the reduction method specified by user.

    Args:
        reducMethod (str): The name of the method to be used ("nmf" or "pca").
        reducMethodParams (dict): Parameters for the method selected.

    Returns:
        matrix/matrices: One matrix if PCA selected, tuple of matrices if NMF selected.
    """
    if reducMethod == "nmf":
        return nonnegativeMatrixFactorization(**reducMethodParams)
    elif reducMethod == "pca":
        return principalComponentAnalysis(**reducMethodParams)
    else:
        print(
            "Invalid dimensionality reduction method provided! Please input 'nmf' or 'pca'."
        )
        # sys.exit()
    return


def nonnegativeMatrixFactorization(
    X, numberOfComponents=-1, min_k=2, max_k=12, outpath="./results/"
):
    """Perform NMF.

    Args:
        X (DataFrame): The marker by cell matrix to be decomposed.
        numberOfComponents (int): Number of components or ranks to learn (if -1, then we will select k).
        min_k (int): Alternatively, provide the minimum number of ranks to test.
        max_k (int): And the maximum number of ranks to test.

    Returns:
        tuple: W and H matrices.
    """
    print("inside the NMF function")
    # check if the user has provided the number of components they would like
    if numberOfComponents == -1:
        # call function to select optimal k
        numberOfComponents = select_optimal_k(X, min_k, max_k, outpath=outpath)
    print("building NMF model")
    # perform NMF
    nmfModel = NMF(n_components=numberOfComponents, init="random", random_state=11)
    W = nmfModel.fit_transform(X)
    H = nmfModel.components_
    print("DONE!")

    return (W, H)


# TODO: selecting best k may be subjective if the silhouette scores are not that different.... this current implenetation is just selecting k based on the reconstruction error
def select_optimal_k(X, min_k, max_k, outpath="./results/"):
    """Select optimal k (number of components) and generate elbow plot for silhouette score.

    Args:
        X (DataFrame): The marker by cell matrix to be decomposed.
        min_k (int): Alternatively, provide the minimum number of ranks to test.
        max_k (int): And the maximum number of ranks to test.

    Returns:
        int: Optimal k for decomposition.
    """
    print("determining the optimal k")
    k_values = range(min_k, max_k + 1)
    reconstruction_errors = []
    silhouette_scores = []
    for k in k_values:
        nmfModel = NMF(n_components=k, init="random", random_state=11)
        transformed = nmfModel.fit_transform(X)
        reconstruction_errors.append(nmfModel.reconstruction_err_)
        kmeans = KMeans(n_clusters=k, n_init="auto", max_iter=500)
        cluster_labels = kmeans.fit_predict(transformed)
        # Calculate silhouette score
        silhouette = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette)

        # print(f"\n{k} - reconstruction error: {nmfModel.reconstruction_err_}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.set_title(f"Reconstruction Error for NMF ranks {min_k} to {max_k}")
    ax1.set_ylabel("Reconstruction Error")
    ax1.plot(k_values, reconstruction_errors, marker="o", linestyle="-", color="b")
    ax1.grid(True)
    ax2.set_title(f"Silhouette Score for NMF ranks {min_k} to {max_k}")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.plot(k_values, silhouette_scores, marker="o", linestyle="-", color="r")
    ax2.grid(True)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Save the combined figure to a PNG file
    plt.savefig(f"{outpath}selectingOptimalNMFk.png")

    final_k = reconstruction_errors.index(min(reconstruction_errors)) + min_k
    """print(f"{final_k}")
    print("plotting...")
    plt.plot(range(min_k, max_k), reconstruction_errors, linestyle="-")
    plt.plot(final_k, min(reconstruction_errors),color='red', marker="o", label=f'Selected k')
    plt.title(f"NMF Reconstruction Error for k = {min_k}:{max_k-1}")
    plt.xlabel("Number of Components (k)")
    plt.ylabel("Reconstruction Error")
    plt.grid(True)
    plt.savefig("./results/NMFReconstructionErrorPlot.png")"""
    """plt.plot(range(min_k, max_k), silhouette_scores, linestyle="-", marker='o')
    plt.title('Elbow Plot for NMF Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig("./results/NMFReconstructionErrorPlot.png")"""

    return final_k


def principalComponentAnalysis(X, var, outpath="./results/"):
    """Perform PCA.

    Args:
        X (DataFrame): The marker by cell matrix to be decomposed.
        var (float): Desired proportion of variance explained.

    Returns:
        dataframe: Principal components.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    pca = PCA(n_components=100, random_state=11)
    pca.fit(scaled_data)
    eigenvalues = pca.explained_variance_ratio_
    components = pca.components_
    print(f"shape of PCA components: {components.shape}")
    # loadings = pca.components_ * np.sqrt(eigenvalues)

    # find the number of components that explain the most variance
    numberOfComponents = select_number_of_components(eigenvalues, var, outpath=outpath)
    print(f"optimal num components: {numberOfComponents}")

    return components[:, :numberOfComponents]
    # return (loadings[:, :numberOfComponents], components[:, :numberOfComponents])


def select_number_of_components(eigenvalues, var, outpath="./results/"):
    """Find the number of the components based on the percentage of accumulated variance.

    Args:
        eigenvalues (array): Array of eigenvalues (explained variances) for the components.
        var (float): Desired proportion of variance explained.

    Returns:
        int: Number of components.
    """
    explained_variances = eigenvalues
    cumsum = np.cumsum(explained_variances)
    # print(cumsum)
    num_selected_components = np.argmax(cumsum >= var) + 1
    plt.plot(range(1, len(eigenvalues) + 1), cumsum, linestyle="-")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("Cumulative Variance Explained Across Components")
    plt.grid(True)
    plt.savefig(f"{outpath}PCAExplainedVariancePlot.png")

    return num_selected_components
