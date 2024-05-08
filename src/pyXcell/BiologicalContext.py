"""Functions for computing SHAP-informed gene importance."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


def gene_importance(
    gene_loadings, exp_mat, shap_mat, meta_data, y, y_variables, outpath="./results/"
):
    """Compute SHAP-informed gene importance for the top features.

    Descriptions:
        This function calculates the SHAP-informed gene importance based on the provided data.

    Args:
        gene_loadings : DataFrame
            The rank by cell NMF loadings.
        exp_mat : DataFrame
            The expression matrix.
        shap_mat : DataFrame
            The DataFrame of SHAP values.
        meta_data : DataFrame
            The meta data table.
        y : str
            Name of the target label column.
        y_variables : Series
            The target labels.
        outpath : (str, optional)
            Path to save output files. Defaults to "./results/".

    Returns:
        DataFrame: The gene importance matrix (cell by key genes).

    # noqa: B950
    """
    print("Gene Importance")

    important_dimensions = "dim7_shap"

    meta_data.set_index(y_variables.index, inplace=True)
    meta_data[y] = y_variables

    key_feature = []
    for i in range(gene_loadings.shape[1]):
        temp = gene_loadings.iloc[:, i].index[
            gene_loadings.iloc[:, i].values
            > np.quantile(gene_loadings.iloc[:, i].values, 0.95)
        ]
        key_feature = np.union1d(key_feature, temp)
    # key_features_df = gene_loadings.loc[gene_loadings.index.isin(key_feature)]

    gene_importance_mat = np.outer(exp_mat, shap_mat[important_dimensions])
    gene_importance_mat = pd.DataFrame(gene_importance_mat, columns=key_feature)
    gene_importance_mat.set_index(meta_data.index, inplace=True)

    correlations = np.corrcoef(shap_mat[important_dimensions], exp_mat)
    correlations_with_matrix = correlations[0, 1:]

    print("Correlation with each column in the matrix:", correlations_with_matrix)

    plt.bar(range(len(correlations_with_matrix)), correlations_with_matrix)
    plt.xlabel("Column Index")
    plt.ylabel("Correlation")
    plt.title("Correlation with Each Column in the Matrix")
    plt.savefig(f"{outpath}corr.png")

    plt.figure(figsize=(20, 20))
    sns.set(font_scale=0.5)
    lut_disease = dict(
        zip(
            set(meta_data[y]),
            sns.cubehelix_palette(n_colors=len(set(meta_data[y])), light=0.9, dark=0.1),
            strict=True,  # Add explicit strict parameter
        )
    )
    row_colors_disease = pd.Series(meta_data[y]).map(lut_disease)
    p = sns.clustermap(
        gene_importance_mat,
        row_colors=[row_colors_disease],
        linewidths=0,
        label="Disease",
    )
    p.fig.suptitle("Heatmap of SHAP-derived gene importance for each cell", fontsize=22)
    p.ax_heatmap.set_yticklabels([])
    hand = [Patch(facecolor=lut_disease[name]) for name in lut_disease]
    plt.legend(
        hand,
        lut_disease,
        title="Disease",
        bbox_to_anchor=(1, 1),
        loc="upper right",
        fontsize=20,
    )
    ax1 = p.ax_heatmap
    ax1.set_xlabel("Key Genes", fontsize=20)
    ax1.set_ylabel("Cells", fontsize=20)
    ax1.tick_params(right=False)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.savefig(f"{outpath}GeneImportanceHeatmap_dim7.png")

    return gene_importance_mat
