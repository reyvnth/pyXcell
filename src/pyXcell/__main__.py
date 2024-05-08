"""Main module for executing the pyXcell pipeline."""

import argparse
import os
import time

import pandas as pd

from .BiologicalContext import gene_importance
from .DimReduc import nonnegativeMatrixFactorization
from .DimReduc import principalComponentAnalysis
from .ModelTraining import train_model
from .ShapAnalysis import boxplotExploration
from .ShapAnalysis import runSHAP


def main(args):
    """Execute the pipeline.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        None
    """
    start = time.time()

    expression_file = args.expression_file
    meta_file = args.meta_file
    data_label = args.data_label
    reduc_method = args.reduction_method
    proportion_var_explained = args.proportion_var_explained
    num_ranks = args.num_ranks
    min_k = args.min_k
    max_k = args.max_k
    covariates = args.covariates
    target = args.target
    method = args.method
    # harmonize = args.harmonize
    # batch_keys = args.batch_keys

    # IMPORT DATA ========
    disease_fib_exp = pd.read_csv(expression_file, index_col=0)
    meta_data = pd.read_csv(meta_file, index_col=0)

    out_path = f"../output/results/{data_label}_{reduc_method}_{num_ranks}_{method}/"
    os.makedirs(out_path, exist_ok=True)

    cell_names = disease_fib_exp.columns
    gene_names = disease_fib_exp.index

    if reduc_method == "nmf":
        W, H = nonnegativeMatrixFactorization(
            disease_fib_exp,
            numberOfComponents=num_ranks,
            min_k=min_k,
            max_k=max_k,
            outpath=out_path,
        )
    elif reduc_method == "pca":
        H = principalComponentAnalysis(
            disease_fib_exp.T, proportion_var_explained, outpath=out_path
        )

    W = pd.DataFrame(W, index=gene_names)
    H = pd.DataFrame(H, columns=cell_names).T

    y_variable = target
    # batch_keys = ["sample"]
    y_labels = meta_data[y_variable]

    rf_model, X_with_covariates = train_model(
        H,
        meta_data,
        covariates,
        y_variable,
        method=method,
        harmonize=False,
        outpath=out_path,
    )

    shap_values = runSHAP(rf_model, X_with_covariates, outpath=out_path)

    shap_values_df = pd.DataFrame(
        shap_values[1],
        columns=[f"{col}_shap" for col in X_with_covariates.columns],
        index=cell_names,
    )
    meta_data[target] = y_labels
    boxplotExploration(shap_values_df, meta_data, target, outpath=out_path)

    shap_values_df_dims_only = shap_values_df.drop(
        list(shap_values_df.filter(regex="^(?!.*dim)")), axis=1, inplace=False
    )
    gene_importance(
        W,
        disease_fib_exp.T,
        shap_values_df_dims_only,
        meta_data,
        y_variable,
        y_labels,
        outpath=out_path,
    )

    end = time.time()
    elapsed = end - start
    print(f"Elapsed time: {elapsed}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XCell input parameters.")
    parser.add_argument(
        "--expression-file",
        dest="expression_file",
        type=str,
        help="Path to expression data file.",
    )
    parser.add_argument(
        "--meta-file", dest="meta_file", type=str, help="Path to meta data file."
    )
    parser.add_argument(
        "--data",
        dest="data_label",
        type=str,
        help="Label for data (to be used when labeling output directory)",
    )
    parser.add_argument(
        "--reduction-method",
        dest="reduction_method",
        type=str,
        default="nmf",
        help="Dimensionality reduction method: 'nmf' or 'pca'.",
    )
    parser.add_argument(
        "--proportion-var-explained",
        dest="proportion_var_explained",
        type=float,
        default=0.95,
        help="Desired proportion of variance explained if PCA is selected.",
    )
    parser.add_argument(
        "--number-ranks",
        dest="num_ranks",
        type=int,
        default=-1,
        help="Number of ranks for NMF (default: -1).",
    )
    parser.add_argument(
        "--minimum-k",
        dest="min_k",
        type=int,
        default=2,
        help="Minimum k value for selecting optimal number of ranks if NMF is selected (default: 2).",
    )
    parser.add_argument(
        "--maximum-k",
        dest="max_k",
        type=int,
        default=7,
        help="Maximum k value for selecting optimal number of ranks if NMF is selected (default: 7).",
    )
    parser.add_argument(
        "--covariates",
        dest="covariates",
        nargs="+",
        type=str,
        help="Covariates to include in classification model.",
    )
    parser.add_argument(
        "--target-variable",
        dest="target",
        type=str,
        help="Name of the target/outcome column in the meta data table.",
    )
    parser.add_argument(
        "--classification-method",
        dest="method",
        type=str,
        default="rf",
        help="Classification model to train - rf for Random Forest (default), xgb for XGBoost.",
    )
    parser.add_argument(
        "--harmonize", dest="harmonize", action="store_true", help="Run harmony?"
    )
    parser.add_argument(
        "--batch-keys",
        dest="batch_keys",
        nargs="+",
        type=str,
        help="If running harmony, please provide a list of batch keys to use. These should be names of columns in the metadata table.",
    )

    args = parser.parse_args()
    main(args)
