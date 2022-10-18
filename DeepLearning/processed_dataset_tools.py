import pandas as pd


PHENOTYPES = ("Faint", "Punctate", "Cyto", "Peripheral")


def _get_df_gene_symbol_subset(df, gene_symbol, view=False):
    """Gets a subset of DataFrame consisting of gene symbol

    Args:
        df (pandas Dataframe): DataFrame to filter
        gene_symbol (str): Gene symbol to filter on (i.e. NPC1)
        view (bool, optional): Whether to return a view or copy.
            Defaults to False.

    Returns:
        pandas Dataframe: Filtered DataFrame
    """
    filtered_df = df[df["gene_symbol"] == gene_symbol]

    if view:
        return filtered_df
    else:
        return filtered_df.copy()


def _get_df_value_count_percentages(df):
    """Returns value counts of DataFrame index as percentage of whole

    Args:
        df (pandas Dataframe): DataFrame to count

    Returns:
        pandas Dataframe: Percentage of each index in the DataFrame
    """
    return (df.value_counts().sort_index().values * 100 / len(df)).round(2)


def _isolate_one_phenotype_model_preds(df, model_preds_col, phenotype,
                                       preds_base_index=1, view=False):
    """Isolates a single phenotype out of model predictions

    Args:
        df (pandas DataFrame): DataFrame to filter
        model_preds_col (str): Name of model predictions column in df
        phenotype (str): Name of phenotype to isolate
            Note: It is assumed the numerical progression is
            [Faint, Punctate, Cyto, Peripheral] in increments of 1
        preds_base_index (int, optional): Prediction index of Faint.
            Defaults to 1.
        view (bool, optional): Whether to return a view or copy.
            Defaults to False.

    Returns:
        pandas Dataframe: DataFrame with phenotype isolated
    """

    assert phenotype in PHENOTYPES, f"Must choose one of {PHENOTYPES} as phenotype"
    filtered_df = df[df[model_preds_col] ==
                     preds_base_index + PHENOTYPES.index(phenotype)]

    if view:
        return filtered_df
    else:
        return filtered_df.copy()


def _get_gene_symbol_counts_and_pcts(df, model_preds_col="svm", phenotype="Faint"):
    """Gets gene symbol counts and percentages based on model predictions

    Args:
        df (pandas DataFrame): DataFrame to filter
        model_preds_col (str, optional):  Name of model predictions column in df.
            Defaults to "svm".
        phenotype (str, optional): Name of phenotype to isolate.
            Defaults to "Faint".

    Returns:
        pandas DataFrame: DataFrame with gene symbol counts and percentages
    """
    phenotype_subset = _isolate_one_phenotype_model_preds(
        df, model_preds_col, phenotype)

    # Extract percentage of each gene symbol within a phenotype
    pcts = pd.DataFrame((phenotype_subset["gene_symbol"].value_counts().sort_index() * 100 /
                         (df.gene_symbol.value_counts().sort_index()))
                         .sort_values(ascending=False)).reset_index()
    pcts.columns = ["gene_symbol", "pct"]

    # Extract counts of each gene symbol from whole dataset
    cts = pd.DataFrame(
        df.gene_symbol.value_counts().sort_index()).reset_index()
    cts.columns = ["gene_symbol", "ct"]

    # Merge DataFrames and replace NaNs with 0% percent in phenotype
    pcts = pcts.merge(cts, on="gene_symbol")
    pcts.loc[pcts[pcts.pct.isna()].index, "pct"] = 0

    return pcts
