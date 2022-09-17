import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from processed_dataset_tools import PHENOTYPES
from graphical_tools_helpers import _display_plot_values_labels,\
    _show_values_on_bars


def generate_heatmap(correct, predicted, save="heatmap.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    sn.set(font_scale=1.4)
    sn.heatmap(confusion_matrix(correct, predicted),
               annot=True,
               annot_kws={"size": 14},
               cmap='rocket_r',
               fmt='g',
               ax=ax)

    labels = PHENOTYPES
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(labels, size=10)
    ax.set_yticklabels(labels, size=10)
    ax.set_title("Validation Confusion: Depth-2 Tree, Basic Feats")

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches='tight')

    return fig


def plot_utility_2d(points, df, unlabeled_point_color="model_preds",
                    labeled_point_color="ground_truth",
                    dim_red_alg="PCA", save="plot2d.png"):
    assert (unlabeled_point_color in {"model_preds", "gray"}),\
        "Pick one of 'model_preds' for model predictions or 'gray'"

    assert (labeled_point_color in {"ground_truth", "model_preds"}),\
        "Pick one of 'ground_truth' for true labels or 'model_preds' " +\
        "for model predictions"

    assert ("ground_truth" in df.columns) and ("label" in df.columns),\
        "Need both 'ground_truth' and 'model_preds' in df columns"

    fig, ax = plt.subplots(figsize=(7, 7))

    # ax.scatter(pca_output[:, 0], pca_output[:, 1], s=5, c='b')
    ax.set_ylabel(f"{dim_red_alg} Component 2")
    ax.set_xlabel(f"{dim_red_alg} Component 1")

    if labeled_point_color == "ground_truth":
        title = "Ground Truth"
    else:
        title = "Model Predicted"

    ax.set_title(f"{dim_red_alg} of Model Features with {title} Labels")

    label_to_color = {1: 'r', 2: 'lawngreen',
                      3: 'b', 4: 'orange', None: "gray"}
    label_to_meaning = {1: 'Faint', 2: 'Punctate',
                        3: 'Cyto', 4: 'Peripheral', None: "Unlabeled"}
    handles = {}

    ground_truth = df["ground_truth"].to_numpy()
    model_preds = df["model_preds"].to_numpy()

    unlabeled = np.argwhere(np.isnan(ground_truth)).ravel()
    labeled = np.argwhere(~np.isnan(ground_truth)).ravel()

    if unlabeled_point_color == "gray":
        unlabeled_color = "gray"
    else:
        unlabeled_color = [label_to_color[model_preds[ind]]
                           for ind in unlabeled]

    scatter_pts = ax.scatter(points[unlabeled, 0], points[unlabeled, 1], s=5,
                             c=unlabeled_color, label="Unlabeled", alpha=0.05)
    handles[-1] = scatter_pts

    for idx in labeled:
        if labeled_point_color == "ground_truth":
            label = int(ground_truth[idx])
        else:
            label = int(model_preds[idx])

        scatter_point = ax.scatter(points[idx, 0], points[idx, 1],
                                   c=label_to_color[label], s=10,
                                   label=f"Cluster {label} [{label_to_meaning[label]}]")
        handles[label] = scatter_point

    ax.legend(handles=sorted(list(handles.values()),
                             key=lambda x: x.get_label()), fontsize=10)

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches='tight')

    return fig


def plot_phenotype_countplot(series, values_on_bars=None, is_pct=False,
                             save="countplot.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sn.countplot(x=series, ax=ax)

    ax = _display_plot_values_labels(ax, "Distribution of Model Predictions",
                                     "Prediction", "Count", values_on_bars, is_pct,
                                     PHENOTYPES)

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches='tight')

    return fig


def plot_phenotype_barplot(df, x_col="index", y_col="model_pred",
                           values_on_bars=None, is_pct=False,
                           save="pheno_barplot.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sn.barplot(data=df, x=x_col, y=y_col, ax=ax)

    ax = _display_plot_values_labels(ax, "Distribution of Model Predictions",
                                     "Prediction", "Count", values_on_bars, is_pct,
                                     PHENOTYPES)

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches='tight')

    return fig


def plot_gene_distribution_barplot(df, x_col="gene_symbol", y_col="pct",
                                   values_on_bars=None, is_pct=False,
                                   phenotype="Faint",
                                   save="gene_barplot.png"):
    assert phenotype in PHENOTYPES, f"Must choose one of {PHENOTYPES} as phenotype"

    fig, ax = plt.subplots(figsize=(10, 7))
    sn.barplot(data=df, x="gene_symbol", y="pct", ax=ax)

    ax = _display_plot_values_labels(ax, f"Distribution of {phenotype} Predictions",
                                     "Gene", "Percentage of Gene in Class",
                                     values_on_bars, is_pct)
    plt.xticks(rotation=60)

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches='tight')

    return fig


def plot_gene_vs_nontargeting_accuracies(accuracies, gene_counts,
                                         plot_type="boxplot",
                                         save="accuracies.png"):
    assert "mean" in accuracies.columns,\
        "'mean' must be a column in DataFrame"

    assert plot_type in {"boxplot", "scatter"},\
        "Plot type must be one of 'boxplot' or 'scatter'"

    fig, ax = plt.subplots(figsize=(15, 7))
    subset = pd.melt(accuracies.sort_values("mean", ascending=False)
                               .head(30).drop(columns="mean").T).copy()
    subset = subset.merge(gene_counts, left_on="variable",
                          right_on="gene_symbol")
    subset["var_count"] = subset["variable"] + \
        " [" + subset["ct"].astype(str) + "]"
    if plot_type == "boxplot":
        sn.boxplot(x="var_count", y="value", data=subset, ax=ax)
    else:
        sn.scatterplot(x="var_count", y="value", data=subset, ax=ax)

    ax = _display_plot_values_labels(ax, "Boxplot of Top Accuracies",
                                     "Gene [Sorted by Mean Accuracy], with counts",
                                     "Accuracy vs Nontargeting")
    plt.xticks(rotation=60)

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches='tight')

    return fig


def plot_model_comparisons(df, save="model_comp.png"):
    assert "Test Accuracy" in df.columns,\
        "Must have 'Test Accuracy' in columns"

    fig, ax = plt.subplots(figsize=(15, 7))
    sn.scatterplot(data=df.T.sort_values("Test Accuracy"), ax=ax, s=500)
    plt.xticks(rotation=45, size=14)
    plt.yticks(size=14)
    plt.legend(fontsize=14, loc="upper left")
    plt.xlabel("Model Type", size=18)
    plt.ylabel("Accuracy", size=18)
    plt.title(
        "Scatter Plot of Various Model Embedding Balanced Accuracies", size=20)

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches='tight')

    return fig
