import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression


def gene_vs_nontargeting_analysis(gene_df, embeddings, seed, method="pick", splits=10):
    """Compare specific genes against nontargeting controls

    Comparison is made via the accuracy of a balanced-class-weighted
    Logistic Regression across a specified number of splits

    Args:
        gene_df (pandas DataFrame): DataFrame containing columns
            "gene_symbol" with gene description strings
            "nontargeting" as a gene string in "gene_symbol"
            "embedding_index" with integer indices into embeddings
        embeddings (numpy array): Array of embeddings for each sample
            in gene_df
        seed (int): Seed used for random number functionality
        method (str, optional): One of "pick" or "all" describing the
            method of selecting gene samples. Defaults to "pick".
        splits (int, optional): Integer > 1 describing number of splits
            to test on. Defaults to 10.

    Returns:
        pandas DataFrame: DataFrame of per-gene accuracy result across splits
    """
    assert method in ("pick", "all"),\
        f"Expected method to be one of ('pick', 'all'), got {method}"

    assert isinstance(splits, int) and splits > 1,\
        f"Splits must be an integer > 1, got {splits}"

    # Minimum number of samples required for a gene to be
    # included in the analysis
    MIN_SAMPLES_THRESHOLD = 40

    per_gene_stats = []

    # Group the genes by gene symbol and extract all nontargeting genes
    grouped_by_gene = gene_df.groupby("gene_symbol")
    gene_groups = list(grouped_by_gene.groups.keys())
    nontargeting = gene_df[gene_df["gene_symbol"].str.startswith("nontargeting")]

    # Iterate over all unique genes in the input DataFrame
    for current_gene in tqdm(gene_groups):
        # Extract the portion of the DataFrame containing only genes of that sort
        current_gene_group = grouped_by_gene.get_group(current_gene)

        # If the gene is nontargeting, skip as nontargeting is the "control"
        # to be compared against
        if current_gene == "nontargeting":
            continue
        # If the gene group has less than some threshold number of elements, skip comparison
        elif len(current_gene_group) < MIN_SAMPLES_THRESHOLD:
            print(f"Skipping {current_gene} as it has less than {MIN_SAMPLES_THRESHOLD}")
            continue

        # The "pick" method implies picking a random set of nontargeting genes in the same quantity
        # as the number of samples of the selected gene
        if method == "pick":
            full_dataset = pd.concat([nontargeting.sample(n=len(current_gene_group), random_state=seed),
                                      current_gene_group]).reset_index(drop=True)
        # The "all" method implies using all of the nontargeting samples, regardless of the number
        # of samples of the selected gene
        elif method == "all":
            full_dataset = pd.concat([nontargeting, current_gene_group]).reset_index(drop=True)

        # Extract gene embeddings for all samples, and generate a target vector which is
        # 0 if the gene is nontargeting and 1 if the gene is the selected one
        # This is a binary classification problem
        gene_embeddings = embeddings[full_dataset["embedding_index"].values]
        gene_target = (full_dataset["gene_symbol"].values == current_gene).astype(int)

        # Split the dataset into random folds, stratified if needed
        # by the number of nontargeting samples
        strat_k_fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

        single_gene_acc = []

        for train_index, test_index in tqdm(strat_k_fold.split(gene_embeddings, gene_target), total=splits):
            # Initialize Logistic Regression classifier with balanced class weight
            classifier = LogisticRegression(class_weight='balanced',
                                            max_iter=10000,
                                            random_state=seed)

            # Extract embedding, target pairs for train and test splits
            X_train = gene_embeddings[train_index]
            y_train = gene_target[train_index]
            X_test = gene_embeddings[test_index]
            y_test = gene_target[test_index]

            # Fit classifier and extract test accuracy in differentiating
            # nontargeting genes from selected gene
            classifier.fit(X_train, y_train)
            test_acc = balanced_accuracy_score(y_test, classifier.predict(X_test))
            single_gene_acc.append(test_acc)

        per_gene_stats.append((current_gene, single_gene_acc))

    # Generate DataFrame containing per-gene results
    accuracies_ok = pd.DataFrame(columns=[f"Fold{i}" for i in range(1, splits + 1)],
                                 index=[stat[0] for stat in per_gene_stats],
                                 data=[stat[1] for stat in per_gene_stats])

    return accuracies_ok
