from sklearn.model_selection import train_test_split


def build_fixed_split(dataframe, label_col=0, test_size=0.2, random_state=40):
    labels = dataframe[label_col]
    # Fall back to non-stratified split when some classes are too rare.
    stratify_labels = labels
    class_counts = labels.value_counts(dropna=False)
    if not class_counts.empty and int(class_counts.min()) < 2:
        stratify_labels = None
    train_idx, test_idx = train_test_split(
        dataframe.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    train_idx = sorted(train_idx.tolist())
    test_idx = sorted(test_idx.tolist())
    return train_idx, test_idx
