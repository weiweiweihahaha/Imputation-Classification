from code.core.classifiers import fit_model, get_model_metadata, predict_with_scores


def fit_reference_model(
    classifier_name,
    ground_df,
    train_idx,
    label_col=0,
    *,
    tuning_cv_folds=5,
    tuning_scoring="f1_macro",
    tuning_n_jobs=-1,
    compute_backend="cpu",
    gpu_device=0,
):
    X_train = ground_df.loc[train_idx, ground_df.columns != label_col]
    y_train = ground_df.loc[train_idx, label_col]
    model = fit_model(
        classifier_name,
        X_train,
        y_train,
        tuning_cv_folds=tuning_cv_folds,
        tuning_scoring=tuning_scoring,
        tuning_n_jobs=tuning_n_jobs,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
    )
    return model, get_model_metadata(model)


def score_reference_model(model, ground_df, test_idx, label_col=0):
    X_test = ground_df.loc[test_idx, ground_df.columns != label_col]
    y_test = ground_df.loc[test_idx, label_col]
    outputs = predict_with_scores(model, X_test)
    return X_test, y_test, outputs
