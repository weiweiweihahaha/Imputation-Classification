from code.core.classifiers import fit_model, get_model_metadata


def fit_retrained_model(
    classifier_name,
    imputed_df,
    train_idx,
    label_col=0,
    *,
    tuning_cv_folds=5,
    tuning_scoring="f1_macro",
    tuning_n_jobs=-1,
    compute_backend="cpu",
    gpu_device=0,
):
    X_train = imputed_df.loc[train_idx, imputed_df.columns != label_col]
    y_train = imputed_df.loc[train_idx, label_col]
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
