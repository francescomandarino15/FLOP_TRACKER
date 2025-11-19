import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from flops_tracker import FlopsTracker


def main():
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )

    clf = LogisticRegression(max_iter=1000)

    # Tracciamo FLOPs del fit
    ft_fit = FlopsTracker(run_name="sklearn_logreg_fit").sklearn_bind(
        model=clf,
        X=X,
        y=y,
        mode="fit",
        log_per_call=True,
        export_path="sklearn_logreg_fit_flops.csv",
        use_wandb=False,
    )

    # Tracciamo FLOPs del predict
    ft_pred = FlopsTracker(run_name="sklearn_logreg_predict").sklearn_bind(
        model=clf,
        X=X,
        mode="predict",
        log_per_call=True,
        export_path="sklearn_logreg_predict_flops.csv",
        use_wandb=False,
    )

    print("Fit FLOPs:", ft_fit.total_flops)
    print("Predict FLOPs:", ft_pred.total_flops)


if __name__ == "__main__":
    main()
