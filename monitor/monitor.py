import argparse
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab
from evidently.dashboard.tabs import ClassificationPerformanceTab


def detect_dataset_drift_on_premise(config_path):
    """
    Returns True if Data Drift is detected, else returns False.
    If get_ratio is True, returns ration of drifted features.
    The Data Drift detection depends on the confidence level and the threshold.
    For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
    Data Drift for the dataset is detected if share of the drifted features is above the selected threshold (default value is 0.5).
    """

    reference_xtrain = pd.read_csv(
        config_path.get("split_data").get("reference_xtrain")
    )
    reference_ytrain = pd.read_csv(
        config_path.get("split_data").get("reference_ytrain")
    ).rename(columns={config_path.get("base_model_params").get("target"): "target"})

    production_xtest = pd.read_csv(
        config_path.get("split_data").get("reference_xtrain")
    )
    production_ytest = pd.read_csv(
        config_path.get("split_data").get("reference_ytrain")
    ).rename(columns={config_path.get("base_model_params").get("target"): "target"})

    reference = pd.concat([reference_xtrain, reference_ytrain], axis=1)
    production = pd.concat([production_xtest, production_ytest], axis=1)

    column_mapping = ColumnMapping()

    # column_mapping.numerical_features = list(config_path.get("numercial_vars").values())
    column_mapping.categorical_features = list(
        config_path.get("categorical_vars").values()
    )

    model_data_and_target_drift_dashboard = Dashboard(
        tabs=[DataDriftTab(verbose_level=1), CatTargetDriftTab(verbose_level=1)]
    )
    model_data_and_target_drift_dashboard.calculate(
        reference, production, column_mapping=column_mapping
    )

    model_data_and_target_drift_dashboard.save(
        config_path.get("reports").get("model_data_and_target_drift_dashboard")
    )


def evaluate_model_drift_on_premise(config_path):

    reference_xtrain = pd.read_csv(
        config_path.get("split_data").get("reference_xtrain")
    )
    reference_ytrain = pd.read_csv(
        config_path.get("split_data").get("reference_ytrain")
    ).rename(columns={config_path.get("base_model_params").get("target"): "target"})

    production_xtest = pd.read_csv(
        config_path.get("split_data").get("reference_xtrain")
    )
    production_ytest = pd.read_csv(
        config_path.get("split_data").get("reference_ytrain")
    ).rename(columns={config_path.get("base_model_params").get("target"): "target"})

    reference = pd.concat([reference_xtrain, reference_ytrain], axis=1)
    production = pd.concat([production_xtest, production_ytest], axis=1)

    column_mapping = ColumnMapping()

    # column_mapping.numerical_features = list(config_path.get("numercial_vars").values())
    column_mapping.categorical_features = list(
        config_path.get("categorical_vars").values()
    )

    loan_model_performance_dashboard = Dashboard(
        tabs=[ClassificationPerformanceTab(verbose_level=1)]
    )
    loan_model_performance_dashboard.calculate(
        reference, production, column_mapping=column_mapping
    )

    loan_model_performance_dashboard.save(
        config_path.get("reports").get("model_performance_drift_dashboard")
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parse_args = args.parse_args()

    data_target_drift = detect_dataset_drift_on_premise(config_path=parse_args.config)
    model_drift = evaluate_model_drift_on_premise(config_path=parse_args.config)
