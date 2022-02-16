import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from imblearn.pipeline import make_pipeline

import argparse
from raw_data.make_dataset import read_params_file
from sklearn import pipeline
from sklearn.svm import SVC

import preprocessor as pp
import data_manager as manager


def run_pipeline(config_path: str) -> pipeline:
    # config_path = read_params_file(config_path)
    svc = SVC()
    feature_eng_pipeline = make_pipeline(
        # pp.MissingValuesImputerWarpper(),
        pp.TemporalFeaturesExtraction(variables="earliest_cr_line"),
        pp.ExtractZipCode(),
        TargetEncoder(True, handle_missing="missing", handle_unknown="missing"),
        pp.ScalerWrapper(),
        pp.MissingValuesImputerWarpper(),
        svc,
    )

    # X_train, y_train = pd.read_csv(config_path.get("split_data").get("X_train_path")), pd.read_csv(config_path.get("split_data").get("y_train_path"))

    # print(feature_eng_pipeline.fit_transform(X_train, y_train[config_path.get("base_model_params").get("target")]))
    return feature_eng_pipeline


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parse_args = args.parse_args()

    pipeline = run_pipeline(config_path=parse_args.config)