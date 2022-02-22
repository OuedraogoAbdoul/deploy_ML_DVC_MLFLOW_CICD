import argparse

from category_encoders import TargetEncoder
from imblearn.pipeline import make_pipeline
from sklearn import pipeline
from sklearn.svm import SVC

import preprocessor as pp
from raw_data.make_dataset import read_params_file


def run_pipeline(config_path: str) -> pipeline:

    # print(config_path)

    var = config_path.get("categorical_vars").get("earliest_cr_line")
    svc = SVC()
    feature_eng_pipeline = make_pipeline(
        # pp.MissingValuesImputerWarpper(),
        pp.TemporalFeaturesExtraction(variables=var),
        pp.ExtractZipCode(),
        TargetEncoder(True, handle_missing="missing", handle_unknown="missing"),
        pp.ScalerWrapper(),
        pp.MissingValuesImputerWarpper(),
        svc,
    )

    # X_train, y_train = pd.read_csv(config_path.get("split_data").get("X_train_path")),
    # pd.read_csv(config_path.get("split_data").get("y_train_path"))

    # print(feature_eng_pipeline.fit_transform(X_train,
    # y_train[config_path.get("base_model_params").get("target")]))
    return feature_eng_pipeline


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parse_args = args.parse_args()

    pipeline = run_pipeline(config_path=parse_args.config)
