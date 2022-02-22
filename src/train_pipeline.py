import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

import pipeline as model_pipeline
from monitor.monitor import (
    detect_dataset_drift_on_premise,
    evaluate_model_drift_on_premise
)
from raw_data.make_dataset import read_params_file


def eval_metrics(actual, pred):
    balanced_accuracy = balanced_accuracy_score(actual, pred)
    f1_score_ = f1_score(actual, pred)
    precision_score_ = precision_score(actual, pred)
    recall_score_ = recall_score(actual, pred)
    return balanced_accuracy, f1_score_, precision_score_, recall_score_


def read_file(config_path: str) -> pd.DataFrame:
    """_summary_

    Args:
        config_path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return pd.read_csv(config_path)


def run_training(config_path) -> None:
    """_summary_

    Args:
        config_path (_type_): _description_
    """
    config_path = read_params_file(config_path)

    model_name = config_path.get("best_model_name")
    model_dir = config_path.get("saved_model_dir")
    model_version = config_path.get("model_version")
    model_format = config_path.get("model_format")

    model_name = model_name + "_" + model_version + model_format

    # model_path = os.path.join(model_dir, model_name)

    # # divide train and test
    X_train = read_file(config_path.get("split_data").get("X_train_path"))
    y_train = read_file(config_path.get("split_data").get("y_train_path"))

    X_test = read_file(config_path.get("split_data").get("X_test_path"))
    y_test = read_file(config_path.get("split_data").get("y_test_path"))

    mlflow.set_experiment("Baseline_model")

    # enable autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        # transform
        loan_pipe = model_pipeline.run_pipeline(config_path)
        loan_pipe.fit(X_train, y_train["is_bad"])
        print("Logged data and model in run: {}".format(run.info.run_id))

        # save model
        # _logger.info(f"saving model version: {_version}")
        joblib.dump(loan_pipe, os.path.join(model_dir, model_name))

        # data saved for model and data drift testing
        X_train_transformed = loan_pipe[:-1].transform(X_train)
        X_test_tranformed = loan_pipe[:-1].transform(X_test)

        X_train_transformed["prediction"] = loan_pipe.predict(X_train)
        X_test_tranformed["prediction"] = loan_pipe.predict(X_test)

        balanced_accuracy, f1_score_, precision_score_, recall_score_ = eval_metrics(
            y_test, X_test_tranformed["prediction"]
        )

        print("  balanced_accuracy: %s" % balanced_accuracy)
        print("  f1_score_: %s" % f1_score_)
        print("  precision_score_: %s" % precision_score_)
        print("  recall_score_: %s" % recall_score_)


    X_train_transformed.to_csv(
        config_path.get("split_data").get("reference_xtrain"), index=False
    )
    y_train.to_csv(config_path.get("split_data").get("reference_ytrain"), index=False)

    X_test_tranformed.to_csv(
        config_path.get("split_data").get("reference_xtest"), index=False
    )
    y_test.to_csv(config_path.get("split_data").get("reference_ytest"), index=False)

    detect_dataset_drift_on_premise(config_path)
    evaluate_model_drift_on_premise(config_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parse_args = args.parse_args()

    run_training(config_path=parse_args.config)
