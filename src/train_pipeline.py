import pipeline as pipeline
import argparse
import pandas as pd
from raw_data.make_dataset import read_params_file
import joblib
import os


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

    model_path = os.path.join(model_dir, model_name)

    # # divide train and test
    X_train = read_file(config_path.get("split_data").get("X_train_path"))
    y_train = read_file(config_path.get("split_data").get("y_train_path"))

    # transform
    loan_pipe = pipeline.run_pipeline(config_path)
    loan_pipe.fit(X_train, y_train)

    # save model
    # _logger.info(f"saving model version: {_version}")
    joblib.dump(loan_pipe, os.path.join(model_dir, model_name))
    print(type(loan_pipe))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parse_args = args.parse_args()

    pipeline = run_training(config_path=parse_args.config)
