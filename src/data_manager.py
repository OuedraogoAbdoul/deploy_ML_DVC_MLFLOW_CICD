import argparse

import pandas as pd
from sklearn import pipeline
# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from raw_data.make_dataset import read_params_file


def load_data(config_path: str) -> pd.DataFrame:
    """Load the dataset from the raw folder

    Args:
        config_path (str): yaml config file

    Returns:
        pd.DataFrame: panda dataFrame
    """

    file_path = config_path["data_source"]["raw_source"]
    df_data = pd.read_csv(file_path)

    return df_data


def split_data(config_path: str) -> pd.DataFrame:
    """load and split the dataset

    Args:
        config_path (str): yaml config file

    Returns:
        pd.DataFrame: panda dataFrame
        pd.DataFrame: panda dataFrame
        pd.DataFrame: panda dataFrame
        pd.DataFrame: panda dataFrame
    """
    config_path = read_params_file(config_path)

    df = load_data(config_path)

    X = df.drop(columns=[config_path.get("base_model_params").get("target")])
    y = df[config_path.get("base_model_params").get("target")]

    random_state = config_path.get("base_model_params").get("random_state")
    test_size = config_path.get("split_data").get("test_size")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv(config_path.get("split_data").get("X_train_path"), index=False)
    y_train.to_csv(config_path.get("split_data").get("y_train_path"), index=False)

    X_test.to_csv(config_path.get("split_data").get("X_test_path"), index=False)
    y_test.to_csv(config_path.get("split_data").get("y_test_path"), index=False)

    return X_train, X_test, y_train, y_test


def save_pipeline(config_path: str) -> Pipeline:
    """Save model pipeline after training

    Args:
        config_path (str): yaml file

    Returns:
        pipeline: sklearn pipeline
    """

    model_pipeline = Pipeline()

    return model_pipeline


def load_pipeline(config_path: str) -> Pipeline:
    """Load model pipeline for prediction

    Args:
        config_path (str): yaml file

    Returns:
        pipeline: sklearn pipeline
    """

    model_pipeline = pipeline()

    return model_pipeline


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parse_args = args.parse_args()

    data = split_data(config_path=parse_args.config)
