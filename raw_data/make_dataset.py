import argparse
import json

import pandas as pd
import yaml


def read_params_file(config_path: str) -> json:
    """Read the and open the params.yaml file

    Args:
        config_path (str): yaml config file

    Returns:
        yaml: yaml file
    """

    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def download_data(config_path: str) -> pd.DataFrame:
    """Download the dataset from s3

    Args:
        config_path (str): yaml config file

    Returns:
        pd.DataFrame: panda dataFrame
    """
    config_path = read_params_file(config_path)
    file_path = config_path["data_source"]["datasets_link"]
    df_data = pd.read_csv(file_path)

    df_data.to_csv(config_path["data_source"]["raw_source"], index=False)
    return df_data


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parse_args = args.parse_args()

    data = download_data(config_path=parse_args.config)
