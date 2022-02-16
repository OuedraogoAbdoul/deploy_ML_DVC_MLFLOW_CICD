from unittest import result
import joblib
from matplotlib.pyplot import axis
import pandas as pd
import json
import argparse
import joblib
from sklearn.pipeline import Pipeline
from raw_data.make_dataset import read_params_file
import os


def make_prediction(input_data: pd.DataFrame) -> json:
    """_summary_

    Args:
        input_data (json): _description_

    Returns:
        json: _description_
    """

    ROOT = os.path.join(os.path.dirname(__file__)).split("src")[0]
    config_file = os.path.join(ROOT, "config/params.yaml")
    # print(config_file)

    config_path = read_params_file(config_file)

    model_path = config_path.get("production").get("model_path")
    model_pipeline = joblib.load(os.path.join("saved_models", "best_model_v1.joblib"))

    data = pd.read_csv("processed_data/x_test.csv").sample(1)
    sample = pd.read_csv("processed_data/x_test.csv").head(10)

    dataset = pd.concat([data, sample], axis=0)

    try:
        
        prediction = model_pipeline.predict(dataset)
    

        prediction = {"predictions": prediction[0]}
    except:
        pass

    if(1 in prediction):
        prediction = {"predictions": 1}
    else:
        prediction = {"predictions": 0}

    # print(prediction)

    return prediction

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="test.json")
    parse_args = args.parse_args()

    pipeline = make_prediction(input_data=parse_args.config)
