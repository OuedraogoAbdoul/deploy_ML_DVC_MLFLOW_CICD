# Machine learning End-End

This project highlight an end to end machine learning development including data, target and model drift monitoring. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install make.

```bash
pip install make
```

## Usage

```python
make build: to build the docker environment

make preprocess: 'to download data and preprocess it'

make train_model: 'to train the model'

make predict: 'to make prediction'

'Alternative using dvc'

dvc repro: 'To train and predict'

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Reports

### Data drift

![alt text](reports/Screenshot from 2022-02-24 11-54-30.png)

### Target drift
![alt text](reports/Screenshot from 2022-02-24 11-56-15.png)

''
![alt text](reports/Screenshot from 2022-02-24 11-57-00.png)

'Target drift by features'
![alt text](reports/Screenshot from 2022-02-24 11-57-45.png)



### Drift by features
![alt text](reports/Screenshot from 2022-02-24 11-56-38.png)

## Metrics

### Confusion

![alt text](mlruns/1/b00e034957b14c25b85b88893ca48967/artifacts/training_confusion_matrix.png)

### Precision and Recall

![alt text](mlruns/1/b00e034957b14c25b85b88893ca48967/artifacts/training_precision_recall_curve.png)

### ROC AUC

![alt text](mlruns/1/b00e034957b14c25b85b88893ca48967/artifacts/training_roc_curve.png)

## License
[MIT](https://choosealicense.com/licenses/mit/)