.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = MAIN_TEMPLATE
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements/requirements_dev.txt

## Make Dataset
data:
	$(PYTHON_INTERPRETER) src/raw_data/make_dataset.py


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint: clean precommit
	isort -rc -sl src/
	# autoflake --remove-all-unused-imports -i -r src
	# isort -rc -m 3 src/
	# mypy src/
	# pylint --disable=R,C ./src

precommit:
	pre-commit run --all-files

## docker build image
build_docker_image:
	docker build . -f docker/app_dev/Dockerfile -t model


## Download Data from remote directory


## Test python environment is setup correctly
pre_build: clean lint precommit
	pytest tests/*.py
	# pytest -vv --cov-report term-missing --cov=app tests/test_data.py


preprocess:
	python src/data/make_dataset.py
	python src/features/build_features.py

train_model:
	# dvc repro
	python src/pipeline.py --config=config/params.yaml
	python src/train_pipeline.py --config=config/params.yaml

predict:
	# dvc repro
	python src/models/predict_model.py


build: build_docker_image
	docker run --rm -d -v $(PWD):/app -p 8000:8000 model
