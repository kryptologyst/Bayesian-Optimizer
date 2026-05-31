import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from loguru import logger

DATASETS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}

def load_data(dataset_name="wine"):
    loader = DATASETS.get(dataset_name, load_wine)
    data = loader()
    X, y = data.data, data.target
    fn = list(data.feature_names)
    logger.info(f"{dataset_name}: {X.shape[0]} samples")
    return X, y, fn
