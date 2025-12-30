from enum import Enum


class ModelEnum(str, Enum):
    LINEAR = "linear"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    RIDGE = "ridge"

