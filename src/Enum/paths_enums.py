from enum import Enum


class PathEnum(str, Enum):
    TRAIN_PATH = r"..\src\data\splits\split\train.csv"
    VAL_PATH = r"..\src\data\splits\split\val.csv"
    TRAIN_VAL_PATH = r"..\src\data\splits\split\train_val.csv"
    TEST_PATH = r"..\src\data\splits\split\test.csv"

