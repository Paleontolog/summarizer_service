from enum import Enum, EnumMeta


class ModelTypeMeta(EnumMeta):
    def __getitem__(self, item: str):
        item = item.upper()
        value = ModelType.__members__.get(item)
        if value is None:
            raise Exception(f"Incorrect enum value {item}. Allowed values: {ModelType.__members__.keys()}")
        return value


class ModelType(Enum, metaclass=ModelTypeMeta):
    BART = "BART"
    BINARY_CLASSIFIER = "BINARY_CLASSIFIER"


