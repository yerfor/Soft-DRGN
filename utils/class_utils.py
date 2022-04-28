import importlib


def get_cls_from_path(path):
    """
    :param path: str, eg: "agents.dgn.DGNAgent"
    :return: class, eg: agents.dgn.DGNAgent
    """
    pkg = ".".join(path.split(".")[:-1])
    cls_name = path.split(".")[-1]
    cls_object = getattr(importlib.import_module(pkg), cls_name)
    return cls_object
