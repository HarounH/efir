"""
Auto-registration code based off of
https://charlesreid1.github.io/python-patterns-the-registry.html
"""

from yacs.config import CfgNode
import torch
import logging
from typing import Callable, Any
from efir.utils import CodeBlock, cfg_node_to_dict


logger = logging.getLogger()

TConstructor = Callable[..., Any]

class Registry(type):
    _REGISTRY = {}

    @classmethod
    def register(cls, key: str, value: TConstructor):
        cls._REGISTRY[key] = value

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.register(new_cls.__name__.lower(), new_cls)
        return new_cls

    @classmethod
    def get(cls, key: str) -> TConstructor:
        return cls._REGISTRY.get(key, None)

    @classmethod
    def build_from_cfg(cls, cfg_node: CfgNode, **dynamic_kwargs) -> Any:
        key = cfg_node.type
        kwargs = cfg_node_to_dict(cfg_node)
        kwargs.pop("type")
        with CodeBlock(f"Building object of type {key}, using {kwargs=}", logger):
            obj = cls.get(key)(**kwargs, **dynamic_kwargs)
        return obj


class AutoRegistrationBase(metaclass=Registry):
    pass


# Register optimizers
Registry.register("sgd", torch.optim.SGD)
Registry.register("adam", torch.optim.Adam)
Registry.register("CosineAnnealingWarmRestarts", torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
# todo: register other stuff too