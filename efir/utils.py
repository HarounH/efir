import sys
import logging
from yacs.config import CfgNode
from typing import List, Dict, Any, Tuple
from copy import deepcopy
from datetime import datetime


logger = logging.getLogger()


def setup_logger() -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def load_config(filename: str) -> CfgNode:
    cfg = CfgNode(new_allowed=True)  # fuck it, lets do this unsafely
    cfg.merge_from_file(filename)
    return cfg

def cfg_node_to_dict(cfg_node: CfgNode, key_list: List[str] = []) -> Dict[str, Any]:
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            logger.error(f"Key {'.'.join(key_list)} with value {type(cfg_node)} is not a valid type; valid types: {_VALID_TYPES}")
        return cfg_node
    else:
        cfg_dict = deepcopy(dict(cfg_node))
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_node_to_dict(v, key_list + [k])
        return cfg_dict


class CodeBlock:
    def __init__(self, description: str, logger: logging.Logger):
        self.logger = logger
        self.description = description

    def __enter__(self):
        self.tic = datetime.now()
        self.logger.info(f"Entering {self.description} at {self.tic.strftime('%m/%d/%Y, %H:%M:%S')}")
        return

    def __exit__(self, ctx_type, ctx_value, ctx_traceback):
        tac = datetime.now()
        self.logger.info(f"Exiting {self.description} at {tac.strftime('%m/%d/%Y, %H:%M:%S')}; wall-time = {(tac-self.tic).total_seconds()}")
        return


class DatasetDenoisingStats:
    # TODO: refactor
    mean: Dict[str, Tuple[float, ...]] = {
        "MNIST": (0.1307,),
        "FashionMNIST": (0.2860,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "CIFAR10": (0.4914, 0.4822, 0.4465),
        "CIFAR100": (0.5071, 0.4866, 0.4409),
        "Imagenet": (0.485, 0.456, 0.406),
    }
    std: Dict[str, Tuple[float, ...]] = {
        "MNIST": (0.3081,),
        "FashionMNIST": (0.3530,),
        "SVHN": (0.1980, 0.2010, 0.1970),
        "CIFAR10": (0.2470, 0.2435, 0.2616),
        "CIFAR100": (0.2673, 0.2564, 0.2762),
        "Imagenet": (0.229, 0.224, 0.225),
    }
