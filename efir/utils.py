from dataclasses import dataclass, field
import sys
import logging
from yacs.config import CfgNode
from typing import List, Dict, Any, Optional, Tuple, Callable
from copy import deepcopy
from datetime import datetime
from numbers import Number
from collections import deque
from collections.abc import Set, Mapping
import numpy as np
import torch
from torch.profiler import profile,  record_function, ProfilerActivity


logger = logging.getLogger()
N_BYTES_IN_MB = 1024 * 1024


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


ZERO_DEPTH_BASES = (str, bytes, Number, range, bytearray)


def get_size(obj_0: Any) -> int:
    """ Recursively iterate to sum size of object & members.
    reference: https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
    """
    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, ZERO_DEPTH_BASES):
            pass # bypass remaining control flow and return
        elif isinstance(obj, np.ndarray):
            return size + obj.nbytes
        elif isinstance(obj, torch.Tensor):
            return size + obj.nelement() * obj.element_size()
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)


def get_on_trace_ready(name: str) -> Callable:
    def on_trace_ready(prof: profile):
        logger.info(
            prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10)
        )
        prof.export_chrome_trace(f"./results/traces/{name}" + str(prof.step_num) + ".pt.trace.json")
    return on_trace_ready


def log_memories(objects_to_inspect: Dict[str, Any], logger: logging.Logger) -> None:
    for k, v in objects_to_inspect.items():
        logger.info(f"MEMORY USAGE: {k} -> {get_size(v) / N_BYTES_IN_MB:.4f} MB")


class CodeBlock:
    @dataclass
    class ProfileKwargs:
        on_trace_ready: Callable
        schedule: Optional[Callable] = field(default_factory=lambda: torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2,
            repeat=1,
        ))
        activities: List[ProfilerActivity] = field(default_factory=lambda: [ProfilerActivity.CPU, ProfilerActivity.CUDA])
        record_shapes: bool = True
        profile_memory: bool = True
        with_stack: bool = True

    def __init__(
        self,
        description: str,
        logger: logging.Logger,
        profile: bool = False,
        profile_kwargs: Optional[ProfileKwargs] = None,
        objects_to_inspect: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logger
        self.description = description
        self.profile = profile
        self.profile_kwargs = profile_kwargs
        self._profiler = None
        self.objects_to_inspect = objects_to_inspect

    def __enter__(self):
        if self.objects_to_inspect is not None:
            log_memories(objects_to_inspect={f"start_{k}": v for k, v in self.objects_to_inspect.items()}, logger=logger)
        if self.profile:
            if self.profile_kwargs is None:
                self.profile_kwargs = CodeBlock.ProfileKwargs(
                    on_trace_ready=get_on_trace_ready(self.description.replace(" ", "_")[:64])
                )

            self._profiler = profile(
                activities=self.profile_kwargs.activities,
                record_shapes=self.profile_kwargs.record_shapes,
                profile_memory=self.profile_kwargs.profile_memory,
                with_stack=self.profile_kwargs.with_stack,
                schedule=self.profile_kwargs.schedule,
                on_trace_ready=self.profile_kwargs.on_trace_ready,
            )
            self._profiler.start()
        else:
            self._profiler = None
        self.tic = datetime.now()
        self.logger.info(f"Entering {self.description} at {self.tic.strftime('%m/%d/%Y, %H:%M:%S')}")
        return self._profiler

    def __exit__(self, ctx_type, ctx_value, ctx_traceback):
        tac = datetime.now()
        self.logger.info(f"Exiting {self.description} at {tac.strftime('%m/%d/%Y, %H:%M:%S')}; wall-time = {(tac-self.tic).total_seconds()}")
        if self._profiler is not None:
            self._profiler.stop()
        if self.objects_to_inspect is not None:
            log_memories(objects_to_inspect={f"end_{k}": v for k, v in self.objects_to_inspect.items()}, logger=logger)
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
