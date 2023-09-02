#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import abc
import concurrent.futures
import functools
import logging
import os
import time
from typing import Callable, Dict, Optional
import torch.cuda

from conf import agents_cfgs, conf_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx
import heyhi
from parlai_diplomacy.wrappers.base_wrapper import BaseWrapper

mp = get_multiprocessing_ctx()

_THE_MODEL = None
_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
CACHE_SIZE = 32  # we have lots of nonsense classifiers that each run in their own executor

# Used when running on a second process
_THE_MODELS = {}
_THE_SECOND_EXECUTOR = None

class ParlaiExecutor(abc.ABC):
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        pass

    @abc.abstractmethod
    def compute(
        self, func_name: str, game: Optional[pydipcc.Game], *args, **kwargs
    ) -> concurrent.futures.Future:
        pass

    @abc.abstractmethod
    def get(self, attr_name: str) -> concurrent.futures.Future:
        pass

    @abc.abstractmethod
    def get_model(self) -> BaseWrapper:
        pass

# This function will load a model into a different process to run on a different GPU
# Called by plausible_order_sampling PlausibleOrderSampler, to run the same model on multiple GPUs, if the gpu_id isn't set,
# or called by nonsense_ensembe, to run different models on different GPUs, if a gpu_id is set
@functools.lru_cache(maxsize=CACHE_SIZE)
def load_wrapper_executor(
    parlai_model_cfg: conf_cfgs.ParlaiModel,
    model_factory: Callable[[conf_cfgs.ParlaiModel], BaseWrapper],
    allow_multi_gpu: bool,
    load_model_on_main: bool,
    key: str,
    gpu_id: Optional[int] = None,
) -> ParlaiExecutor:
    if allow_multi_gpu and (torch.cuda.device_count() == 2 or os.getenv("SECOND_GPU") is not None) and (gpu_id is None or gpu_id > 0) and not load_model_on_main:
        logging.info(f"load_wrapper_executor SecondProcessParlaiExecutor {key}")
        return SecondProcessParlaiExecutor(parlai_model_cfg, model_factory, load_model_on_main, key)
    elif (gpu_id is None and torch.cuda.device_count() > 2 and allow_multi_gpu) or (
        gpu_id is not None and gpu_id < torch.cuda.device_count()
    ):
        logging.info("load_wrapper_executor MultiProcessParlaiExecutor")
        return MultiProcessParlaiExecutor(
            parlai_model_cfg, model_factory, load_model_on_main, gpu_id
        )
    else:
        return PseudoParlaiExecutor(model_factory(parlai_model_cfg))


def wrap_parlai_model_to_executor(parlai_model: BaseWrapper,) -> ParlaiExecutor:
    return PseudoParlaiExecutor(parlai_model)


class PseudoParlaiExecutor(ParlaiExecutor):
    def __init__(self, parlai_model: BaseWrapper):
        logging.info("Buillding PseudoParlaiExecutor (no parallelizm)")
        self._model = parlai_model

    def is_loaded(self) -> bool:
        return True

    def compute(
        self, func_name: str, game: Optional[pydipcc.Game], *args, **kwargs
    ) -> concurrent.futures.Future:
        func = getattr(self._model, func_name)

        if game is not None:
            result = func(game, *args, **kwargs)
        else:
            result = func(*args, **kwargs)
        return InstantFuture(result)

    def get(self, attr_name: str) -> concurrent.futures.Future:
        return InstantFuture(getattr(self._model, attr_name))

    def get_model(self) -> BaseWrapper:
        return self._model

# this is called from plausible_order_sampling and from factory -> load_ensemble_nonsense_classifier_wrapper
class MultiProcessParlaiExecutor(ParlaiExecutor):
    def __init__(
        self,
        cfg: conf_cfgs.ParlaiModel,
        model_factory: Callable[[conf_cfgs.ParlaiModel], BaseWrapper],
        load_model_on_main: bool = True,
        gpu_id: Optional[int] = None,
    ):
        assert _THE_MODEL is None, "Expected the model to be non-loaded in the main process"
        self.cfg = cfg

        if gpu_id is None:
            # This loads the same model on multiple GPUs and executes immidiately, used for the plausible order sampler
            assert torch.cuda.device_count() >= 2, torch.cuda.device_count()
            # Will not use GPU:0
            self._num_workers = torch.cuda.device_count() - 1
            logging.info("Building MultiProcessParlaiExecutor for %d devices", self._num_workers)
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._num_workers, mp_context=mp,
            )
            for _ in self._executor.map(
                _load_model_to_global_var,
                [(cfg, i + 1, model_factory) for i in range(self._num_workers)],
            ):
                pass
            self._model_loading_fut = None
        else:
            # This loads one model on a specified GPU and executes lazily, used for the 16 nonsense models
            assert gpu_id < torch.cuda.device_count(), (gpu_id, torch.cuda.device_count())
            self._num_workers = 1
            logging.info(
                "Building MultiProcessParlaiExecutor for model dispatched to device %d", gpu_id
            )
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._num_workers, mp_context=mp,
            )
            self._model_loading_fut = self._executor.submit(
                _load_model_to_global_var, (cfg, gpu_id, model_factory)
            )

        self._model = model_factory(cfg) if load_model_on_main else None

    def is_loaded(self) -> bool:
        if self._model_loading_fut is None:
            # This means that the subprocesses were waited on inside the constructor
            return True

        # If there is a future, wait for it.
        self._model_loading_fut.result()
        return True

    def compute(
        self, func_name: str, game: Optional[pydipcc.Game], *args, **kwargs
    ) -> concurrent.futures.Future:
        assert self.is_loaded()
        # Resolve immidiately so that multiple scripts aren't all trying tp
        # load data at the same time, which causes the sysm to crash in a way
        # that is random.
        future = self._executor.submit(
            _compute, func_name, game.to_json() if game is not None else None, *args, **kwargs,
        )
        # If experiencing random CUDA errors uncomment this to ensure all CUDA calls
        # are done sequentially. (Though it will slow down processing)
        #future.result()
        return future

    def get(self, attr_name: str) -> concurrent.futures.Future:
        assert self.is_loaded()
        future = self._executor.submit(_get, attr_name)
        # If experiencing random CUDA errors uncomment this to ensure all CUDA calls
        # are done sequentially. (Though it will slow down processing)
        #future.result()
        return future

    def get_model(self) -> BaseWrapper:
        if self._model:
            return self._model
        else:
            return self._executor.submit(get_global_model).result()

    def __del__(self):
        print("Sunsetting the process pool for", self.cfg)
        self._executor.shutdown()


# Like MultiProcess, but runs everything in one other single process on another GPU, instead of running a process for each 
# model.
# Internally multiple of these refer to a single executor which runs all the models, using a dictionary of models.
# This is done because using a new process for each nonsense filter adds ~700MB/filter, almost doubling the memory requirements
# for 16 filters. With 2 GPUs there should only be 2 processes
class SecondProcessParlaiExecutor(ParlaiExecutor):
    def __init__(
        self,
        cfg: conf_cfgs.ParlaiModel,
        model_factory: Callable[[conf_cfgs.ParlaiModel], BaseWrapper],
        load_model_on_main: bool,
        key: str,
    ):
        global _THE_SECOND_EXECUTOR

        if _THE_SECOND_EXECUTOR is None:
            logging.info("Starting the second process executor with first model {key}")
            _THE_SECOND_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=mp,
            )
            _THE_SECOND_EXECUTOR.submit(
                _start_process_dict
            ).result()
            logging.info("Started the second process executor")
        else:
            logging.info("Second process is already started, loading new model {key} onto it")
        
        self.cfg = cfg
        self.key = key

        self._model_loading_fut = _THE_SECOND_EXECUTOR.submit(
            _load_model_to_global_var_dict, (key, cfg, 1, model_factory)
        )

        #self._model_loading_fut.result() # Load synchronously for debugging

        if load_model_on_main:
            logging.warn("load_model_on_main is true on the second GPU, that's probably wrong")
        
        self._model = model_factory(cfg) if load_model_on_main else None

    def is_loaded(self) -> bool:
        if self._model_loading_fut is None:
            # This means that the subprocesses were waited on inside the constructor
            return True

        # If there is a future, wait for it.
        self._model_loading_fut.result()
        return True

    def compute(
        self, func_name: str, game: Optional[pydipcc.Game], *args, **kwargs
    ) -> concurrent.futures.Future:
        global _THE_SECOND_EXECUTOR
        # Resolve immidiately so that multiple scripts aren't all trying tp
        # load data at the same time, which causes the sysm to crash in a way
        # that is random.
        future = _THE_SECOND_EXECUTOR.submit(
            _compute_dict, self.key, func_name, game.to_json() if game is not None else None, *args, **kwargs,
        )
        # If experiencing random CUDA errors uncomment this to ensure all CUDA calls
        # are done sequentially. (Though it will slow down processing)
        #future.result()
        return future

    def get(self, attr_name: str) -> concurrent.futures.Future:
        global _THE_SECOND_EXECUTOR
        future = _THE_SECOND_EXECUTOR.submit(_get_dict, self.key, attr_name)
        # If experiencing random CUDA errors uncomment this to ensure all CUDA calls
        # are done sequentially. (Though it will slow down processing)
        #future.result()
        return future

    def get_model(self) -> BaseWrapper:
        global _THE_SECOND_EXECUTOR
        logging.warn(f"get_model called to get model for {self.key}")
        return _THE_SECOND_EXECUTOR.submit(get_global_model_dict, self.key).result()

    def __del__(self):
        global _THE_SECOND_EXECUTOR
        if _THE_SECOND_EXECUTOR is not None:
            print("Sunsetting the second executor for ", self.key)
            _THE_SECOND_EXECUTOR.shutdown()
        else:
            print("Already shut down second executor ", self.key)


def _load_model_to_global_var(args):
    parlai_model_cfg, gpu_id, model_factory = args
    global _THE_MODEL
    assert _THE_MODEL is None, f"Double loading? ({os.getpid()})"
    # As soon as torch.cuda.XXX is called, we load ~1GB of CUDA context on GPU
    # To avoid this, we set CUDA_VISIBLE_DEVICES=gpu_id up front before torch.cuda
    # gets initialized. From the perspective of this process, there's just a single
    # GPU.

    os.environ[_CUDA_VISIBLE_DEVICES] = str(gpu_id)
    assert torch.cuda.device_count() == 1, gpu_id

    heyhi.setup_logging(label=f"pid:{os.getpid()} single executor")
    _THE_MODEL = model_factory(parlai_model_cfg)
    assert isinstance(_THE_MODEL, BaseWrapper)
    logging.info(f"Process {os.getpid()} : Done loading")


def _start_process_dict():
    global _THE_MODELS
    _THE_MODELS = {}
    heyhi.setup_logging(label=f"pid:{os.getpid()} second GPU executor")
    logging.info(f"Second process started {os.getpid()}")

    gpu_id = "0"
    if os.getenv("SECOND_GPU") is not None:
        gpu_id = os.getenv("SECOND_GPU")
    logging.info(f"Process {os.getpid()} : Setting secondary GPU to {gpu_id}")
    
    # PyTorch has a memory cache / allocation layer to make deallocations faster .. unfortunately it also causes
    # "out of memory" errors that make no sense, and makes debugging memory use very difficult
    #os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    # This makes debugging easier as errors will happen when they occur
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    loaded = False
    while not loaded:
        try:
            torch.cuda.set_device("cuda:0") # the above changes the device to be 0
            cur_device = torch.zeros(1).to("cuda").device
            cur_available = torch.cuda.is_available()
            logging.info(f"Device: {cur_device}, Available: {cur_available}")
            loaded = True
        except RuntimeError as e:
            logging.error(f"GPU 0 RuntimeError loading data: {e}")
            logging.error(f"GPU 0 Trying again...")
        except:
            logging.error(f"GPU 0 Error loading data")
            logging.error(f"GPU 0 Trying again...")

# Load the model into a global dictionary
def _load_model_to_global_var_dict(args):
    key, parlai_model_cfg, gpu_id, model_factory = args
    global _THE_MODELS
    assert _THE_MODELS is not None, f"No _THE_MODELS dictionary, _start_process_dict not called?"
    assert _THE_MODELS.get(key) is None, f"Double loading {key}? ({os.getpid()})"
    
    heyhi.setup_logging(label=f"#2 pid:{os.getpid()}")
    _THE_MODELS[key] = model_factory(parlai_model_cfg)
    assert isinstance(_THE_MODELS[key], BaseWrapper)
    logging.info(f"Process {os.getpid()} : Done loading {key}")


def get_global_model():
    global _THE_MODEL
    assert _THE_MODEL is not None, f"Model is not loaded in process {os.getpid()}"
    return _THE_MODEL

def get_global_model_dict(key: str):
    global _THE_MODELS
    assert _THE_MODELS[key] is not None, f"Model {key} is not loaded in process {os.getpid()}"
    return _THE_MODELS[key]

def _compute(func_name: str, game_json: Optional[str], *args, **kwargs):
    global _THE_MODEL
    assert _THE_MODEL is not None, f"Model is not loaded in process {os.getpid()}"

    func = getattr(_THE_MODEL, func_name)
    if game_json is not None:
        game = pydipcc.Game.from_json(game_json)
        result = func(game, *args, **kwargs)
    else:
        result = func(*args, **kwargs)

    return result

def _compute_dict(key: str, func_name: str, game_json: Optional[str], *args, **kwargs):
    global _THE_MODELS
    assert _THE_MODELS[key] is not None, f"Model {key} is not loaded in process {os.getpid()}"

    func = getattr(_THE_MODELS[key], func_name)
    if game_json is not None:
        game = pydipcc.Game.from_json(game_json)
        result = func(game, *args, **kwargs)
    else:
        result = func(*args, **kwargs)

    return result

def _get(attr_name: str):
    global _THE_MODEL
    assert _THE_MODEL is not None, f"Model is not loaded in process {os.getpid()}"

    return getattr(_THE_MODEL, attr_name)

def _get_dict(key: str, attr_name: str):
    global _THE_MODELS
    assert _THE_MODELS[key] is not None, f"Model {key} is not loaded in process {os.getpid()}"

    return getattr(_THE_MODELS[key], attr_name)

class InstantFuture(concurrent.futures.Future):
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result
