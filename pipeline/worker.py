from contextlib import contextmanager
from queue import Queue
import sys
from threading import Thread
from types import TracebackType
from typing import Dict, Generator, List, Tuple, Type, Any

import torch
from torch import Tensor, nn

from .utils import make_minibatch


ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]
ExcDone = (False, None)
InQueue = Queue[Tuple[int, Tuple[Any, ...]]]
OutQueue = Queue[Tuple[int, Tuple[Any, ...]]]


def normalize_device(device: torch.device) -> torch.device:
    if device.type == 'cuda' and device.index is None:
        return torch.device('cuda', index=torch.cuda.current_device())

    if device.type == 'cpu' and device.index is not None:
        return torch.device('cpu')

    return device


class Worker:
    def __init__(self, minibatch_size: int) -> None:
        self.minibatch_size = minibatch_size
        self.results: Dict[str, List[Any]] = {"outputs": []}
        self.first_queue = None
        self.last_queue = None

    def parse_kwargs(self, kwargs: dict, _input: tuple, device: torch.device, minibatch_id: int) -> dict:
        parsed = kwargs.copy()
        minibatch_keys = parsed.pop('_minibatch') if '_minibatch' in parsed.keys() else []
        minibatch_cache = parsed.pop('_cache') if '_cache' in parsed.keys() else {}

        for key in minibatch_keys:
            if key in minibatch_cache:
                minibatch = minibatch_cache[key]
            else:
                minibatch = make_minibatch(parsed[key], self.minibatch_size)
                
                if '_cache' not in kwargs.keys():
                    kwargs['_cache'] = {}
                kwargs['_cache'][key] = minibatch

            parsed[key] = minibatch[minibatch_id]
        
        # TODO copy value with cuda.Stream
        for key, value in parsed.items():
            if isinstance(value, str) and '_input' in value:
                value = eval(value)
            if isinstance(value, Tensor):
                value.to(device)
            parsed[key] = value

        return parsed

    def forward(
        self,
        in_queue: InQueue,
        out_queue: OutQueue,
        device: torch.device,
        partition: List[nn.ModuleList],
        kwargs: List[dict]
    ) -> None:
        """The main loop of a worker thread."""
        torch.set_grad_enabled(False)  # we do not use it for train

        # with torch.cuda.device(device):
        while True:
            minibatch_id, input_batch = in_queue.get()
            if input_batch is None:
                break
            next_batch = input_batch    # for readable codes

            # recurring next_batch
            for part, kw in zip(partition, kwargs):
                parsed_kwargs = self.parse_kwargs(kw, next_batch, device, minibatch_id)
                next_batch = part(**parsed_kwargs)

                if minibatch_id not in self.results.keys():
                    self.results[minibatch_id] = []
                self.results[minibatch_id].append(next_batch)

            output_batch = next_batch   # for readable codes
            out_queue.put((minibatch_id, output_batch))
        out_queue.put(ExcDone)

    @contextmanager
    def spawn_threads(
        self,
        devices: List[torch.device],
        partitions: List[nn.ModuleList],
        kwargs_list: List[dict]
    ) -> Generator[Dict[int, Tuple[InQueue, OutQueue, torch.device]], None, None]:
        """Spawns worker threads. A worker thread is bound to a device."""
        # Spawn workers.
        threads: Dict[int, Tuple[InQueue, OutQueue, torch.device]] = {}

        self.first_queue = before_out_queue = Queue()
        for idx, (device, partition, kwargs) in enumerate(zip(devices, partitions, kwargs_list)):
            device = normalize_device(device)

            # TODO make multi inputs and outputs to make graph
            in_queue = before_out_queue
            out_queue = Queue()
            before_out_queue = out_queue
            threads[idx] = (in_queue, out_queue, device)

            t = Thread(
                target=self.forward,
                args=(in_queue, out_queue, device, partition, kwargs),
                daemon=True,
            )
            t.start()
        self.last_queue = out_queue

        try:
            yield threads
        finally:
            # Close workers.
            self.first_queue.put(ExcDone)

            # Join running worker
            while True:
                # if worker finished, it return ExcDone = (False, None)
                _minibatch_id, payload = self.last_queue.get()
                if payload is None:
                    break
                else:
                    self.results['outputs'].append(payload)
