from queue import Queue
from typing import List, Optional, Tuple, Any

import torch
from torch import Tensor, nn

from .worker import Worker
from .utils import make_minibatch


class Pipeline:
    """The pipeline parallelism for inference."""

    def __init__(self,
                 batch: Tensor,
                 minibatch_size: int,
                 partitions: List[nn.ModuleList],
                 kwargs_list: List[dict] = None,
                 devices: Optional[List[torch.device]] = None,
                 copy_streams: Optional[List[List[torch.cuda.Stream]]] = None,
                 ) -> None:
        self.batch = batch
        self.minibatch_size = minibatch_size

        if devices is None:
            if torch.cuda.is_available():
                devices = [torch.device(f'cuda:{idx}') for idx in range(torch.cuda.device_count())]
            else:
                # FIXME
                devices = [torch.device('cpu') for _ in partitions]
        self.devices = devices

        if kwargs_list is None:
            kwargs_list = [{}] * len(partitions)

        # Split partitions and kwargs
        plength = len(partitions)
        worker_size = len(self.devices)
        chunk_size = round(plength / worker_size)

        self.partitions = []
        self.kwargs_list = []
        for idx in range(worker_size):
            start = idx*chunk_size
            end = (idx+1)*chunk_size if idx+1 < worker_size else plength
            self.partitions.append(partitions[start:end])
            self.kwargs_list.append(kwargs_list[start:end])
        assert len(self.partitions) == len(self.devices) == len(self.kwargs_list)

        # if copy_streams is None:
        #     copy_streams = [[torch.cuda.current_stream(d)] * len(batches) for d in devices]
        # self.copy_streams = copy_streams

    def run(self) -> dict:
        """Runs pipeline parallelism.
        It modifies the given batches in place.
        return: dict
            dict[str(batch_idx)] = [output] * len(partitions)
            dict["outputs"] = [result] * len(batches)

        """
        devices = self.devices
        partitions = self.partitions
        kwargs_list = self.kwargs_list
        worker = Worker(self.minibatch_size)

        with worker.spawn_threads(devices, partitions, kwargs_list) as threads:
            first_queue, _next, _device = threads[0]
            for idx, minibatch in make_minibatch(self.batch, self.minibatch_size).items():
                first_queue.put(
                    (idx, minibatch)
                )

        return worker.results

    def __repr__(self) -> str:
        # TODO
        pass
