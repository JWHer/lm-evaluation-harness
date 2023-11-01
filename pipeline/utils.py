from typing import Dict
from torch import Tensor


def make_minibatch(tensor: Tensor, minibatch_size: int, dataset_length: int = -1) -> Dict[str, Tensor]:
    batch_size = tensor.shape[0]

    # Implicit dataset_length
    if dataset_length < 0:
        dataset_length = batch_size // minibatch_size + (batch_size % minibatch_size > 0)

    minibatch = {}
    for idx in range(dataset_length):
        minibatch[str(idx)] = tensor[idx * minibatch_size:(idx + 1) * minibatch_size, :]

    return minibatch
