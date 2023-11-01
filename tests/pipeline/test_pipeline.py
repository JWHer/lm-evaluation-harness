import pytest
import torch
from torch import Tensor, nn

# import sys
# sys.path.append('/data/Codes/lm-evaluation-harness')
from pipeline.pipeline import Pipeline


class SimpleAdd(nn.Module):
    def __init__(self, addnum: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.addnum = addnum

    def forward(self, tensor: Tensor):
        tensor = torch.add(tensor, self.addnum)
        return tensor


@pytest.mark.parametrize("use_gpu", [True, False])
def test_pipeline(use_gpu):
    dataset_length = 3
    # batches = [torch.randint(1, 10, (3,)) for _ in range(dataset_length)]
    minibatch_size = 3
    batch = torch.randint(1, 10, (dataset_length*minibatch_size, 3,))

    # devices_length = torch.cuda.device_count() if use_gpu else 3
    # devices = [torch.device(f'cuda:{i}') for i in range(devices_length)] if use_gpu else None

    partition_length = 10
    partitions = nn.ModuleList([SimpleAdd(i) for i in range(partition_length)])

    kwargs_length = partition_length
    kwargs_list = [{"tensor": "_input"}] * kwargs_length

    pipeline = Pipeline(batch, minibatch_size, partitions, kwargs_list)
    results = pipeline.run()
    outputs = results['outputs']

    addnum = sum(range(partition_length))
    for idx in range(dataset_length):
        # end of minibatch pipeline == outputs
        assert torch.equal(results[str(idx)][-1], outputs[idx])

        minibatch = batch[idx*minibatch_size:(idx+1)*minibatch_size]
        assert torch.equal(torch.add(minibatch, addnum), outputs[idx])
