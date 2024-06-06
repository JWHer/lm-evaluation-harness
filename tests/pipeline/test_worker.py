import pytest
import torch

def test_stream():
    if not torch.cuda.is_available():
        pytest.skip("You can test when cuda is avaliable")
    
    # suppose first device num 0
    first_device = torch.cuda('cuda:0')
    last_device  = torch.cuda(f'cuda:{torch.cuda.device_count()-1}')

    tensor_cpu = torch.randn([1, 10])
    # tensor_gpu = torch.randn([1, 10]).to(first_device)

    stream = torch.cuda.Stream(device=last_device)
    with torch.cuda.stream(stream):
        tensor_instream = tensor_cpu.cuda()

    assert tensor_instream.device == stream.device

