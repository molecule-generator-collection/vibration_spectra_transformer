import torch

from scripts.common import DistributedContext, initialize_distributed


def test_single_process_context_keeps_requested_device(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    context = initialize_distributed("cpu")

    assert context == DistributedContext(device=torch.device("cpu"))
    assert not context.enabled
    assert context.is_main
    assert context.all_ranks_true(True)
    assert context.sum(2.5) == 2.5
    assert context.broadcast_float(1.25) == 1.25


def test_multi_gpu_rejects_cpu_before_process_group_initialization(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")

    try:
        initialize_distributed("cpu")
    except ValueError as error:
        assert "--device cpu" in str(error)
    else:
        raise AssertionError("multi-GPU CPU request should have failed")
