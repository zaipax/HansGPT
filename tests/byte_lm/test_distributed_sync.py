import pytest
import torch

from hansgpt_research.distributed_sync import (
    DEFAULT_GRADIENT_REDUCE_BUCKET_MIB,
    allocate_gradient_reduce_buffer,
    gradient_bucket_elements,
    sync_gradients,
)


def test_default_gradient_bucket_is_256_mib_of_fp32_elements():
    assert DEFAULT_GRADIENT_REDUCE_BUCKET_MIB == 256
    assert gradient_bucket_elements(DEFAULT_GRADIENT_REDUCE_BUCKET_MIB) == 64 * 1024 * 1024


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_gradient_bucket_rejects_invalid_sizes(value):
    with pytest.raises(ValueError):
        gradient_bucket_elements(value)


def test_allocate_gradient_reduce_buffer_uses_fp32_and_requested_capacity():
    buffer = allocate_gradient_reduce_buffer(torch.device("cpu"), 1)
    assert buffer.dtype == torch.float32
    assert buffer.shape == (256 * 1024,)

    double_buf = allocate_gradient_reduce_buffer(torch.device("cpu"), 1, double_buffered=True)
    assert double_buf.dtype == torch.float32
    assert double_buf.shape == (2, 256 * 1024)


def test_sync_gradients_reduces_across_parameter_and_bucket_boundaries(monkeypatch):
    first = torch.nn.Parameter(torch.zeros(3))
    second = torch.nn.Parameter(torch.zeros(6))
    first.grad = torch.tensor([1.0, 2.0, 3.0])
    second.grad = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    calls = []

    def fake_all_reduce(bucket):
        calls.append(bucket.clone())
        bucket.mul_(2)

    monkeypatch.setattr("hansgpt_research.distributed_sync.dist.all_reduce", fake_all_reduce)

    sync_gradients([first, second], torch.empty(4, dtype=torch.float32))

    assert [len(bucket) for bucket in calls] == [3, 4, 2]
    torch.testing.assert_close(first.grad, torch.tensor([2.0, 4.0, 6.0]))
    torch.testing.assert_close(second.grad, torch.tensor([8.0, 10.0, 12.0, 14.0, 16.0, 18.0]))


def test_sync_gradients_double_buffered_matches_single_buffer(monkeypatch):
    first = torch.nn.Parameter(torch.zeros(3))
    second = torch.nn.Parameter(torch.zeros(6))
    first.grad = torch.tensor([1.0, 2.0, 3.0])
    second.grad = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    calls = []

    class FakeWork:
        def wait(self):
            pass

    def fake_async_all_reduce(bucket, async_op=True):
        calls.append(bucket.clone())
        bucket.mul_(2)
        return FakeWork()

    monkeypatch.setattr("hansgpt_research.distributed_sync.dist.all_reduce", fake_async_all_reduce)

    double_buf = torch.empty((2, 4), dtype=torch.float32)
    sync_gradients([first, second], double_buf)

    assert [len(bucket) for bucket in calls] == [4, 4, 1]
    torch.testing.assert_close(first.grad, torch.tensor([2.0, 4.0, 6.0]))
    torch.testing.assert_close(second.grad, torch.tensor([8.0, 10.0, 12.0, 14.0, 16.0, 18.0]))
