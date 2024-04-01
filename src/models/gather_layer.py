import torch
import torch.distributed as dist


# class GatherLayer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         output = [torch.zeros_like(input)\
#             for _ in range(dist.get_world_size())]
#         dist.all_gather(output, input)
#         return tuple(output)

#     @staticmethod
#     def backward(ctx, *grads):
#         input, = ctx.saved_tensors
#         grad_out = torch.zeros_like(input)
#         grad_out[:] = grads[dist.get_rank()]
#         return grad_out

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]