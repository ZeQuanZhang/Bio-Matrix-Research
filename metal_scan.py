"""One Metal thread per channel executes the exact first-order state recurrence."""
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def library():
    return torch.mps.compile_shader(r"""
    #include <metal_stdlib>
    using namespace metal;
    kernel void forward_scan(device const float* a, device const float* b,
                             device const float* initial, device float* states,
                             constant uint& length, constant uint& width,
                             uint feature [[thread_position_in_grid]]) {
        uint offset = (feature / width) * length * width + feature % width;
        float value = initial[feature];
        for (uint t = 0; t < length; ++t) {
            uint i = offset + t * width;
            value = a[i] * value + b[i];
            states[i] = value;
        }
    }
    kernel void backward_scan(device const float* a, device const float* states,
                              device const float* initial, device const float* grad,
                              device float* da, device float* db, device float* di,
                              constant uint& length, constant uint& width,
                              uint feature [[thread_position_in_grid]]) {
        uint offset = (feature / width) * length * width + feature % width;
        float adjoint = 0;
        for (int t = int(length) - 1; t >= 0; --t) {
            uint i = offset + uint(t) * width;
            adjoint = grad[i] + (uint(t) + 1 < length ? a[i + width] * adjoint : 0);
            float previous = t > 0 ? states[i - width] : initial[feature];
            da[i] = adjoint * previous;
            db[i] = adjoint;
        }
        di[feature] = adjoint * a[offset];
    }
    """)


def forward(a, b, initial):
    a, b, initial = a.contiguous(), b.contiguous(), initial.contiguous()
    states = torch.empty_like(b)
    batch, length, width = b.shape
    library().forward_scan(a, b, initial, states, length, width, threads=batch * width)
    return states, a, initial


def backward(a, states, initial, gradient):
    da, db, di = torch.empty_like(a), torch.empty_like(states), torch.empty_like(initial)
    batch, length, width = states.shape
    library().backward_scan(a, states, initial, gradient.contiguous(), da, db, di,
                            length, width, threads=batch * width)
    return da, db, di
