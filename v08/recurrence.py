"""Full-matrix delta memory with sequential, affine-scan and triangular forms.

H[t] = alpha[t] H[t-1] (I - beta[t] k[t] k[t]^T)
       + beta[t] v[t] k[t]^T.

Shapes: initial [..., d, d], key/value [..., T, d], alpha/beta [..., T].
Leading dimensions broadcast; outputs are [..., T, d, d] and [..., d, d].
There is no state clipping, key normalization, or implicit gradient detach.
Callers generate the coefficients without depending on states inside the scan.

The parallel implementation has ceil(log2(T)) time-dependent scan rounds, not
T token iterations. It materializes only per-node d-by-d transitions, never a
global (N*d*d)-square matrix. Its cost is O(T*d^3*log(T)), versus O(T*d^2) for
the sequential rank-one reference. Training retains O(T*d^2*log(T)) scan
activations; this reference scan trades more work/memory for a shorter chain.

The triangular form builds a unit-lower T-by-T system for rank-one writes.
It uses no token loop and no division by cumulative retention, including when
alpha is zero. Its work is O(T^2*d^2 + T^2*d + T*d^2) and forward working storage
O(T^2 + T*d^2), so a bounded chunk size is necessary. It is mathematically the
same recurrence, with different floating-point operation order and cost.

For |alpha| <= 1 and 0 <= beta*||k||^2 <= 2, the homogeneous transition is
nonexpansive in Frobenius norm. With writes, one step obeys
||H[t]||_F <= |alpha[t]| ||H[t-1]||_F + |beta[t]| ||v[t]|| ||k[t]||.
These are local bounds, not a bound on unconstrained learned coefficients.
"""
import torch


def _prepare(initial, key, value, alpha, beta):
    tensors = (initial, key, value, alpha, beta)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('initial, key, value, alpha and beta must be tensors')
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError('delta recurrence requires floating-point tensors')
    if any(tensor.dtype != initial.dtype or tensor.device != initial.device for tensor in tensors[1:]):
        raise ValueError('delta recurrence inputs must share dtype and device')
    if initial.ndim < 2 or key.ndim < 2 or value.ndim < 2 or alpha.ndim < 1 or beta.ndim < 1:
        raise ValueError('expected initial [...,d,d], key/value [...,T,d], alpha/beta [...,T]')
    d, steps = key.shape[-1], key.shape[-2]
    if d == 0 or initial.shape[-2:] != (d, d) or value.shape[-2:] != (steps, d):
        raise ValueError('initial must be square and key/value must have matching T,d')
    if alpha.shape[-1] != steps or beta.shape[-1] != steps:
        raise ValueError('alpha/beta time dimension must equal key/value time dimension')
    try:
        leading = torch.broadcast_shapes(initial.shape[:-2], key.shape[:-2], value.shape[:-2],
                                         alpha.shape[:-1], beta.shape[:-1])
    except RuntimeError as error:
        raise ValueError('delta recurrence leading dimensions are not broadcastable') from error
    return (initial.expand(*leading, d, d), key.expand(*leading, steps, d),
            value.expand(*leading, steps, d), alpha.expand(*leading, steps),
            beta.expand(*leading, steps))


def sequential_delta(initial, key, value, alpha, beta):
    """Rank-one O(T*d^2) reference, with a sequential token loop."""
    initial, key, value, alpha, beta = _prepare(initial, key, value, alpha, beta)
    state, states = initial, []
    for step in range(key.shape[-2]):
        k, v = key[..., step, :], value[..., step, :]
        a, b = alpha[..., step, None, None], beta[..., step, None, None]
        projected = (state @ k.unsqueeze(-1)).squeeze(-1)
        state = a*state + b*(v-a.squeeze(-1)*projected).unsqueeze(-1)*k.unsqueeze(-2)
        states.append(state)
    if not states:
        return initial.unsqueeze(-3)[..., :0, :, :], initial
    return torch.stack(states, dim=-3), state


def parallel_delta(initial, key, value, alpha, beta):
    """Inclusive Hillis-Steele affine scan; only log2(T) dependent rounds.

    Each transition is (A,B): H -> H@A+B. Composing an earlier (A0,B0)
    with a later (A1,B1) gives (A0@A1, B0@A1+B1).
    """
    initial, key, value, alpha, beta = _prepare(initial, key, value, alpha, beta)
    steps, d = key.shape[-2:]
    if not steps:
        return initial.unsqueeze(-3)[..., :0, :, :], initial
    identity = torch.eye(d, dtype=key.dtype, device=key.device)
    key_column = key.unsqueeze(-1)
    transition = alpha[..., None, None]*(identity-beta[..., None, None]*key_column*key.unsqueeze(-2))
    write = beta[..., None, None]*value.unsqueeze(-1)*key.unsqueeze(-2)
    offset = 1
    while offset < steps:
        later = transition[..., offset:, :, :]
        joined_transition = transition[..., :-offset, :, :] @ later
        joined_write = write[..., :-offset, :, :] @ later + write[..., offset:, :, :]
        transition = torch.cat((transition[..., :offset, :, :], joined_transition), dim=-3)
        write = torch.cat((write[..., :offset, :, :], joined_write), dim=-3)
        offset *= 2
    states = initial.unsqueeze(-3) @ transition + write
    return states, states[..., -1, :, :]


def triangular_delta(initial, key, value, alpha, beta):
    """Solve all writes in one unit-lower system, then reconstruct all states.

    With H[t] = alpha[t]*H[t-1] + u[t] k[t]^T, define
    D[t,j] = prod(alpha[j+1:t+1]) for j <= t, gamma[t] = prod(alpha[:t+1]).
    Then L[t,j] = I[t,j] + 1[j<t]*beta[t]*D[t,j]*dot(k[j],k[t]),
    and L@u = beta*(v-gamma*H0@k). All products are built without division.
    """
    initial, key, value, alpha, beta = _prepare(initial, key, value, alpha, beta)
    steps, d = key.shape[-2:]
    if not steps:
        return initial.unsqueeze(-3)[..., :0, :, :], initial
    position = torch.arange(steps, device=key.device)
    after = position[:, None] > position[None, :]
    factor = torch.where(after, alpha.unsqueeze(-1), torch.ones_like(alpha.unsqueeze(-1)))
    decay = torch.cumprod(factor, dim=-2).tril()
    retained = torch.cumprod(alpha, dim=-1)
    similarity = key @ key.transpose(-1, -2)
    lower = (beta.unsqueeze(-1)*decay*similarity).tril(diagonal=-1)
    lower = lower+torch.eye(steps, device=key.device, dtype=key.dtype)
    residual = beta.unsqueeze(-1)*(value-retained.unsqueeze(-1)*(key @ initial.transpose(-1, -2)))
    write_value = torch.linalg.solve_triangular(lower, residual, upper=False, unitriangular=True)
    writes = (write_value.unsqueeze(-1)*key.unsqueeze(-2)).flatten(-2)
    states = (decay @ writes).reshape(*writes.shape[:-1], d, d)
    states = states+retained[..., None, None]*initial.unsqueeze(-3)
    return states, states[..., -1, :, :]
