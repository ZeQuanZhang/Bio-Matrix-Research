"""Optional compiled-step adapter for the frozen v0.7 implementation.

The model, optimizer, data, gate schedule and state layout stay in v07. Only
Network.step is compiled. The changing gate blend uses one fixed-address scalar
inside forward, and the runner still sees its original serializable float.
Importing this module is stdlib-only; eager scheduler workers need no adapter
installation. Compilation failures propagate and are never replaced by eager.
"""
import hashlib
import json
from pathlib import Path
import traceback


SETTINGS = dict(backend='inductor', mode='default', fullgraph=True, dynamic=False)
# The complete workflow has legitimate static signatures for B2 training/B1
# evaluation, train/eval and three intervention booleans, fresh versus sliced
# credit-ring strides, and initial/carried/optimizer-detached requires_grad.
# These axes are not all independent. 32 is a bounded per-code cache budget for
# this workflow, not a promise to cover every Cartesian combination. Exceeding
# it still fails under fullgraph=True; the adapter never falls back to eager.
CACHE_POLICY = dict(recompile_limit=32, accumulated_recompile_limit=256)
BIO_VARIANTS = ('message', 'independent', 'full', 'bypass', 'no_local', 'no_slow',
                'no_probability', 'no_dynamic', 'single_timescale', 'no_threshold')
_INSTALLED = None


def adapter_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def runtime_metadata(compile_step, variant, device='cuda', **settings):
    """Stable execution identity; runtime progress fields are added separately."""
    active = bool(compile_step and variant in BIO_VARIANTS)
    selected = SETTINGS | settings
    return dict(schema='biomatrix-v07-acceleration-1', compile_step=bool(compile_step),
                mode='compiled_step' if active else 'eager', variant=variant,
                compiler='torch.compile' if active else None,
                settings=selected if active else None, device=str(device), dtype='float32',
                cache_policy=dict(CACHE_POLICY) if active else None,
                precision_policy='strict_fp32_tf32_disabled' if active else 'core_float32',
                adapter_sha256=adapter_hash())


def write_record(path, value):
    if path is None:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _strict_fp32(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def _ensure_initialized(model):
    import torch
    control = model._v07_acceleration_control
    parameter = model.embedding.weight
    device = parameter.device
    if parameter.dtype != torch.float32 or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Compiled v0.7 adapter requires unchanged FP32 model parameters')
    expected = control['requested_device']
    if expected is not None and (device.type != expected.type or (expected.index is not None and device.index != expected.index)):
        raise ValueError(f'Adapter expected {expected}, model is on {device}')
    if control['initialized']:
        if device != control['device']:
            raise RuntimeError('Moving a compiled model to another device requires a new model/adapter')
        return
    if device.type != 'cuda' and control['settings']['backend'] not in ('eager', 'aot_eager'):
        raise ValueError('Formal compiled-step adapter requires CUDA; eager/aot_eager are CPU testing only')
    if model.cfg.checkpoint_events:
        raise ValueError('Compiled-step adapter currently requires checkpoint_events=false (the formal protocol)')
    _strict_fp32(torch)
    # Compiler cache policy is execution metadata, not a torch.compile kwarg.
    torch._dynamo.config.recompile_limit = CACHE_POLICY['recompile_limit']
    torch._dynamo.config.accumulated_recompile_limit = CACHE_POLICY['accumulated_recompile_limit']
    control['device'] = device
    control['blend'] = torch.ones((), device=device, dtype=torch.float32)
    model._v07_acceleration.update(device=str(device), torch=str(torch.__version__),
                                   status='compiling', initialized=True)
    write_record(control['record_path'], model._v07_acceleration)
    # Keep a bound original method: no new Module/buffer/parameter or state key.
    model.step = torch.compile(control['original_step'], **control['settings'])
    control['initialized'] = True


def _forward(model, original_forward, tokens, state=None, reset_mask=None, diagnostic_mask=None):
    import torch
    control = model._v07_acceleration_control
    previous = model.gate_blend
    try:
        _ensure_initialized(model)
        blend = control['blend']
        # Multiple document-reset segments can share a backward call. Rewriting
        # an equal scalar would still change its version and invalidate saved
        # tensors in the earlier segment's graph.
        value = float(previous.detach()) if isinstance(previous, torch.Tensor) else float(previous)
        if control.get('last_blend') != value:
            with torch.no_grad():
                blend.fill_(value)
            control['last_blend'] = value
        model.gate_blend = blend
        result = original_forward(tokens, state, reset_mask, diagnostic_mask)
        if model._v07_acceleration['status'] != 'active':
            model._v07_acceleration['status'] = 'active'
            write_record(control['record_path'], model._v07_acceleration)
        return result
    except BaseException:
        model._v07_acceleration.update(status='failed', error=traceback.format_exc())
        write_record(control['record_path'], model._v07_acceleration)
        raise
    finally:
        # train_update/metrics/checkpoints must keep the original public scalar.
        model.gate_blend = previous


def attach_compiled_step(model, *, device=None, backend='inductor', mode='default',
                         fullgraph=True, dynamic=False, record_path=None):
    """Attach lazily to an existing Network; return mutable runtime metadata.

    CPU parameter-count construction and .to(cuda) happen before any compilation.
    backend='eager' is an explicit numerical test path, never a fallback backend.
    A second identical attachment is harmless; different settings are rejected.
    """
    import torch
    from v07.model import Network
    if not isinstance(model, Network):
        raise TypeError('Compiled-step adapter only applies to v07.model.Network')
    if model.cfg.variant not in BIO_VARIANTS:
        raise ValueError('Compiled-step adapter requires a graph Network variant')
    settings = dict(backend=backend, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    if not fullgraph or dynamic:
        raise ValueError('Frozen acceleration protocol requires fullgraph=True, dynamic=False')
    requested = torch.device(device) if device is not None else None
    if hasattr(model, '_v07_acceleration_control'):
        control = model._v07_acceleration_control
        if settings != control['settings'] or requested != control['requested_device']:
            raise ValueError('A model cannot mix acceleration settings/devices')
        return model._v07_acceleration
    original_forward = model.forward
    if _INSTALLED is not None and getattr(original_forward, '__func__', None) is Network.forward:
        import types
        original_forward = types.MethodType(_INSTALLED['original_forward'], model)
    model._v07_acceleration = runtime_metadata(True, model.cfg.variant,
                                              str(requested or model.embedding.weight.device), **settings)
    model._v07_acceleration.update(status='pending', initialized=False)
    model._v07_acceleration_control = dict(settings=settings, requested_device=requested,
                                           record_path=record_path, initialized=False,
                                           original_step=model.step, original_forward=original_forward)
    def forward(tokens, state=None, reset_mask=None, diagnostic_mask=None):
        return _forward(model, original_forward, tokens, state, reset_mask, diagnostic_mask)
    model.forward = forward
    return model._v07_acceleration


def install(*, device='cuda', backend='inductor', mode='default', fullgraph=True,
            dynamic=False, record_path=None):
    """Patch Network's class forward before runpy runs v07.run.

    No model is created and no compile constructor runs here. Transformer and
    Stateless classes are untouched. Return the original class method for tests.
    """
    global _INSTALLED
    from v07.model import Network
    settings = dict(device=device, backend=backend, mode=mode, fullgraph=fullgraph,
                    dynamic=dynamic, record_path=record_path)
    if _INSTALLED is not None:
        if _INSTALLED['settings'] != settings:
            raise ValueError('Acceleration install settings already fixed for this worker')
        return _INSTALLED['original_forward']
    original = Network.forward
    def forward(model, tokens, state=None, reset_mask=None, diagnostic_mask=None):
        if not hasattr(model, '_v07_acceleration_control'):
            # Binding the unpatched method avoids recursively capturing this wrapper.
            import types
            model.forward = types.MethodType(original, model)
            attach_compiled_step(model, **settings)
        return model.forward(tokens, state, reset_mask, diagnostic_mask)
    Network.forward = forward
    _INSTALLED = dict(settings=settings, original_forward=original)
    return original


def uninstall():
    """Restore the class hook for isolated local tests; existing models stay attached."""
    global _INSTALLED
    if _INSTALLED is not None:
        from v07.model import Network
        Network.forward = _INSTALLED['original_forward']
        _INSTALLED = None
