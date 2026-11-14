"""Minimal stand-in for kaolin.utils.testing.check_tensor (only symbol FlexiCubes imports).
Kaolin has no wheels for torch 2.7 / sm_120; FlexiCubes only uses it for shape assertions."""


def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
    ok = True
    if shape is not None:
        ok = len(shape) == tensor.dim() and all(s is None or s == t for s, t in zip(shape, tensor.shape))
    if ok and dtype is not None:
        ok = tensor.dtype == dtype
    if ok and device is not None:
        ok = tensor.device == device
    if not ok and throw:
        raise ValueError(f"tensor check failed: {tuple(tensor.shape)} {tensor.dtype} {tensor.device}")
    return ok
