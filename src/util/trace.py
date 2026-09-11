import torch


def dump_trace(trace: dict, path: str, name: str = None):
    from datetime import datetime

    out_path = (
        f"{path}{datetime.now()}.trace"
        if name is None
        else f"{path}{datetime.now()}.trace"
    )
    torch.save(trace, out_path)


def linear(
    shape: tuple, diff: bool = False, gelu: bool = False, nbit_map: torch.Tensor = None
) -> dict:
    return {
        "layer": "linear",
        # [m, n, k]
        "shape": [shape[0], shape[1], shape[2]],
        "diff": diff,
        "gelu": gelu,
        "nbit_map": nbit_map,
    }


def attention(shape: tuple, reuse: bool, attn_mask: torch.Tensor):
    return {
        "layer": "attention",
        # [b, h, n, k]
        "shape": [shape[0], shape[1], shape[2], shape[3]],
        "reuse": reuse,
        "attn_mask": attn_mask,
    }


def matmul_bias_int8(m: int, n: int, k: int, bias: bool) -> dict:
    return {
        "kernel": "matmul_bias_int8",
        "shape": [m, n, k],
        "bias": bias,
    }


def matmul_bias_int8_gelu(m: int, n: int, k: int, bias: bool) -> dict:
    return {
        "kernel": "matmul_bias_int8",
        "shape": [m, n, k],
        "bias": bias,
    }


def matmul_bias_diffq_gelu(
    m: int, n: int, k: int, bias: bool, nbit_map: torch.Tensor
) -> dict:
    return {
        "kernel": "matmul_bias_diffq_gelu",
        "shape": [m, n, k],
        "bias": bias,
        "nbit_map": nbit_map,
    }


def matmul_bias_diffq(
    m: int, n: int, k: int, bias: bool, nbit_map: torch.Tensor
) -> dict:
    return {
        "kernel": "matmul_bias_int8",
        "shape": [m, n, k],
        "bias": bias,
        "nbit_map": nbit_map,
    }


def flash_attn(b: int, h: int, m: int, n: int, k: int):
    pass


def masked_flash_attn(b: int, h: int, m: int, n: int, k: int, attn_mask: torch.Tensor):
    pass


def masked_flash_attn_reuse(
    b: int, h: int, m: int, n: int, k: int, attn_mask: torch.Tensor
):
    pass
