import torch
import torch.nn.functional as F
from diff.quantize import (
    quantize_group,
    quantizeAdapt,
)
from util.trace import *

stat = []

_DIFF_QUANT_MODE = "adapt"
_DIFF_QUANT_GROUP_SIZE = (1, 128)


def normalize_diff_quant_mode(mode: str | None) -> str:
    if mode is None:
        return "adapt"

    mode = str(mode).lower().replace("_", "-")
    aliases = {
        "": "adapt",
        "adapt": "adapt",
        "adaptive": "adapt",
        "diffq": "adapt",
        "ours": "adapt",
        "group": "group",
        "quantize-group": "group",
        "quantize-grouped": "group",
        "ditto": "group",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported diffQuantMode <{mode}>. Use one of: adapt, group, ditto."
        )
    return aliases[mode]


def set_diff_quant_config(
    mode: str | None = "adapt",
    group_size: tuple[int, int] | list[int] = (1, 128),
) -> None:
    global _DIFF_QUANT_MODE, _DIFF_QUANT_GROUP_SIZE

    mode = normalize_diff_quant_mode(mode)
    if len(group_size) != 2:
        raise ValueError("diffQuantGroupSize must contain exactly two integers.")

    group_size = (int(group_size[0]), int(group_size[1]))
    if group_size[0] <= 0 or group_size[1] <= 0:
        raise ValueError("diffQuantGroupSize values must be positive.")

    _DIFF_QUANT_MODE = mode
    _DIFF_QUANT_GROUP_SIZE = group_size


def pad_tensor(x, htile, wtile, value):
    from torch.nn import functional as F

    h = x.shape[-2]
    w = x.shape[-1]

    h_padding = (htile - h % htile) % htile
    w_padding = (wtile - w % wtile) % wtile

    x_paded = F.pad(x, (0, w_padding, 0, h_padding), value=value).to(torch.float)

    return x_paded


def diff_linear(
    input: torch.Tensor,
    cache_input: torch.Tensor,
    cache_output: torch.Tensor,
    weight: torch.Tensor,
    base: float = 1e-3,
    debug: bool = False,
    bitPair: tuple = None,
    scale: float = None,
    enableTraceGen: bool = False,
):
    # d_x = input - cache_input
    d_x = input - cache_input

    if _DIFF_QUANT_MODE == "group":
        d_x_q, nbit = quantize_group(
            d_x,
            _DIFF_QUANT_GROUP_SIZE,
            return_nbit=True,
            verbose=False,
        )
    else:
        d_x_q, nbit = quantizeAdapt(d_x, (64, 1), 0, scale)
        # d_x_q, nbit = quantizeAdaptDyn(d_x, (64, 1), 0, scale)
        # d_x_q, nbit = quantizePerTensorOutlierAware(d_x)

    stat.append(nbit.float().mean())
    # print(sum(stat) / len(stat))
    # print(nbit.float().mean())

    d_output = torch.matmul(d_x_q, weight)

    return d_output + cache_output, nbit


# def diff_ffn(
#     input: torch.Tensor,
#     cache: dict,
#     symbol: str,
#     force_fill: bool,
#     ffn_layer: torch.nn.Module,
#     debug: bool = False,
#     scale: float = None,
#     enableTraceGen: bool = False,
# ):
#     if force_fill:

#         ffn1_output = ffn_layer.net[0].proj(input)
#         cache[f"{symbol}_ffn1_input"] = input
#         cache[f"{symbol}_ffn1_output"] = ffn1_output

#         act_ffn1_output = F.gelu(ffn1_output, approximate="tanh")

#         if enableTraceGen:
#             dump_trace(
#                 matmul_bias_int8_gelu(
#                     input.shape[0] * input.shape[1],
#                     ffn_layer.net[0].proj.weight.shape[0],
#                     input.shape[2],
#                     True,
#                 )
#             )

#         ffn2_output = ffn_layer.net[2](act_ffn1_output)

#         if enableTraceGen:
#             dump_trace(
#                 matmul_bias_int8(
#                     act_ffn1_output.shape[0] * act_ffn1_output.shape[1],
#                     ffn_layer.net[2].weight.shape[0],
#                     act_ffn1_output.shape[2],
#                     True,
#                 )
#             )

#         # cache[f"{symbol}_ffn2_input"] = act_ffn1_output
#         cache[f"{symbol}_ffn2_output"] = ffn2_output
#     else:
#         ffn1_output, nbit1 = diff_linear(
#             input,
#             cache[f"{symbol}_ffn1_input"],
#             cache[f"{symbol}_ffn1_output"],
#             ffn_layer.net[0].proj.weight.T,
#             debug=debug,
#             scale=scale,
#         )

#         act_ffn1_output = F.gelu(ffn1_output, approximate="tanh")

#         if enableTraceGen:
#             dump_trace(
#                 matmul_bias_diffq_gelu(
#                     input.shape[0] * input.shape[1],
#                     ffn_layer.net[0].proj.weight.shape[0],
#                     input.shape[2],
#                     True,
#                     nbit1,
#                 )
#             )

#         ffn2_output, nbit2 = diff_linear(
#             act_ffn1_output,
#             # cache[f"{symbol}_ffn2_input"],
#             F.gelu(cache[f"{symbol}_ffn1_output"], approximate="tanh"),
#             cache[f"{symbol}_ffn2_output"],
#             ffn_layer.net[2].weight.T,
#             debug=debug,
#             scale=scale,
#         )

#         if enableTraceGen:
#             dump_trace(
#                 matmul_bias_diffq_gelu(
#                     act_ffn1_output.shape[0] * act_ffn1_output.shape[1],
#                     ffn_layer.net[2].weight.shape[0],
#                     act_ffn1_output.shape[2],
#                     True,
#                     nbit2,
#                 )
#             )

#         cache[f"{symbol}_ffn1_input"] = input
#         cache[f"{symbol}_ffn1_output"] = ffn1_output

#         # cache[f"{symbol}_ffn2_input"] = act_ffn1_output
#         cache[f"{symbol}_ffn2_output"] = ffn2_output

#     return ffn2_output


def diff_ffn(
    input: torch.Tensor,
    cache: dict,
    symbol: str,
    force_fill: bool,
    ffn_layer: torch.nn.Module = None,
    weight0: torch.Tensor = None,
    bias0: torch.Tensor = None,
    weight1: torch.Tensor = None,
    bias1: torch.Tensor = None,
    debug: bool = False,
    scale: float = None,
    enableTraceGen: bool = False,
    tracePath: str = None,
    disableReorder: bool = False,
):
    def _nbit_trace_map(nbit: torch.Tensor):
        nbit_block = nbit.view(*nbit.shape[:-1], nbit.shape[-1] // 128, 128)
        nbit_sum = nbit_block.sum(dim=-1)
        if disableReorder:
            shift_overhead = torch.diff(nbit_block, dim=-1).abs().sum(dim=-1) *2
            nbit_sum = nbit_sum + shift_overhead
        return nbit_sum.flatten(0, 1).cpu()

    if ffn_layer is not None:
        weight0 = ffn_layer.net[0].proj.weight
        bias0 = ffn_layer.net[0].proj.bias
        weight1 = ffn_layer.net[2].weight
        bias1 = ffn_layer.net[2].bias

    if force_fill:

        ffn1_output = torch.matmul(input, weight0.T) + bias0
        cache[f"{symbol}_ffn1_input"] = input
        cache[f"{symbol}_ffn1_output"] = ffn1_output

        act_ffn1_output = F.gelu(ffn1_output, approximate="tanh")

        if enableTraceGen:
            dump_trace(
                linear(
                    (
                        input.shape[0] * input.shape[1],
                        weight0.shape[0],
                        input.shape[2],
                    ),
                    False,
                    True,
                    None,
                ),
                tracePath,
            )

        ffn2_output = torch.matmul(act_ffn1_output, weight1.T) + bias1

        if enableTraceGen:
            dump_trace(
                linear(
                    (
                        act_ffn1_output.shape[0] * act_ffn1_output.shape[1],
                        weight1.shape[0],
                        act_ffn1_output.shape[2],
                    ),
                    False,
                    False,
                    None,
                ),
                tracePath,
            )

        # cache[f"{symbol}_ffn2_input"] = act_ffn1_output
        cache[f"{symbol}_ffn2_output"] = ffn2_output
    else:
        ffn1_output, nbit1 = diff_linear(
            input,
            cache[f"{symbol}_ffn1_input"],
            cache[f"{symbol}_ffn1_output"],
            weight0.T,
            debug=debug,
            scale=scale,
        )

        act_ffn1_output = F.gelu(ffn1_output, approximate="tanh")

        if enableTraceGen:
            dump_trace(
                linear(
                    (
                        input.shape[0] * input.shape[1],
                        weight0.shape[0],
                        input.shape[2],
                    ),
                    True,
                    True,
                    _nbit_trace_map(nbit1),
                ),
                tracePath,
            )

        ffn2_output, nbit2 = diff_linear(
            act_ffn1_output,
            # cache[f"{symbol}_ffn2_input"],
            F.gelu(cache[f"{symbol}_ffn1_output"], approximate="tanh"),
            cache[f"{symbol}_ffn2_output"],
            weight1.T,
            debug=debug,
            scale=scale,
        )

        if enableTraceGen:
            dump_trace(
                linear(
                    (
                        act_ffn1_output.shape[0] * act_ffn1_output.shape[1],
                        weight1.shape[0],
                        act_ffn1_output.shape[2],
                    ),
                    True,
                    False,
                    _nbit_trace_map(nbit2),
                ),
                tracePath,
            )

        cache[f"{symbol}_ffn1_input"] = input
        cache[f"{symbol}_ffn1_output"] = ffn1_output

        # cache[f"{symbol}_ffn2_input"] = act_ffn1_output
        cache[f"{symbol}_ffn2_output"] = ffn2_output

    return ffn2_output


def sparse_reuse_ffn(
    input: torch.Tensor,
    cache: dict,
    symbol: str,
    force_fill: bool,
    ffn_layer: torch.nn.Module = None,
    weight0: torch.Tensor = None,
    bias0: torch.Tensor = None,
    weight1: torch.Tensor = None,
    bias1: torch.Tensor = None,
    threshold: float = 0.01,
    enableTraceGen: bool = False,
    tracePath: str = None,
):
    if ffn_layer is not None:
        weight0 = ffn_layer.net[0].proj.weight
        bias0 = ffn_layer.net[0].proj.bias
        weight1 = ffn_layer.net[2].weight
        bias1 = ffn_layer.net[2].bias

    ffn1_output = torch.matmul(input, weight0.T)
    if bias0 is not None:
        ffn1_output = ffn1_output + bias0
    act_ffn1_output = F.gelu(ffn1_output, approximate="tanh")

    cache_key = f"{symbol}_act_ffn1_output"
    mask_key = f"{symbol}_reuse_mask"
    if force_fill or cache_key not in cache or mask_key not in cache:
        cache[cache_key] = act_ffn1_output.detach()
        cache[mask_key] = act_ffn1_output.abs() < threshold
    else:
        cached_act = cache[cache_key]
        reuse_mask = cache[mask_key]
        act_ffn1_output = torch.where(reuse_mask, cached_act, act_ffn1_output)

    if enableTraceGen:
        dump_trace(
            linear(
                (
                    input.shape[0] * input.shape[1],
                    weight0.shape[0],
                    input.shape[2],
                ),
                False,
                True,
                None,
            ),
            tracePath,
        )

    ffn2_output = torch.matmul(act_ffn1_output, weight1.T)
    if bias1 is not None:
        ffn2_output = ffn2_output + bias1

    if enableTraceGen:
        dump_trace(
            linear(
                (
                    act_ffn1_output.shape[0] * act_ffn1_output.shape[1],
                    weight1.shape[0],
                    act_ffn1_output.shape[2],
                ),
                False,
                False,
                None,
            ),
            tracePath,
        )

    cache[cache_key] = act_ffn1_output.detach()
    cache[f"{symbol}_ffn2_output"] = ffn2_output.detach()
    return ffn2_output


def diff_proj(
    input: torch.Tensor,
    cache: dict,
    symbol: str,
    force_fill: bool,
    proj_layer: torch.nn.Module,
    debug: bool = False,
):
    if force_fill:
        proj_out = proj_layer(input)
        cache[f"{symbol}_input"] = input
        cache[f"{symbol}_output"] = proj_out
    else:
        proj_out = diff_linear(
            input,
            cache[f"{symbol}_input"],
            cache[f"{symbol}_output"],
            proj_layer.weight.T,
            debug=debug,
        )
        cache[f"{symbol}_input"] = input
        cache[f"{symbol}_output"] = proj_out

    return proj_out
