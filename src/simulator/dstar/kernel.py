import torch
import math
try:
    from .arch import *
    from .inst import *
except ImportError:  # Preserve direct execution from the original source tree.
    from arch import *
    from inst import *


def krnl_quantize(
    id_core: int,
    shape: tuple,
    nbit_map: torch.Tensor = None,
    diff: bool = False,
    tile_size: tuple = (64, 128),
):
    m, n = shape
    if nbit_map is not None:
        nbit_map = nbit_map.view(m // tile_size[0], n // tile_size[1]).flatten(0, 1)

    # Tiling parameter
    # tileSize = 64

    for i0 in range((m // tile_size[0]) * (n // tile_size[1]) // N_CORE):
        for i1 in range(N_CORE):
            if i1 == id_core:

                # Load activation
                yield LOAD(
                    DRAM_ADDR_START, UNIBUF0_ADDR_START, tile_size[0] * tile_size[1] * 2
                )

                if diff:
                    # Load cached activation
                    yield LOAD(
                        DRAM_ADDR_START,
                        UNIBUF1_ADDR_START,
                        tile_size[0] * tile_size[1] * 2,
                    )
                    # Diff quantization
                    yield DIFFQUANTIZE(
                        UNIBUF0_ADDR_START,
                        UNIBUF1_ADDR_START,
                        UNIBUF2_ADDR_START,
                        tile_size[1],
                    )
                else:
                    yield QUANTIZE(UNIBUF0_ADDR_START, UNIBUF2_ADDR_START, tile_size[1])
                # Write back
                store_size = (
                    nbit_map[i0 * N_CORE + i1].item() * tile_size[0] // 8
                    if nbit_map is not None
                    else 1 * tile_size[1] * tile_size[0]
                )
                yield STORE(
                    UNIBUF2_ADDR_START,
                    DRAM_ADDR_START,
                    store_size,
                )


def krnl_gemm(
    id_core: int,
    shape: tuple,
    nbit_map: torch.Tensor = None,
    gelu: bool = False,
):
    m, n, k = shape

    # Tiling parameter
    tileSize = 64
    tileM = 4
    tileN = 8

    for i0 in range(m // (tileM * tileSize)):
        for i1 in range(n // (tileN * tileSize)):
            for i2 in range(k // 128):
                yield SYNC()
                for i3 in range(tileM * tileN // N_CORE):
                    for i4 in range(N_CORE):
                        if id_core == i4:
                            mma_latency = (
                                math.ceil(
                                    nbit_map[i0 * tileM + i3 * 4 + id_core // 8][i2] / 8
                                )
                                if nbit_map is not None
                                else 128
                            )
                            yield LOAD(
                                SCRATCHPAD_ADDR_START,
                                UNIBUF0_ADDR_START,
                                64 * mma_latency,
                            )
                            yield LOAD(
                                SCRATCHPAD_ADDR_START,
                                UNIBUF1_ADDR_START,
                                64 * 128,
                            )

                            yield MMA(
                                UNIBUF0_ADDR_START,
                                UNIBUF1_ADDR_START,
                                UNIBUF2_ADDR_START,
                                128,
                                mma_latency,
                                last=(i2 == k // 128 - 1),
                            )

                            if i2 == k // 128 - 1:
                                if gelu:
                                    yield GELU(UNIBUF2_ADDR_START, UNIBUF3_ADDR_START)
                                    yield STORE(
                                        UNIBUF3_ADDR_START,
                                        DRAM_ADDR_START,
                                        tileSize * tileSize * 2,
                                    )
                                else:
                                    yield STORE(
                                        UNIBUF2_ADDR_START,
                                        DRAM_ADDR_START,
                                        tileSize * tileSize * 2,
                                    )


def krnl_quantize_gemm_wrap(
    id_core: int,
    shape: tuple,
    nbit_map: torch.Tensor,
    diff: bool = False,
    gelu: bool = False,
):
    m, n, k = shape

    # yield from krnl_quantize(id_core, (m, k), nbit_map, diff, (64, 128))
    yield from krnl_gemm(id_core, shape, nbit_map, gelu)


def krnl_gemm_dma(
    shape: tuple,
    nbit_map: torch.Tensor = None,
):
    m, n, k = shape

    # Tiling parameter
    tileSize = 64
    tileM = 4
    tileN = 8

    load_stat = 0
    for i0 in range(m // (tileM * tileSize)):
        for i1 in range(n // (tileN * tileSize)):
            for i2 in range(k // 128):
                weight_size = tileSize * 128 * tileN
                act_size = (
                    (
                        nbit_map[
                            i0 * tileM : i0 * tileM + tileM,
                            i2,
                        ]
                        .sum()
                        .item()
                        * 64
                        // 8
                    )
                    if nbit_map is not None
                    else 64 * 128 * tileN
                )
                load_stat += 1
                yield LOAD_DMA(weight_size + act_size)


def krnl_masked_flash_attention(
    id_core: int,
    shape: tuple,
    attn_mask: torch.Tensor,
):
    b, h, n, k = shape

    # ncore_using = N_CORE if 64 * N_CORE <= n else n // 64
    ncore_using = min(N_CORE, attn_mask.shape[2])

    # print(b, h, n, k, ncore_using, attn_mask.shape)
    n_tile = min(16, attn_mask.shape[2])

    for i0 in range(b):
        for i1 in range(h):
            for i2 in range(attn_mask.shape[2] // ncore_using):

                yield SYNC()
                for i3 in range(attn_mask.shape[2] // n_tile):

                    yield SYNC()

                    for i4 in range(ncore_using):
                        if id_core == i4:

                            n_block = (
                                attn_mask[i0][i1][i2 * ncore_using + i4][
                                    i3 * n_tile : i3 * n_tile + n_tile
                                ]
                                .sum()
                                .item()
                            )

                            for i5 in range(0, n_block, 8):
                                for i6 in range(0, min(i5 + 8, n_block) - i5):
                                    # Load Q
                                    yield LOAD(
                                        SCRATCHPAD_ADDR_START,
                                        UNIBUF0_ADDR_START,
                                        64 * k,
                                    )
                                    # Load K
                                    yield LOAD(
                                        SCRATCHPAD_ADDR_START,
                                        UNIBUF1_ADDR_START,
                                        64 * k,
                                    )
                                    yield MMA(
                                        UNIBUF0_ADDR_START,
                                        UNIBUF1_ADDR_START,
                                        UNIBUF2_ADDR_START,
                                        k,
                                        k,
                                        True,
                                    )

                                    yield SOFTMAX(
                                        UNIBUF2_ADDR_START, UNIBUF3_ADDR_START
                                    )
                                    yield QUANTIZE(
                                        UNIBUF3_ADDR_START,
                                        UNIBUF4_ADDR_START + i6 * 8192,
                                    )

                                for i6 in range(0, min(i5 + 8, n_block) - i5):
                                    # Load V
                                    yield LOAD(
                                        SCRATCHPAD_ADDR_START,
                                        UNIBUF12_ADDR_START,
                                        64 * k,
                                    )
                                    last_tile = (
                                        i3 == n // (64 * n_tile) - 1
                                        and i6 == min(i5 + 8, n_block) - i5 - 1
                                    )
                                    yield MMA(
                                        UNIBUF4_ADDR_START + i6 * 8192,
                                        UNIBUF12_ADDR_START,
                                        UNIBUF13_ADDR_START + i6 * 8192,
                                        k,
                                        k,
                                        last_tile,
                                    )
                                    if last_tile:
                                        yield STORE(
                                            UNIBUF13_ADDR_START + i6 * 8192,
                                            DRAM_ADDR_START,
                                            64 * k * 2,
                                        )


def krnl_attention_dma(
    shape: tuple,
    attn_mask: torch.Tensor,
):
    b, h, n, k = shape

    ncore_using = min(N_CORE, attn_mask.shape[2])
    n_tile = min(16, n // 64)

    for i0 in range(b):
        for i1 in range(h):
            for i2 in range(n // (64 * ncore_using)):
                yield LOAD_DMA(64 * ncore_using * k)
                for i3 in range(n // (64 * n_tile)):
                    yield LOAD_DMA(64 * n_tile * k * 2)


def krnl_masked_flash_attention_reuse(
    id_core: int,
    shape: tuple,
    attn_mask: torch.Tensor,
):
    b, h, n, k = shape

    ncore_using = min(N_CORE, attn_mask.shape[2])

    for i0 in range(b):
        for i1 in range(h):
            for i2 in range(n // (64 * ncore_using)):
                yield SYNC()
                for i3 in range(n // 64):
                    yield SYNC()
                    for i4 in range(ncore_using):
                        if i4 == id_core:
                            yield LOAD(
                                SCRATCHPAD_ADDR_START,
                                UNIBUF0_ADDR_START,
                                64 * 64,
                            )
                            yield LOAD(
                                SCRATCHPAD_ADDR_START,
                                UNIBUF1_ADDR_START,
                                64 * k,
                            )
                            yield MMA(
                                UNIBUF0_ADDR_START,
                                UNIBUF1_ADDR_START,
                                UNIBUF2_ADDR_START,
                                k,
                                k,
                                (i3 == n // 64 - 1),
                            )
                            if i3 == n // 64 - 1:
                                yield STORE(
                                    UNIBUF2_ADDR_START, DRAM_ADDR_START, 64 * k * 2
                                )


def krnl_attentio_reuse_dma(
    shape: tuple,
    attn_mask: torch.Tensor,
):
    b, h, n, k = shape

    ncore_using = min(N_CORE, attn_mask.shape[2])

    for i0 in range(b):
        for i1 in range(h):
            for i2 in range(n // (64 * ncore_using)):
                yield LOAD_DMA(64 * N_CORE * 64)
                for i3 in range(n // 64):
                    yield LOAD_DMA(64 * k)


def krnl_attention_wrap(
    id_core: int,
    shape: tuple,
    attn_mask: torch.Tensor,
):
    b, h, n, k = shape

    # Quantize QKV
    # yield from krnl_quantize(id_core, (b * n, h * k), tile_size=(64, k))
    # yield from krnl_quantize(id_core, (b * n, h * k), tile_size=(64, k))
    # yield from krnl_quantize(id_core, (b * n, h * k), tile_size=(64, k))

    # Attention
    yield from krnl_masked_flash_attention(id_core, shape, attn_mask)


def krnl_attentio_reuse_wrap(
    id_core: int,
    shape: tuple,
    attn_mask: torch.Tensor,
):
    b, h, n, k = shape

    # Quantize V
    # yield from krnl_quantize(id_core, (b * n, h * k), tile_size=(64, k))

    # Attention
    yield from krnl_masked_flash_attention_reuse(id_core, shape, attn_mask)


# def krnl_attention_dma_wrap(
#     shape: tuple,
#     reuse: bool = False,
# ):
#     b, h, n, k = shape
#     yield from krnl_gemm_dma((b * n, h * k, h * k))
#     if not reuse:
#         yield from krnl_gemm_dma((b * n, h * k, h * k))
#         yield from krnl_gemm_dma((b * n, h * k, h * k))

#     yield from krnl_attention_dma(shape)
#     yield from krnl_gemm_dma((b * n, h * k, h * k))
