import torch
import triton
import triton.language as tl


def block_mask_to_csr_fast(mask):
    B, H, M, _ = mask.shape
    mask = mask.int()

    # flatten BH
    mask2d = mask.view(B * H, M, M)

    row_ptr = torch.zeros((B * H, M + 1), dtype=torch.int32, device=mask.device)

    col_inds = []

    nnz_counter = 0

    for bh in range(B * H):
        nz = torch.nonzero(mask2d[bh], as_tuple=False)

        rows = nz[:, 0]
        cols = nz[:, 1]

        counts = torch.bincount(rows, minlength=M)

        row_ptr[bh, 1:] = torch.cumsum(counts, dim=0)
        row_ptr[bh] += nnz_counter

        col_inds.append(cols)

        nnz_counter += cols.numel()

    col_ind = torch.cat(col_inds)

    row_ptr = row_ptr.view(B, H, M + 1)

    return row_ptr, col_ind


@triton.jit
def block_spmm_kernel(
    A_ptr,
    V_ptr,
    O_ptr,
    row_ptr,
    col_ind,
    stride_abh,
    stride_am,
    stride_an,
    stride_vbh,
    stride_vn,
    stride_vd,
    stride_obh,
    stride_om,
    stride_od,
    B,
    H,
    N,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_BLOCK_PER_PROG: tl.constexpr,
):

    pid_bh = tl.program_id(0)
    pid_row_block_group = tl.program_id(1)

    bh = pid_bh
    row_block_start = pid_row_block_group * ROW_BLOCK_PER_PROG

    d_offsets = tl.arange(0, D)

    for rb in range(ROW_BLOCK_PER_PROG):
        row_block_id = row_block_start + rb
        start = tl.load(row_ptr + row_block_id)
        end = tl.load(row_ptr + row_block_id + 1)
        acc = tl.zeros((BLOCK_SIZE, D), dtype=tl.float32)
        for idx in range(start, end):
            col_block = tl.load(col_ind + idx)
            k_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

            A_row_offsets_2d = (
                tl.arange(0, BLOCK_SIZE)[:, None] + row_block_id * BLOCK_SIZE
            )
            A_col_offsets_2d = k_offsets[None, :]

            a_ptrs = (
                A_ptr
                + bh * stride_abh
                + A_row_offsets_2d * stride_am
                + A_col_offsets_2d * stride_an
            )
            A_block = tl.load(a_ptrs)

            V_row_offsets_2d = k_offsets[:, None]
            V_col_offsets_2d = d_offsets[None, :]
            v_ptrs = (
                V_ptr
                + bh * stride_vbh
                + V_row_offsets_2d * stride_vn
                + V_col_offsets_2d * stride_vd
            )
            V_block = tl.load(v_ptrs)

            acc_block = tl.dot(A_block, V_block)

            acc += acc_block

        row_offsets = (row_block_start + rb) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        o_ptrs = (
            O_ptr
            + bh * stride_obh
            + row_offsets[:, None] * stride_om
            + d_offsets[None, :] * stride_od
        )
        tl.store(o_ptrs, acc)


class BlockSparseSpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, V, row_ptr, col_idx, block_size, block_per_program=4):
        B, H, N, _ = A.shape
        D = V.shape[-1]

        _, _, R = row_ptr.shape

        O = torch.zeros((B, H, N, D), device=A.device, dtype=A.dtype)

        assert (R - 1) % block_per_program == 0

        grid = (B * H, (R - 1) // block_per_program)

        block_spmm_kernel[grid](
            A,
            V,
            O,
            row_ptr,
            col_idx,
            *A.reshape(-1, N, N).stride(),
            *V.reshape(-1, N, D).stride(),
            *O.reshape(-1, N, D).stride(),
            B,
            H,
            N,
            D,
            block_size,
            block_per_program,
        )
        return O

    @staticmethod
    def backward(ctx, grad_output):
        # For research prototype:
        # fallback to dense backward
        raise NotImplementedError


def block_sparse_spmm(A, V, row_ptr, col_idx, block_size=64):
    assert len(A.shape) == 4
    assert len(V.shape) == 4
    # assert len(mask.shape) == 4

    assert A.is_cuda
    assert V.is_cuda

    pad_s = A.shape[-2] % block_size
    if pad_s:
        A = A[:, :, : A.shape[-2] - pad_s, : A.shape[-1] - pad_s]
        V = V[:, :, : V.shape[-2] - pad_s, :]

    d = V.shape[-1]
    pad_d = 1 << (d - 1).bit_length()
    if pad_d != d:
        V = torch.nn.functional.pad(V, (0, pad_d - d))

    # row_ptr, col_idx = block_mask_to_csr_fast(mask)

    return BlockSparseSpMM.apply(A, V, row_ptr, col_idx, block_size)
