import torch

_ATTENTION_PRUNE_MODE = "none"
_TOPK_KEEP_RATIO = 0.8


def normalize_attention_prune_mode(mode: str | None) -> str:
    if mode is None:
        return "none"

    mode = str(mode).lower().replace("_", "-")
    aliases = {
        "": "none",
        "false": "none",
        "no": "none",
        "none": "none",
        "dense": "none",
        "vanilla": "none",
        "sar": "sar",
        "sparse": "sar",
        "sparse-attn": "sar",
        "sparse-attention": "sar",
        "topk": "topk",
        "top-k": "topk",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported attentionPruneMode <{mode}>. "
            "Use one of: none, sar, topk."
        )
    return aliases[mode]


def set_attention_prune_config(
    mode: str | None = "none",
    topk_keep_ratio: float = 0.8,
) -> None:
    global _ATTENTION_PRUNE_MODE, _TOPK_KEEP_RATIO

    mode = normalize_attention_prune_mode(mode)
    topk_keep_ratio = float(topk_keep_ratio)
    if topk_keep_ratio <= 0:
        raise ValueError("topkKeepRatio must be positive.")

    _ATTENTION_PRUNE_MODE = mode
    _TOPK_KEEP_RATIO = min(topk_keep_ratio, 1.0)


def apply_topk_attention_pruning(
    attn_scores: torch.Tensor,
    keep_ratio: float | None = None,
) -> torch.Tensor:
    if _ATTENTION_PRUNE_MODE != "topk":
        return attn_scores

    keep_ratio = _TOPK_KEEP_RATIO if keep_ratio is None else float(keep_ratio)
    if keep_ratio >= 1:
        return attn_scores

    n_token = attn_scores.shape[-1]
    n_keep = max(1, min(n_token, int(keep_ratio * n_token)))
    topk_ids = attn_scores.softmax(dim=-1).topk(n_keep, dim=-1).indices
    topk_mask = torch.full_like(attn_scores, -torch.inf)
    topk_mask.scatter_(-1, topk_ids, 0)
    return attn_scores + topk_mask


def block_sum(x: torch.Tensor, size_tile: int = 64) -> torch.Tensor:
    b, h, s, _ = x.shape
    n_block = s // size_tile

    x_block = x.reshape(b, h, n_block, size_tile, n_block, size_tile).permute(
        0, 1, 2, 4, 3, 5
    )

    return x_block.sum(dim=-1).sum(dim=-1)


def unpack_mask(x: torch.Tensor, size_tile: int = 64) -> torch.Tensor:
    b, h, n, _ = x.shape
    return (
        x.reshape(b, h, n, n, 1, 1)
        .expand(b, h, n, n, size_tile, size_tile)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(b, h, n * size_tile, n * size_tile)
    )


def record_sparse_profile(
    sparse_prof: list[list[dict]] | None,
    idx_timestep: int | None,
    idx_block: int | None,
    name_attn_block: str,
    attn_map: torch.Tensor,
    n_token: int,
    size_tile: int = 64,
) -> None:
    if sparse_prof is None or idx_timestep is None or idx_block is None:
        return

    n_token_pic = attn_map.shape[-1] - (attn_map.shape[-1] % size_tile)
    if n_token_pic <= 0:
        return

    attn_map_block = block_sum(
        attn_map[..., :n_token_pic, :n_token_pic].detach().float().cpu(),
        size_tile=size_tile,
    )
    handle_attn_prof = sparse_prof[idx_timestep][idx_block].get(name_attn_block)
    if handle_attn_prof is None:
        sparse_prof[idx_timestep][idx_block][name_attn_block] = {
            "n_sample": 1,
            "attn_map": attn_map_block,
            "n_token": n_token,
        }
        return

    n_sample = handle_attn_prof["n_sample"]
    saved_prof = handle_attn_prof["attn_map"]
    sparse_prof[idx_timestep][idx_block][name_attn_block] = {
        "n_sample": n_sample + 1,
        "attn_map": (attn_map_block + n_sample * saved_prof) / (n_sample + 1),
        "n_token": n_token,
    }
