import torch

stat = []


def getOutlierMaskBy3Sigma(x: torch.Tensor, dim: int = -1):
    """
    Get outlier mask in dimension dim using 3sigma principle
    """
    mu = x.mean(dim=dim)
    sigma = x.std(dim=dim)
    upperBound = (mu + 3 * sigma).unsqueeze(-1).expand(x.shape)
    lowerBound = (mu - 3 * sigma).unsqueeze(-1).expand(x.shape)
    mask = (x < lowerBound) | (x > upperBound)
    return mask


def getOutlierMaskByThreshold(x: torch.Tensor, threshold: float, dim: int = -1):
    """
    Get outlier mask in dimension dim using threshold
    """

    return x.abs() > threshold


def getOutlierMaskByTopk(x: torch.Tensor, k: int, dim: int = -1):
    """
    Get outlier mask in dimension dim using topk selection
    """

    _, idx = torch.topk(x.abs(), k, dim=dim)

    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(dim, idx, True)

    return mask


def inferBitByErr(x: torch.Tensor, dim: int = -1):
    for n in range(2, 8):
        scale = (2 ** (n - 1) - 1) / x.abs().max(dim=-1).values.unsqueeze(-1).expand(
            x.shape
        )
        err = torch.abs(torch.round(x * scale) / (scale + 1e-8) - x)


def getRangeMask(x: torch.Tensor, dim: int = -1):
    mu = x.mean(dim=dim).unsqueeze(-1).expand(x.shape)
    sigma = x.std(dim=dim).unsqueeze(-1).expand(x.shape)

    mask2b = (x >= mu - sigma) & (x <= mu + sigma)
    # mask3b = (x > mu - 2 * sigma) & (x < mu + 2 * sigma)
    mask4b = (x >= mu - 3 * sigma) & (x <= mu + 3 * sigma)
    return mask2b, mask4b, ~mask4b


def quantizeMSQ(x: torch.Tensor, sizeGroup: tuple):
    hGroup, wGroup = sizeGroup
    h, w = x.shape[-2:]

    assert h % hGroup == 0 and w % wGroup == 0

    xTile = (
        x.reshape(*x.shape[:-2], h // hGroup, hGroup, w // wGroup, wGroup).transpose(
            -3, -2
        )
    ).float()

    maskOutlier = getOutlierMaskBy3Sigma(xTile.flatten(-2, -1)).reshape(xTile.shape)
    # maskOutlier = getOutlierMaskByThreshold(xTile.flatten(-2, -1), 0.05).reshape(
    #     xTile.shape
    # )
    maskInlier = ~maskOutlier

    stat.append(maskOutlier.sum() / maskOutlier.numel())
    print(sum(stat) / len(stat))

    scaleInlier = (
        (
            7
            / torch.masked_fill(
                xTile.flatten(-2, -1).abs(), maskOutlier.flatten(-2, -1), -torch.inf
            )
            .max(dim=-1)
            .values
        )
        .clamp(min=1, max=1e7)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )
    xQuantizedInlier = torch.round(xTile * scaleInlier) / (scaleInlier + 1e-8)

    scaleOutlier = (
        (
            127
            / torch.masked_fill(xTile.abs(), maskInlier, -torch.inf).max(dim=-2).values
        )
        .clamp(min=1, max=1e7)
        .unsqueeze(-2)
        .expand(xTile.shape)
    )
    xQuantizedOutlier = torch.round(xTile * scaleOutlier) / (scaleOutlier + 1e-8)

    xTileQuantize = xTile[...]
    xTileQuantize[maskInlier] = xQuantizedInlier[maskInlier]
    xTileQuantize[maskOutlier] = xQuantizedOutlier[maskOutlier]

    xTileQuantize = (
        xTileQuantize.view(-1, h // hGroup, w // wGroup, hGroup, wGroup)
        .transpose(-3, -2)
        .reshape(-1, x.shape[-2], x.shape[-1])
    ).to(x.dtype)

    return xTileQuantize


def quantizeSDQ(x: torch.Tensor, sizeGroup: tuple, scale: float):
    hGroup, wGroup = sizeGroup
    h, w = x.shape[-2:]

    assert h % hGroup == 0 and w % wGroup == 0

    xTile = (
        x.reshape(*x.shape[:-2], h // hGroup, hGroup, w // wGroup, wGroup)
        .transpose(-3, -2)
        .flatten(-2, -1)
    ).float()

    muTile = xTile.mean(dim=-1).unsqueeze(-1).expand(xTile.shape)
    xTile = xTile - muTile

    maskOutlier = getOutlierMaskBy3Sigma(xTile)
    # maskOutlier = getOutlierMaskByTopk(xTile, 4)
    # maskOutlier = getOutlierMaskByThreshold(xTile, 0.03)
    maskInlier = ~maskOutlier

    # stat.append(5.0)
    # print(sum(stat) / len(stat))

    scaleInlier = (
        (
            15.0
            / torch.masked_fill(xTile.abs(), maskOutlier, -torch.inf).max(dim=-1).values
        )
        .clamp(min=1, max=1e7)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )

    xQuantizedInlier = torch.round(xTile * scaleInlier) / (scaleInlier + 1e-8)

    scaleOutlier = (
        (
            127
            / torch.masked_fill(xTile.abs(), maskInlier, -torch.inf).max(dim=-1).values
        )
        .clamp(min=1, max=1e7)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )
    xQuantizedOutlier = torch.round(xTile * scaleOutlier) / (scaleOutlier + 1e-8)

    xTileQuantize = xTile[...]
    xTileQuantize[maskInlier] = xQuantizedInlier[maskInlier]
    xTileQuantize[maskOutlier] = xQuantizedOutlier[maskOutlier]

    xTileQuantize = xTileQuantize + muTile

    xTileQuantize = (
        xTileQuantize.view(-1, h // hGroup, w // wGroup, hGroup, wGroup)
        .transpose(-3, -2)
        .reshape(-1, x.shape[-2], x.shape[-1])
    ).to(x.dtype)

    print((xTileQuantize - x).abs().mean())

    return xTileQuantize


def quantizeAdaptiveSDQ(x: torch.Tensor, sizeGroup: tuple, k: int, scale: float = None):
    hGroup, wGroup = sizeGroup
    h, w = x.shape[-2:]

    assert k <= hGroup * wGroup
    assert h % hGroup == 0 and w % wGroup == 0

    xTile = (
        x.reshape(*x.shape[:-2], h // hGroup, hGroup, w // wGroup, wGroup)
        .transpose(-3, -2)
        .flatten(-2, -1)
    ).float()

    # muTile = xTile.mean(dim=-1).unsqueeze(-1).expand(xTile.shape)
    # xTile = xTile - muTile

    maskOutlier = getOutlierMaskByTopk(xTile, k)
    # maskOutlier = getOutlierMaskBy3Sigma(xTile)
    maskInlier = ~maskOutlier

    # from matplotlib import pyplot as plt
    # import seaborn as sns
    # import datetime

    # for i in maskOutlier.reshape(
    #     *x.shape[:-2], h // hGroup, w // wGroup, hGroup, wGroup
    # ).flatten(0, -3):
    #     sns.heatmap(i.float().cpu()).figure.savefig(
    #         f"/home/czhang/adapt/pic/{datetime.datetime.now()}.png"
    #     )
    #     plt.clf()

    # boundaries = torch.tensor(
    #     [0.015, 0.040, 0.100, 0.200, 0.420, 0.840], device=x.device
    # )
    # boundaries = torch.tensor([0.01, 0.03, 0.06, 0.15, 0.25, 0.55], device=x.device)

    # boundaries = torch.tensor(
    #     [0.025, 0.075, 0.175, 0.375, 0.775, 1.575], device=x.device
    # )

    # 0.001
    # boundaries = torch.tensor(
    #     [0.004, 0.012, 0.025, 0.060, 0.120, 0.400], device=x.device
    # )

    # 0.005
    # boundaries = torch.tensor(
    #     [0.020, 0.060, 0.130, 0.300, 0.600, 1.200], device=x.device
    # )
    # boundaries = torch.tensor(
    #     [0.004, 0.012, 0.028, 0.060, 0.124, 0.252], device=x.device
    # )
    # mappedBit = torch.tensor(
    #     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], device=x.device
    # )
    mappedBit = torch.tensor([2, 4, 6, 8], device=x.device)
    # mappedScale = torch.tensor(
    #     [250.0, 250.0, 250.0, 250.0, 250.0, 250.0, 250.0], device=x.device
    # )
    mappedRange = 2 ** (mappedBit - 1) - 1
    boundaries = mappedRange[:-1] / scale

    def inferBit(x: torch.Tensor):
        bins = torch.bucketize(x, boundaries)
        nBit = mappedBit[bins]
        scaleRange = 2 ** (nBit - 1) - 1
        return nBit, scaleRange

    # def inferScale(x: torch.Tensor):
    #     bins = torch.bucketize(x, boundaries)
    #     nBit = mappedBit[bins]
    #     scale = mappedScale[bins]
    #     return nBit, scale

    maskedMaxAbsInlier = (
        torch.masked_fill(xTile.abs(), maskOutlier, -torch.inf).max(dim=-1).values
    )
    nBitInlier, scaleRange = inferBit(maskedMaxAbsInlier)
    # nBitInlier, scaleInlier = inferScale(maskedMaxAbsInlier)
    # scaleInlier = scaleInlier.unsqueeze(-1).expand(xTile.shape)

    # nBitInlier, scaleRange = inferBit(
    #     torch.masked_fill(xTile, maskOutlier, 0).std(dim=-1) * 3
    # )

    # scaleInlier = (
    #     (scaleRange / maskedMaxAbsInlier)
    #     .clamp(min=1, max=1e7)
    #     .unsqueeze(-1)
    #     .expand(xTile.shape)
    # )
    scaleInlier = scale

    # scaleInlier = torch.where(xTile.abs() < 0.0078, 128, 64)
    # scaleInlier = torch.full_like(xTile, 160.0)
    # print(
    #     scaleInlier.masked_scatter(maskInlier, torch.full_like(scaleInlier, 0))
    #     .max(dim=-1)
    #     .values,
    #     nBitInlier,
    # )
    xQuantizedInlier = torch.round(xTile * scaleInlier) / (scaleInlier + 1e-8)

    maskedMaxAbsOutlier = (
        torch.masked_fill(xTile.abs(), maskInlier, -torch.inf).max(dim=-1).values
    )
    nBitOutlier, scaleRange = inferBit(maskedMaxAbsOutlier)
    # nBitOutlier, scaleOutlier = inferScale(maskedMaxAbsOutlier)

    # scaleOutlier = (
    #     (scaleRange / maskedMaxAbsOutlier)
    #     .clamp(min=1, max=1e7)
    #     .unsqueeze(-1)
    #     .expand(xTile.shape)
    # )
    scaleOutlier = scale
    # scaleOutlier = scaleOutlier.unsqueeze(-1).expand(xTile.shape)

    xQuantizedOutlier = torch.round(xTile * scaleOutlier) / (scaleOutlier + 1e-8)

    # print(nBitInlier.float().mean(), nBitOutlier.float().mean(), x.abs().mean())
    # print(xTile.mean(dim=-1))
    # maskOutlierTile = maskOutlier.sum(dim=-1) > 0
    # print(
    #     "inlier",
    #     nBitInlier,
    #     "outlier",
    #     nBitOutlier.masked_scatter(maskOutlierTile, torch.full_like(nBitOutlier, 0)),
    #     "tile",
    #     xTile,
    # )

    # Statistics
    stat.append(
        (
            (nBitInlier * (hGroup * wGroup - k)).sum()
            + (nBitOutlier * k).sum()
            + nBitOutlier.numel() * k * math.ceil(math.log2(float(hGroup * wGroup)))
        )
        / x.numel()
    )
    print(sum(stat) / len(stat))

    xTileQuantize = xTile[...]
    xTileQuantize[maskInlier] = xQuantizedInlier[maskInlier]
    xTileQuantize[maskOutlier] = xQuantizedOutlier[maskOutlier]

    # xTileQuantize = xTileQuantize + muTile

    xTileQuantize = (
        xTileQuantize.reshape(*x.shape[:-2], h // hGroup, w // wGroup, hGroup, wGroup)
        .transpose(-3, -2)
        .reshape(*x.shape[:-2], x.shape[-2], x.shape[-1])
    ).to(x.dtype)

    # print(((xTileQuantize - x).abs()).mean())

    return xTileQuantize


def quantizeAdaptivePerTensorQuantize(x: torch.Tensor, scale: float = None):
    xFloat = x.float()

    mappedBit = torch.tensor([2, 3, 4, 5, 6, 7, 8], device=x.device)
    mappedRange = 2 ** (mappedBit - 1) - 1
    boundaries = mappedRange[:-1] / scale

    def inferBit(x: torch.Tensor):
        bins = torch.bucketize(x, boundaries)
        nBit = mappedBit[bins]
        scaleRange = 2 ** (nBit - 1) - 1
        return nBit, scaleRange

    nBit, scaleRange = inferBit(xFloat.flatten(-2, -1).abs().max(dim=-1).values)

    print(nBit.float().mean())

    xQuantized = torch.round(xFloat * scale) / (scale + 1e-8)

    return xQuantized.to(x.dtype)


def quantizeDstar(x: torch.Tensor, sizeGroup: tuple):
    hGroup, wGroup = sizeGroup
    h, w = x.shape[-2:]

    assert h % hGroup == 0 and w % wGroup == 0

    xTile = (
        x.reshape(*x.shape[:-2], h // hGroup, hGroup, w // wGroup, wGroup)
        .transpose(-3, -2)
        .flatten(-2, -1)
    ).float()

    maskOutlier = (
        (xTile.abs().max(dim=-1).values > 0.03).unsqueeze(-1).expand(xTile.shape)
    )
    maskInlier = ~maskOutlier

    stat.append(maskOutlier.sum() / maskOutlier.numel())
    print(sum(stat) / len(stat))

    rangeTile = torch.where(maskOutlier, 127, 3)

    scale = (
        rangeTile / xTile.abs().max(dim=-1).values.unsqueeze(-1).expand(xTile.shape)
    ).clamp(min=1, max=1e7)
    xQuantized = torch.round(xTile * scale) / (scale + 1e-8)

    xTileQuantize = (
        xQuantized.view(-1, h // hGroup, w // wGroup, hGroup, wGroup)
        .transpose(-3, -2)
        .reshape(-1, x.shape[-2], x.shape[-1])
    ).to(x.dtype)

    return xTileQuantize


def quantizePerTensorOutlierAware(x: torch.Tensor):
    xTile = x.float()

    maskOutlier = getOutlierMaskByThreshold(xTile, 0.03)
    # maskOutlier = getOutlierMaskBy3Sigma(xTile)
    maskInlier = ~maskOutlier

    stat.append(maskOutlier.sum() / maskOutlier.numel())
    print(sum(stat) / len(stat))

    scaleInlier = (
        (7 / torch.masked_fill(xTile.abs(), maskOutlier, -torch.inf).max(dim=-1).values)
        .clamp(min=1, max=1e7)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )

    xQuantizedInlier = torch.round(xTile * scaleInlier) / (scaleInlier + 1e-8)

    scaleOutlier = (
        (
            127
            / torch.masked_fill(xTile.abs(), maskInlier, -torch.inf).max(dim=-1).values
        )
        .clamp(min=1, max=1e7)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )
    xQuantizedOutlier = torch.round(xTile * scaleOutlier) / (scaleOutlier + 1e-8)

    xTileQuantize = xTile[...]
    xTileQuantize[maskInlier] = xQuantizedInlier[maskInlier]
    xTileQuantize[maskOutlier] = xQuantizedOutlier[maskOutlier]

    xTileQuantize = xTileQuantize.to(x.dtype)

    return xTileQuantize, maskInlier


def quantizePerTensor(x: torch.Tensor):
    xTile = x.float()

    scale = (
        (127 / xTile.abs().flatten(-2, -1).max(dim=-1).values)
        .clamp(min=1, max=1e7)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )

    xQuantized = torch.round(xTile * scale) / (scale + 1e-8)

    xTileQuantize = xQuantized.to(x.dtype)

    return xTileQuantize


# def quantize_group(x: torch.Tensor, sizeGroup: tuple):
#     hGroup, wGroup = sizeGroup
#     h, w = x.shape[-2:]

#     assert h % hGroup == 0 and w % wGroup == 0

#     xTile = (
#         x.reshape(-1, h // hGroup, hGroup, w // wGroup, wGroup)
#         .transpose(-3, -2)
#         .flatten(-2, -1)
#     ).float()

#     mask2b, mask4b, mask8b = getRangeMask(xTile)

#     scale2b = (
#         (7 / torch.masked_fill(xTile.abs(), (~mask2b), -torch.inf).max(dim=-1).values)
#         .unsqueeze(-1)
#         .expand(xTile.shape)
#     )
#     scale4b = (
#         (7 / torch.masked_fill(xTile.abs(), (~mask4b), -torch.inf).max(dim=-1).values)
#         .unsqueeze(-1)
#         .expand(xTile.shape)
#     )
#     scale8b = (
#         (127 / torch.masked_fill(xTile.abs(), (~mask8b), -torch.inf).max(dim=-1).values)
#         .unsqueeze(-1)
#         .expand(xTile.shape)
#     )

#     xQuantizedInt2 = (xTile * scale2b).to(torch.int).to(torch.float) / (scale2b + 1e-8)
#     xQuantizedInt4 = (xTile * scale4b).to(torch.int).to(torch.float) / (scale4b + 1e-8)
#     xQuantizedInt8 = (xTile * scale8b).to(torch.int).to(torch.float) / (scale8b + 1e-8)

#     xTileQuantize = xTile[...]
#     # xTileQuantize[mask2b] = xQuantizedInt2[mask2b]
#     xTileQuantize[mask4b] = xQuantizedInt4[mask4b]
#     xTileQuantize[mask8b] = xQuantizedInt8[mask8b]

#     xTileQuantize = (
#         xTileQuantize.view(-1, h // hGroup, w // wGroup, hGroup, wGroup)
#         .transpose(-3, -2)
#         .reshape(-1, x.shape[-2], x.shape[-1])
#     ).to(x.dtype)

#     return xTileQuantize


def quantize_group(
    x: torch.Tensor,
    sizeGroup: tuple,
    return_nbit: bool = False,
    verbose: bool = True,
):
    hGroup, wGroup = sizeGroup
    h, w = x.shape[-2:]

    assert h % hGroup == 0 and w % wGroup == 0

    leading_shape = x.shape[:-2]
    xTile = (
        x.reshape(*leading_shape, h // hGroup, hGroup, w // wGroup, wGroup)
        .transpose(-3, -2)
        .flatten(-2, -1)
    ).float()

    maskOutlier = getOutlierMaskBy3Sigma(xTile)
    maskInlier = torch.logical_not(maskOutlier)

    if verbose:
        stat.append(maskOutlier.sum() / maskOutlier.numel())
        print(sum(stat) / len(stat))

    scaleInlier = (
        (
            7
            / torch.masked_fill(xTile.abs(), maskOutlier, -torch.inf)
            .max(dim=-1)
            .values.clamp(min=1e-8)
        )
        .clamp(min=1, max=1e7)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )
    scaleOutlier = (
        (
            127
            / torch.masked_fill(xTile.abs(), maskInlier, -torch.inf).max(dim=-1).values
        )
        .clamp(min=1, max=1e7)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )

    xQuantizedInt8 = (xTile * scaleOutlier).to(torch.int).to(torch.float) / (
        scaleOutlier + 1e-8
    )
    xQuantizedInt4 = (xTile * scaleInlier).to(torch.int).to(torch.float) / (
        scaleInlier + 1e-8
    )

    xTileQuantize = xTile.clone()
    xTileQuantize[maskInlier] = xQuantizedInt4[maskInlier]
    xTileQuantize[maskOutlier] = xQuantizedInt8[maskOutlier]

    xTileQuantize = (
        xTileQuantize.reshape(*leading_shape, h // hGroup, w // wGroup, hGroup, wGroup)
        .transpose(-3, -2)
        .reshape(*leading_shape, x.shape[-2], x.shape[-1])
    ).to(x.dtype)

    if not return_nbit:
        return xTileQuantize

    nBit = torch.where(
        maskInlier,
        torch.full_like(xTile, 4, dtype=torch.int16),
        torch.full_like(xTile, 8, dtype=torch.int16),
    )
    nBit = (
        nBit.reshape(*leading_shape, h // hGroup, w // wGroup, hGroup, wGroup)
        .transpose(-3, -2)
        .reshape(*leading_shape, x.shape[-2], x.shape[-1])
    )

    return xTileQuantize, nBit


def quantizeAdaptiveMSQ(x: torch.Tensor, sizeGroup: tuple, k: int):
    hGroup, wGroup = sizeGroup
    h, w = x.shape[-2:]

    assert k <= hGroup * wGroup
    assert h % hGroup == 0 and w % wGroup == 0

    xTile = (
        x.reshape(*x.shape[:-2], h // hGroup, hGroup, w // wGroup, wGroup).transpose(
            -3, -2
        )
    ).float()

    muTile = (
        xTile.flatten(-2, -1)
        .mean(dim=-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )

    xTile = xTile - muTile

    maskOutlier = (
        getOutlierMaskByTopk(xTile.abs().max(dim=-2).values, k)
        .unsqueeze(-2)
        .expand(xTile.shape)
    )
    maskInlier = ~maskOutlier

    boundaries = torch.tensor(
        [0.025, 0.075, 0.175, 0.375, 0.775, 1.575], device=x.device
    )
    mappedBit = torch.tensor([2, 3, 4, 5, 6, 7, 8], device=x.device)
    mappedRange = 2 ** (mappedBit - 1) - 1

    def inferBit(x: torch.Tensor):
        bins = torch.bucketize(x, boundaries)
        nBit = mappedBit[bins]
        scaleRange = 2 ** (nBit - 1) - 1
        return nBit, scaleRange

    maskedMaxAbsInlier = (
        torch.masked_fill(xTile.abs(), maskOutlier, -torch.inf).max(dim=-1).values
    )
    nBitInlier, scaleRange = inferBit(maskedMaxAbsInlier)
    # nBitInlier, scaleRange = inferBit(
    #     torch.masked_fill(xTile, maskOutlier, 0).std(dim=-1) * 3
    # )
    scaleInlier = (
        (scaleRange / maskedMaxAbsInlier)
        .clamp(min=1, max=1e7)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )
    xQuantizedInlier = torch.round(xTile * scaleInlier) / (scaleInlier + 1e-8)

    maskedMaxAbsOutlier = (
        torch.masked_fill(xTile.abs(), maskInlier, -torch.inf).max(dim=-1).values
    )
    nBitOutlier, scaleRange = inferBit(maskedMaxAbsOutlier)
    scaleOutlier = (
        (127 / maskedMaxAbsOutlier)
        .clamp(min=1, max=1e7)
        .unsqueeze(-1)
        .expand(xTile.shape)
    )
    xQuantizedOutlier = torch.round(xTile * scaleOutlier) / (scaleOutlier + 1e-8)

    # print(nBitInlier.float().mean(), nBitOutlier.float().mean(), x.abs().mean())

    # Statistics

    xTileQuantize = xTile[...]
    xTileQuantize[maskInlier] = xQuantizedInlier[maskInlier]
    xTileQuantize[maskOutlier] = xQuantizedOutlier[maskOutlier]

    xTileQuantize = xTileQuantize + muTile

    xTileQuantize = (
        xTileQuantize.view(-1, h // hGroup, w // wGroup, hGroup, wGroup)
        .transpose(-3, -2)
        .reshape(-1, x.shape[-2], x.shape[-1])
    ).to(x.dtype)

    # print(((xTileQuantize - x).abs()).mean())

    return xTileQuantize


def quantizeAdapt(x: torch.Tensor, sizeGroup: tuple, k: int, scaleBar: float = None):
    hGroup, wGroup = sizeGroup
    h, w = x.shape[-2:]

    assert k <= hGroup * wGroup
    assert h % hGroup == 0 and w % wGroup == 0

    xTile = (
        x.reshape(*x.shape[:-2], h // hGroup, hGroup, w // wGroup, wGroup)
        .transpose(-3, -2)
        .flatten(-2, -1)
    ).float()

    mappedBit = torch.tensor([2, 4, 6, 8], device=x.device)
    mappedRange = 2 ** (mappedBit - 1) - 1
    boundaries = mappedRange[:-1] / scaleBar

    def inferScale(x: torch.Tensor):
        xAbsMax = x.abs().max(dim=-1).values
        bins = torch.bucketize(xAbsMax, boundaries)
        nBit = mappedBit[bins]
        scale = ((2 ** (nBit - 1) - 1) / xAbsMax).clamp(min=1, max=1e7)
        scalePoT = scaleBar * 2 ** torch.floor(torch.log2(scale / scaleBar))

        return nBit, scalePoT.unsqueeze(-1).expand(x.shape)

    nBit, scale = inferScale(xTile)

    xQuantized = torch.round(xTile * scale) / (scale + 1e-8)

    # Statistics
    # stat.append(nBit.float().mean())
    # print(sum(stat) / len(stat))

    # xTileQuantize = xTileQuantize + muTile

    xQuantized = (
        xQuantized.reshape(*x.shape[:-2], h // hGroup, w // wGroup, hGroup, wGroup)
        .transpose(-3, -2)
        .reshape(*x.shape[:-2], x.shape[-2], x.shape[-1])
    ).to(x.dtype)

    # print(((xTileQuantize - x).abs()).mean())

    return xQuantized, nBit


def quantizeAdaptDyn(x: torch.Tensor, sizeGroup: tuple, k: int, scaleBar: float = None):
    hGroup, wGroup = sizeGroup
    h, w = x.shape[-2:]

    assert k <= hGroup * wGroup
    assert h % hGroup == 0 and w % wGroup == 0

    xTile = (
        x.reshape(*x.shape[:-2], h // hGroup, hGroup, w // wGroup, wGroup)
        .transpose(-3, -2)
        .flatten(-2, -1)
    ).float()

    mappedBit = torch.tensor([2, 4, 6, 8], device=x.device)
    mappedRange = 2 ** (mappedBit - 1) - 1
    boundaries = mappedRange[:-1] / scaleBar

    def inferScale(x: torch.Tensor):
        xAbsMax = x.abs().max(dim=-1).values
        bins = torch.bucketize(xAbsMax, boundaries)
        nBit = mappedBit[bins]
        scale = ((2 ** (nBit - 1) - 1) / xAbsMax).clamp(min=1, max=1e7)
        # scalePoT = scaleBar * 2 ** torch.floor(torch.log2(scale / scaleBar))

        return nBit, scale.unsqueeze(-1).expand(x.shape)

    nBit, scale = inferScale(xTile)

    xQuantized = torch.round(xTile * scale) / (scale + 1e-8)

    # Statistics
    # stat.append(nBit.float().mean())
    # print(sum(stat) / len(stat))

    # xTileQuantize = xTileQuantize + muTile

    xQuantized = (
        xQuantized.reshape(*x.shape[:-2], h // hGroup, w // wGroup, hGroup, wGroup)
        .transpose(-3, -2)
        .reshape(*x.shape[:-2], x.shape[-2], x.shape[-1])
    ).to(x.dtype)

    # print(((xTileQuantize - x).abs()).mean())

    return xQuantized, nBit
