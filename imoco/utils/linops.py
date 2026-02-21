import sigpy as sp
import numpy as np


def Vstacks(L_Linop, oshape, ishape):
    assert oshape[0] == len(L_Linop), "Number of Linop mismatch!"

    Linops = sp.linop.Vstack(L_Linop)
    o_vec_len = 1
    for tmp in oshape:
        o_vec_len = o_vec_len * tmp

    R1 = sp.linop.Reshape(oshape=(o_vec_len,), ishape=oshape)
    Linops = R1.H * Linops

    return Linops


def Diags(L_Linop, oshape, ishape):
    assert oshape[0] == ishape[0], "First dim mismatch!"
    assert oshape[0] == len(L_Linop), "Number of Linop mismatch!"
    Linops = sp.linop.Diag(L_Linop)
    i_vec_len = 1
    for tmp in ishape:
        i_vec_len = i_vec_len * tmp
    o_vec_len = 1
    for tmp in oshape:
        o_vec_len = o_vec_len * tmp

    R1 = sp.linop.Reshape(oshape=(o_vec_len,), ishape=oshape)
    R2 = sp.linop.Reshape(oshape=(i_vec_len,), ishape=ishape)
    Linops = R1.H * Linops * R2

    return Linops


def DLD(Linop, device=sp.Device(-1), idevice=sp.Device(-1)):
    B1 = sp.linop.ToDevice(Linop.ishape, idevice=idevice, odevice=device)
    B2 = sp.linop.ToDevice(Linop.oshape, idevice=idevice, odevice=device)
    Linop = B2.H * Linop * B1
    return Linop


def NFTs(ishape, coord, device=sp.Device(-1), idevice=sp.Device(-1)):
    n_Channel = ishape[0]
    oshape = list((n_Channel,)) + list(coord.shape[:-1])

    NFT = sp.linop.NUFFT(ishape[1:], coord=coord)
    nfts = Diags([DLD(NFT, device=device, idevice=idevice) for i in range(n_Channel)], oshape, ishape)

    return nfts


def FD(ishape, axes=None):
    """Finite difference gradient linear operator.

    Args:
        ishape (tuple of ints): Input shape.
        axes (tuple of ints): Axes along which to compute finite differences.

    Returns:
        Linop: Finite difference gradient operator.
    """
    I = sp.linop.Identity(ishape)
    axes = sp.util._normalize_axes(axes, len(ishape))
    ndim = len(ishape)
    linops = []
    for i in axes:
        D = I - sp.linop.Circshift(ishape, [0] * i + [1] + [0] * (ndim - i - 1))
        R = sp.linop.Reshape([1] + list(ishape), ishape)
        linops.append(R * D)

    G = sp.linop.Vstack(linops, axis=0)
    return G


def TVt_prox(X, lamda, iter_max=10, tv_device=sp.Device(-1)):
    """Proximal operator for temporal total variation.

    Args:
        X (ndarray): Input array.
        lamda (float): Regularization weight.
        iter_max (int): Maximum iterations.
        tv_device (Device): Computing device.

    Returns:
        ndarray: Denoised array.
    """
    xp = sp.Device(tv_device).xp
    X = sp.to_device(X, tv_device)
    scale = xp.max(xp.abs(X))
    X = X / scale
    TVt = FD(X.shape, axes=(0,))
    X_b = X
    Y = TVt * X
    Y = Y / (xp.abs(Y) + 1e-9) * xp.minimum(xp.abs(Y) + 1e-9, 1)
    for _ in range(iter_max):
        X_b = X_b - ((X_b - X) + lamda * TVt.H * Y)
        Y = Y + lamda * TVt * X_b
        Y = Y / (xp.abs(Y) + 1e-9) * xp.minimum(xp.abs(Y) + 1e-9, 1)

    X_b = X_b * scale
    return sp.to_device(X_b)
