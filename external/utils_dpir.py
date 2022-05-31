import torch


def test_mode(model, L, mode=0, refield=32, min_size=256, sf=1, modulo=1):
    '''
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Some testing modes
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # (0) normal: test(model, L)
    # (1) pad: test_pad(model, L, modulo=16)
    # (2) split: test_split(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (3) x8: test_x8(model, L, modulo=1)
    # (4) split and x8: test_split_x8(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (4) split only once: test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # ---------------------------------------
    '''
    if mode == 0:
        raise NotImplementedError
    elif mode == 1:
        raise NotImplementedError
    elif mode == 2:
        raise NotImplementedError
    elif mode == 3:
        raise NotImplementedError
    elif mode == 4:
        raise NotImplementedError
    elif mode == 5:
        E = test_onesplit(model, L, refield, min_size, sf, modulo)
    return E


def test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]

    top = slice(0, (h//2//refield+1)*refield)
    bottom = slice(h - (h//2//refield+1)*refield, h)
    left = slice(0, (w//2//refield+1)*refield)
    right = slice(w - (w//2//refield+1)*refield, w)
    Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
    Es = [model(Ls[i]) for i in range(4)]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
    E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
    E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
    E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E
