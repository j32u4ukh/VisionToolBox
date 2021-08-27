import numpy as np


def ssim(x, y, is_normalized=False):
    k1 = 0.01
    k2 = 0.03
    L = 1.0 if is_normalized else 255.0
    c1 = np.power(k1 * L, 2)
    c2 = np.power(k2 * L, 2)
    c3 = c2 / 2
    
    ux = x.mean()
    uy = y.mean()

    std_x = x.std()
    std_y = y.std()

    xy = (x - ux) * (y - uy)
    std_xy = xy.mean()

    l_xy = (2 * ux * uy + c1) / (np.power(ux, 2) + np.power(uy, 2) + c1)
    c_xy = (2 * std_x * std_y + c2) / (np.power(std_x, 2) + np.power(std_y, 2) + c2)
    s_xy = (std_xy + c3) / (std_x * std_y + c3)

    _ssim = l_xy * c_xy * s_xy
    _ssim = np.clip(_ssim, -1.0, 1.0)

    return _ssim


def ssim3(x, y, is_normalized=True):
    x1, x2, x3 = np.split(x, 3, axis=2)
    y1, y2, y3 = np.split(y, 3, axis=2)

    s1 = ssim(x1, y1, is_normalized)
    s2 = ssim(x2, y2, is_normalized)
    s3 = ssim(x3, y3, is_normalized)

    result = (s1 + s2 + s3) / 3.0

    return result


def ssim4(x, y, is_normalized=False):
    each_loss = []

    for _x, _y in zip(x, y):
        each_loss.append(ssim3(_x, _y, is_normalized))

    each_loss = np.array(each_loss)

    total_loss = each_loss.mean()

    return total_loss, each_loss


if __name__ == "__main__":
    pass
