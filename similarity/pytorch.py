import cv2
import numpy as np
import torch

from utils import showImage


class PyTorchLoss:
    # shape: [N, C, H, W]
    @staticmethod
    def ssim(x, y, is_normalized=False):
        k1 = 0.01
        k2 = 0.03
        L = 1.0 if is_normalized else 255.0

        c1 = np.power(k1 * L, 2.0)
        c2 = np.power(k2 * L, 2.0)
        c3 = c2 / 2.0

        ux = x.mean()
        uy = y.mean()

        std_x = x.std()
        std_y = y.std()

        xy = (x - ux) * (y - uy)
        std_xy = xy.mean()

        l_xy = (2.0 * ux * uy + c1) / (ux ** 2.0 + uy ** 2.0 + c1)
        c_xy = (2.0 * std_x * std_y + c2) / (std_x ** 2.0 + std_y ** 2.0 + c2)
        s_xy = (std_xy + c3) / (std_x * std_y + c3)

        ssim = l_xy * c_xy * s_xy
        ssim = torch.clamp(ssim, -1.0, 1.0)

        return ssim

    @staticmethod
    def ssim3(x, y, is_normalized=True):
        xr, xg, xb = torch.split(x, 1, dim=0)
        yr, yg, yb = torch.split(y, 1, dim=0)

        r = PyTorchLoss.ssim(xr, yr, is_normalized)
        g = PyTorchLoss.ssim(xg, yg, is_normalized)
        b = PyTorchLoss.ssim(xb, yb, is_normalized)

        result = (r + g + b) / 3.0

        return result

    @staticmethod
    def ssim4(x, y, is_normalized=True):
        ssim4_loss = 0
        n_image = 0
        for x_image, y_image in zip(x, y):
            ssim4_loss += PyTorchLoss.ssim3(x_image, y_image, is_normalized)
            n_image += 1

        ssim4_loss /= n_image

        return ssim4_loss


if __name__ == "__main__":
    # ================================================================================
    img1 = cv2.imread("data/splice1.png")
    img2 = cv2.imread("data/splice2.png")
    img3 = cv2.imread("data/splice3.png")
    img4 = cv2.imread("data/splice4.png")
    images = [img1, img2, img3, img4]
    showImage(img1, img2, img3, img4)

    dsts = []
    for i in range(len(images)):
        img = images[i] / 255.0
        # INTER_CUBIC = cv2.resize(_img, (_resize_cols, _resize_rows), interpolation=cv2.INTER_CUBIC)
        images[i] = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        dst = cv2.GaussianBlur(images[i].copy(), (5, 5), 0)
        dsts.append(dst)

    images = np.array(images)
    dsts = np.array(dsts)

    pt_images = torch.from_numpy(images)
    pt_images = pt_images.permute(0, 3, 1, 2)  # pt_images.shape = torch.Size([4, 3, 256, 256])
    pt_dsts = torch.from_numpy(dsts)
    pt_dsts = pt_dsts.permute(0, 3, 1, 2)  # pt_dsts.shape = torch.Size([4, 3, 256, 256])

    pt_image1 = pt_images[0]
    print("pt_image1.shape:", pt_image1.shape)  # torch.Size([3, 256, 256])
    result = torch.split(pt_image1, 1, dim=0)

    pt_loss1 = PyTorchLoss.ssim(pt_images[0], pt_dsts[0], is_normalized=True)
    print("pt_loss1:", pt_loss1)  # pt_loss1: tensor(0.9748, dtype=torch.float64)
    pt_loss2 = PyTorchLoss.ssim(pt_images[1], pt_dsts[1], is_normalized=True)
    print("pt_loss2:", pt_loss2)  # pt_loss2: tensor(0.9754, dtype=torch.float64)
    pt_loss3 = PyTorchLoss.ssim(pt_images[2], pt_dsts[2], is_normalized=True)
    print("pt_loss3:", pt_loss3)  # pt_loss3: tensor(0.9838, dtype=torch.float64)
    pt_loss4 = PyTorchLoss.ssim(pt_images[3], pt_dsts[3], is_normalized=True)
    print("pt_loss4:", pt_loss4)  # pt_loss4: tensor(0.9857, dtype=torch.float64)
    # pt_total_loss: tensor(0.9851, dtype=torch.float64)
    pt_total_loss = torch.mean(torch.tensor([pt_loss1, pt_loss2, pt_loss3, pt_loss4]))
    pt_loss = PyTorchLoss.ssim(torch.from_numpy(images), torch.from_numpy(dsts), is_normalized=True)
    print("pt_loss:", pt_loss)  # pt_loss: tensor(0.9851, dtype=torch.float64)

    """
    上面會將圖片的三個通道"共同"計算 ssim 值
    下面則將圖片的三個通道"分別"計算 ssim 值
    """

    pt_ssim3_loss = 0
    for img, dst in zip(pt_images, pt_dsts):
        pt_ssim3_loss += PyTorchLoss.ssim3(img, dst, is_normalized=True)
    pt_ssim3_loss /= pt_images.shape[0]
    # pt_ssim3_loss: 0.9782
    print("pt_ssim3_loss:", pt_ssim3_loss)

    # PyTorch.ssim4: tensor(0.9782, dtype=torch.float64)
    pt_dsts = pt_dsts.requires_grad_(True)
    ssim4_loss = PyTorchLoss.ssim4(pt_images, pt_dsts, is_normalized=True)
    print("PyTorch.ssim4:", ssim4_loss)
