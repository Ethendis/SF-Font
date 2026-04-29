import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import torch
from skimage import morphology
import cv2

def skeleton(img, args):  # tensor
        re = []
        binary = Binary(img)
        skeleton0 = morphology.skeletonize(binary)  # 骨架提取
        skeleton = skeleton0.astype(np.uint8) * 255

        gray = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)

        dst = 255/255 - gray

        stacked_img = np.stack((dst,) * 3, axis=-1)
        stacked_img = torch.tensor(stacked_img)
        stacked_img = stacked_img.permute(2, 0, 1)
        # print(stacked_img.shape)
        re.append(stacked_img)
        img_skeleton_tensor = torch.stack(re)
        img_skeleton_tensor =img_skeleton_tensor.cuda(args.gpu)

        return img_skeleton_tensor

def Binary(img):

    img = img.cpu()
    for i in range(img.size(0)):
        x = img[i, :, :, :]
        x = x.permute(1, 2, 0)
        x = x.numpy()
        # print(type(x))
        _, binary = cv2.threshold(x, 200 / 255, 255 / 255, cv2.THRESH_BINARY_INV)  # 二值化处理
        binary[binary == 255 / 255] = 1
        return binary