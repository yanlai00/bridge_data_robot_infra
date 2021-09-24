import numpy as np
import cv2
import time
from widowx_envs.utils.utils import np_unstack
import torch

def get_random_crop(images, size, center_crop=False):
    # assume images are in Tlen, C, H, W or C, H, W  order
    im_size = np.array(images.shape[-2:])
    if center_crop:
        shift_r = (im_size[0] - size[0]) // 2
        shift_c = (im_size[1] - size[1]) // 2
    else:
        shift_r = np.random.randint(0, im_size[0] - size[0], dtype=np.int)
        shift_c = np.random.randint(0, im_size[1] - size[1], dtype=np.int)

    if len(images.shape) == 4:
        return images[:, :, shift_r : shift_r + size[0], shift_c : shift_c + size[1]]
    elif len(images.shape) == 3:
        return images[:, shift_r : shift_r + size[0], shift_c : shift_c + size[1]]
    else:
        raise ValueError('wrong shape ', images.shape)


def get_random_color_aug(images, scale, minus_one_to_one_range=False):
    """
    alternative color jitter based on cv2
    :param images: shape: tlen/batch, 3, height, width
    :param scale:
    :return:
    """
    if len(images.shape) == 4:
        tlen = images.shape[0]
        if isinstance(images, torch.Tensor):
            images = images.permute(0, 2, 3, 1)
        elif isinstance(images, np.ndarray):
            images = images.transpose(0, 2, 3, 1)
        else:
            raise ValueError('not supported data type!')
        images = np.concatenate(np_unstack(images, 0), axis=0)
        assert images.dtype == np.float32
        if minus_one_to_one_range:
            images = (images + 1)/2  # convert to 0 to 1 range
        assert np.min(images) >= 0 and np.max(images) <= 1
        images = (images*255).astype(np.uint8)
        hsv = np.asarray(cv2.cvtColor(images, cv2.COLOR_RGB2HSV))
        hsv_rand = np.random.uniform(np.ones(3) - scale, np.ones(3) + scale)
        hsv = np.clip(hsv * hsv_rand[None, None], 0, 255)
        hsv = hsv.astype(np.uint8)
        rgb = np.asarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        images = np.stack(np.split(rgb, tlen, axis=0), axis=0)
        images = images.transpose(0, 3, 1, 2)
        images = images.astype(np.float32)/255.
        return images
    elif len(images.shape) == 3:
        if isinstance(images, torch.Tensor):
            images = images.permute(1, 2, 0)
        elif isinstance(images, np.ndarray):
            images = images.transpose(1, 2, 0)
        else:
            raise ValueError('not supported data type!')
        assert images.dtype == np.float32
        if minus_one_to_one_range:
            images = (images + 1)/2  # convert to 0 to 1 range
        assert np.min(images) >= 0 and np.max(images) <= 1
        images = (images*255).astype(np.uint8)
        hsv = np.asarray(cv2.cvtColor(images, cv2.COLOR_RGB2HSV))
        hsv_rand = np.random.uniform(np.ones(3) - scale, np.ones(3) + scale)
        hsv = np.clip(hsv * hsv_rand[None, None], 0, 255)
        hsv = hsv.astype(np.uint8)
        images = np.asarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        images = images.transpose(2, 0, 1)
        images = images.astype(np.float32)/255.
        return images
    else:
        raise ValueError('wrong shape ', images.shape)

