import cv2
import numpy as np
from imgaug import augmenters as iaa
from scipy.interpolate import griddata
import sys

INTERPOLATION = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC
}


class GridDistortion:

    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, img):
        should_transform = np.random.choice(np.arange(0, 2), p=[1 - self.prob, self.prob])
        img = np.array(img)
        if should_transform:
            return warp_image(img)
        return img


def warp_image(img, random_state=None, **kwargs):
    if random_state is None:
        random_state = np.random.RandomState()

    w_mesh_interval = kwargs.get('w_mesh_interval', 12)
    w_mesh_std = kwargs.get('w_mesh_std', 1.5)

    h_mesh_interval = kwargs.get('h_mesh_interval', 12)
    h_mesh_std = kwargs.get('h_mesh_std', 1.5)

    interpolation_method = kwargs.get('interpolation', 'linear')

    h, w = img.shape[:2]

    if kwargs.get("fit_interval_to_image", True):
        # Change interval so it fits the image size
        w_ratio = w / float(w_mesh_interval)
        h_ratio = h / float(h_mesh_interval)

        w_ratio = max(1, round(w_ratio))
        h_ratio = max(1, round(h_ratio))

        w_mesh_interval = w / w_ratio
        h_mesh_interval = h / h_ratio
        ############################################

    # Get control points
    source = np.mgrid[0:h+h_mesh_interval:h_mesh_interval, 0:w+w_mesh_interval:w_mesh_interval]
    source = source.transpose(1,2,0).reshape(-1,2)

    if kwargs.get("draw_grid_lines", False):
        if len(img.shape) == 2:
            color = 0
        else:
            color = np.array([0,0,255])
        for s in source:
            img[int(s[0]):int(s[0])+1,:] = color
            img[:,int(s[1]):int(s[1])+1] = color

    # Perturb source control points
    destination = source.copy()
    source_shape = source.shape[:1]
    destination[:,0] = destination[:,0] + random_state.normal(0.0, h_mesh_std, size=source_shape)
    destination[:,1] = destination[:,1] + random_state.normal(0.0, w_mesh_std, size=source_shape)

    # Warp image
    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(destination, source, (grid_x, grid_y), method=interpolation_method).astype(np.float32)
    map_x = grid_z[:,:,1]
    map_y = grid_z[:,:,0]
    warped = cv2.remap(img, map_x, map_y, INTERPOLATION[interpolation_method], borderValue=(0,0,0))

    return warped


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.35, iaa.GaussianBlur(sigma=(0, 1.5))),
            iaa.Sometimes(0.35,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.05)),
                                     iaa.CoarseDropout(0, size_percent=0.05)])),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)