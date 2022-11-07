import numpy as np
import torch

# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler
from PIL import Image

# pathmgr = PathManager()
# pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)

DETECTOR_PATH = "/data/sandcastle/boxes/fbsource/xplat/arfx/tracking/body/models/158/body_tracking_model_init.pb"


def resize(img):
    # resize shortest edge
    min_size = 224
    max_size = 400
    h, w = img.shape[:2]
    size = min_size
    scale = size * 1.0 / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    # print(neww)
    # print(newh)
    pil_image = Image.fromarray(img)
    pil_image = pil_image.resize((neww, newh), Image.BILINEAR)
    return np.asarray(pil_image), 1.0 / scale


def dataloader_preprocess(rgb):
    # convert to bgr
    bgr = rgb[:, :, ::-1]
    bgr_resized, rescale_factor = resize(bgr)
    return (
        torch.as_tensor(bgr_resized.transpose(2, 0, 1).astype("float32")),
        rescale_factor,
    )


def init_detector():
    # with pathmgr.open(DETECTOR_PATH, "rb") as f:
    with open(DETECTOR_PATH, "rb") as f:
        bbox_model = torch.jit.load(f)
    return bbox_model


def inference(img, model):
    img, rescale_factor = dataloader_preprocess(img)
    preprocessed_input = [
        img.unsqueeze(0),
        torch.Tensor([img.shape[1], img.shape[2], 1.0]).unsqueeze(0),
    ]
    bbox, confidence, var, kpts = list(model(preprocessed_input))
    bbox *= rescale_factor
    kpts *= rescale_factor

    return bbox, confidence, var, kpts
