import numpy as np
from PIL import Image


def resize_and_pad_image(image, target_size=1024):
    orig_h, orig_w = image.shape[:2]
    
    # Resize keeping aspect ratio
    scale = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    image_resized = np.array(Image.fromarray(image).resize((new_w, new_h)))
    
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    image_padded = np.pad(
        image_resized,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )
    
    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "scale": scale
    }
    
    image_pil = Image.fromarray(image_padded)
    
    return image_pil, meta


def reverse_resize_and_pad(padded_img, meta):
    pad_top = meta["pad_top"]
    pad_bottom = meta["pad_bottom"]
    pad_left = meta["pad_left"]
    pad_right = meta["pad_right"]

    unpadded = padded_img[
        pad_top : padded_img.shape[0] - pad_bottom,
        pad_left : padded_img.shape[1] - pad_right
    ]

    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]

    restored = np.array(Image.fromarray(unpadded).resize((orig_w, orig_h)))

    return restored
