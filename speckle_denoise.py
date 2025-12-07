import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from PIL import Image

def speckle_nlm(img_np, patch_size=15, patch_distance=35, h_factor=3):

    if img_np.ndim == 3:
        img_gray = np.mean(img_np, axis=2)
    else:
        img_gray = img_np.astype(np.float32)

    img_norm = img_gray / 255.0 + 1e-6
    log_img = np.log(img_norm)

    sigma_est = np.mean(estimate_sigma(log_img, channel_axis=None))

    denoised_log = denoise_nl_means(
        log_img,
        h=h_factor * sigma_est,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=None
    )

    denoised = np.exp(denoised_log)
    denoised = (denoised - denoised.min()) / (denoised.max() - denoised.min() + 1e-8)
    denoised = (denoised * 255).astype(np.uint8)
    return denoised



class SpeckleNLMTransform:


    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)
        denoised_np = speckle_nlm(img_np)
        return Image.fromarray(denoised_np)
