# visualize_denoise.py
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from speckle_denoise import speckle_nlm 

def visualize_denoise(image_paths, save_fig=True, output_dir="denoise_vis"):

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)


        denoised = speckle_nlm(img)


        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img.astype(np.uint8), cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(denoised, cmap='gray')
        plt.title("NLM Denoised")
        plt.axis("off")

        # 保存图像
        if save_fig:
            base = os.path.basename(img_path)
            save_path = os.path.join(output_dir, f"compare_{base}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")

        plt.show()


if __name__ == "__main__":

    images = [
        "/Users/chachaen/Desktop/dataset/1/11 - air.jpg",
        "/Users/chachaen/Desktop/bladdemask/valid/Apache_C62_Bladder_mp4-0000_jpg.rf.6ac5d107eb6119b4920bad50a6410b07.jpg",
    ]
    visualize_denoise(images)
