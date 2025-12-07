
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi



def load_gray(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)

def load_mask(path):
    m = Image.open(path).convert("L")
    arr = np.array(m, dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def build_rois(mask, wall_px=4):

    lumen = mask.astype(bool)

    dil = ndi.binary_dilation(lumen, iterations=wall_px)
    ero = ndi.binary_erosion(lumen, iterations=max(1, wall_px // 2))

    wall = np.logical_and(dil, np.logical_not(ero))
    outer = np.logical_not(dil)

    return lumen, wall, outer


def srad(img_float, iters=30, lam=0.25):
    I = img_float.copy().astype(np.float64)
    I = np.clip(I, 1e-8, 1.0)

    for _ in range(iters):
        dN = np.zeros_like(I); dS=np.zeros_like(I)
        dW=np.zeros_like(I); dE=np.zeros_like(I)

        dN[1:,:]  = I[1:,:] - I[:-1,:]
        dS[:-1,:] = I[:-1,:] - I[1:,:]
        dW[:,1:]  = I[:,1:] - I[:,:-1]
        dE[:,:-1] = I[:,:-1] - I[:,1:]

        G2 = (dN**2 + dS**2 + dW**2 + dE**2) / (I**2)
        L  = (dN + dS + dW + dE) / I

        num = 0.5 * G2 - (1/16) * (L**2)
        den = 1 + 0.25 * L
        q2  = num / (den + 1e-12)

        q0 = np.mean(q2)
        c = 1 / (1 + (q2 - q0) / (q0 * (1+q0) + 1e-12))
        c = np.clip(c, 0, 1)

        cN=np.zeros_like(c); cS=np.zeros_like(c)
        cW=np.zeros_like(c); cE=np.zeros_like(c)

        cN[1:,:]  = c[1:,:]
        cS[:-1,:] = c[:-1,:]
        cW[:,1:]  = c[:,1:]
        cE[:,:-1] = c[:,:-1]

        div = cN*dN + cS*dS + cW*dW + cE*dE
        I = I + (lam/4)*div
        I = np.clip(I, 0, 1)

    return I


def roi_filter(img, mask, wall_px=4):


    lumen, wall, outer = build_rois(mask, wall_px=wall_px)


    lumen_srad = srad(img_f, iters=30, lam=0.25)
    lumen_srad_u8 = (lumen_srad * 255).astype(np.uint8)

    wall_bilateral = cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)

    outer_gauss = cv2.GaussianBlur(img, (5,5), sigmaX=1.0)


    result = outer_gauss.copy()
    result[wall] = wall_bilateral[wall]
    result[lumen] = lumen_srad_u8[lumen]


    result = cv2.GaussianBlur(result, (3,3), sigmaX=0.5)

    return result


def show_result(orig, mask, den):
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(orig, cmap='gray')
    plt.title("Input file:Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(mask*255, cmap='gray')
    plt.title("Input file: Bladder Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(den, cmap='gray')
    plt.title("ROI Despeckled")
    plt.axis("off")

    plt.show()



def main():

    img_path  = "/Users/chachaen/Desktop/bladdemask/train/Apache_C62_Bladder_mp4-0006_jpg.rf.63d225f0ab92b163393686feda8cd27f.jpg"
    mask_path = "/Users/chachaen/Desktop/bladdemask/train_masks/Apache_C62_Bladder_mp4-0006_jpg.rf.63d225f0ab92b163393686feda8cd27f_mask.png"

    img  = load_gray(img_path)
    mask = load_mask(mask_path)

    den = roi_filter(img, mask, wall_px=4)

    show_result(img, mask, den)


if __name__ == "__main__":
    main()
