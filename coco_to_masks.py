

import os
import json
import argparse
from PIL import Image, ImageDraw
import numpy as np

try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCO = True
except Exception:
    HAS_PYCOCO = False

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def polygons_to_mask(polygons, image_size):


    w, h = image_size
    mask_img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        # some polygons may be nested lists (segmentation from COCO)
        if isinstance(poly[0], list) or isinstance(poly[0], tuple):
            # list of rings
            for p in poly:
                draw.polygon(p, outline=1, fill=1)
        else:
            draw.polygon(poly, outline=1, fill=1)
    mask = np.array(mask_img, dtype=np.uint8) * 255
    return mask

def rle_to_mask(rle, image_size):

    if not HAS_PYCOCO:
        raise RuntimeError("RLE encountered but pycocotools not installed. pip install pycocotools")
    h = image_size[1]
    w = image_size[0]
    # mask_utils.decode expects rle in COCO format
    m = mask_utils.decode(rle)  # returns (H,W) with 0/1 values
    if m.ndim == 3:
        m = m.any(axis=2).astype(np.uint8)
    return (m * 255).astype(np.uint8)

def main(coco_json, images_dir, out_dir, split=None, vis=False):
    # load json
    print("Loading JSON:", coco_json)
    with open(coco_json, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data.get('images', [])}
    annotations = data.get('annotations', [])


    ann_map = {}
    for ann in annotations:
        img_id = ann['image_id']
        seg = ann.get('segmentation', None)
        if seg is None:
            continue
        ann_map.setdefault(img_id, []).append(ann)

    ensure_dir(out_dir)

    n_missing = 0
    processed = 0

    for img_id, img_info in images.items():
        filename = img_info.get('file_name') or img_info.get('filename')
        width = img_info.get('width')
        height = img_info.get('height')


        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            n_missing += 1
            print(f"[WARN] image file not found: {img_path}")
            continue
        if width is None or height is None:
            with Image.open(img_path) as im:
                width, height = im.size

        # create empty mask
        mask_total = np.zeros((height, width), dtype=np.uint8)

        anns = ann_map.get(img_id, [])
        for ann in anns:
            seg = ann.get('segmentation')
            if seg is None:
                continue
            if isinstance(seg, dict):
                # RLE
                try:
                    m = rle_to_mask(seg, (width, height))
                except RuntimeError as e:
                    raise
            else:


                polygons = []
                if len(seg) == 0:
                    continue

                if isinstance(seg[0], list):
                    polygons = seg
                else:

                    polygons = [seg]
                m = polygons_to_mask(polygons, (width, height))

            mask_total = np.maximum(mask_total, m)

        # save mask
        mask_fname = os.path.splitext(filename)[0] + "_mask.png"
        mask_path = os.path.join(out_dir, mask_fname)
        Image.fromarray(mask_total).save(mask_path)
        processed += 1

        if vis and processed <= 5:

            import matplotlib.pyplot as plt
            with Image.open(img_path) as im:
                fig, ax = plt.subplots(1,1, figsize=(6,6))
                ax.imshow(im.convert('L'), cmap='gray')
                ax.imshow(mask_total, cmap='jet', alpha=0.35)
                ax.set_title(filename)
                ax.axis('off')
                plt.show()

    print(f"Processed {processed} images. Missing files: {n_missing}. Masks saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', required=True, help="path to _annotations.coco.json")
    parser.add_argument('--images_dir', required=True, help="path to images folder")
    parser.add_argument('--out_dir', required=True, help="where to save masks")
    parser.add_argument('--vis', action='store_true', help="show first few overlays")
    args = parser.parse_args()
    main(args.coco, args.images_dir, args.out_dir, vis=args.vis)
