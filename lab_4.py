"""
lab_4.py
========
Mini Project 1 – Toothbrush Bristle Defect Detection
-----------------------------------------------------
Step 1 : Build the reference model from good training images.
Step 2 : Evaluate defect-detection on the test split using IoU (segmentation)
         and binary classification metrics (confusion matrix / report).
"""

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from model import build_reference_model, load_reference_model, predict
import os

# ── Paths ──────────────────────────────────────────────────────────────────
GOOD_TRAIN_FOLDER  = r"toothbrush\train\good"
BAD_TRAIN_FOLDER   = r"toothbrush\train\defective"
GT_FOLDER          = r"toothbrush\ground_truth\defective"
MODEL_PATH         = "reference_model.pkl"

N_GOOD_TOTAL  = 60   # total good images available
N_BAD_TOTAL   = 30   # total bad  images available
N_GOOD_TRAIN  = 45   # good images used to BUILD the reference
N_BAD_TRAIN   = 20   # bad  images kept for training  (not evaluated)
# test splits: good[45:] (15 images), bad[20:] (10 images)

# ── 1. Build (or load) reference model ────────────────────────────────────
if os.path.exists(MODEL_PATH):
    load_reference_model(MODEL_PATH)
else:
    build_reference_model(GOOD_TRAIN_FOLDER, n_images=N_GOOD_TRAIN, save_path=MODEL_PATH)

# ── 2. Helpers ─────────────────────────────────────────────────────────────
DEFECT_PIXEL_RATIO_THRESHOLD = 0.005   # fraction of image that must be anomalous
                                        # to classify image as "defective"

def classify(image):
    """Return 0='defective', 1='good' for a single BGR image."""
    mask = predict(image)
    ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return 0 if ratio > DEFECT_PIXEL_RATIO_THRESHOLD else 1, mask


def compute_iou(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin   = (gt_mask  > 0).astype(np.uint8)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or (pred_bin, gt_bin).sum()
    return 1.0 if union == 0 else inter / union


# ── 3. Classification evaluation ───────────────────────────────────────────
y_true, y_pred = [], []

# good test images (should be classified as 1 = good)
for i in range(N_GOOD_TRAIN, N_GOOD_TOTAL):
    path = os.path.join(GOOD_TRAIN_FOLDER, f"{i:03d}.png")
    img  = cv2.imread(path)
    if img is None:
        print(f"  Missing: {path}")
        continue
    label, _ = classify(img)
    y_true.append(1)
    y_pred.append(label)

# bad test images (should be classified as 0 = defective)
for i in range(N_BAD_TRAIN, N_BAD_TOTAL):
    path = os.path.join(BAD_TRAIN_FOLDER, f"{i:03d}.png")
    img  = cv2.imread(path)
    if img is None:
        print(f"  Missing: {path}")
        continue
    label, _ = classify(img)
    y_true.append(0)
    y_pred.append(label)

print("\n=== Classification Results ===")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["defective", "good"]))

# ── 4. Segmentation evaluation (IoU) on bad test images ───────────────────
ious = []
for i in range(N_BAD_TRAIN, N_BAD_TOTAL):
    img_path = os.path.join(BAD_TRAIN_FOLDER, f"{i:03d}.png")
    gt_path  = os.path.join(GT_FOLDER,        f"{i:03d}_mask.png")

    img     = cv2.imread(img_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if img is None or gt_mask is None:
        print(f"  Skipping {i:03d} – file missing")
        continue

    pred_mask = predict(img)
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

    iou = compute_iou(pred_mask, gt_mask)
    ious.append(iou)
    print(f"  Image {i:03d}: IoU = {iou:.4f}")

print(f"\nAverage IoU over {len(ious)} test defective images: {np.mean(ious):.4f}")

# ── 5. Optional: visual inspection ────────────────────────────────────────
SHOW_VISUALS = False   # set to True to step through images interactively

if SHOW_VISUALS:
    for i in range(N_BAD_TRAIN, N_BAD_TOTAL):
        img_path = os.path.join(BAD_TRAIN_FOLDER, f"{i:03d}.png")
        gt_path  = os.path.join(GT_FOLDER,        f"{i:03d}_mask.png")
        img     = cv2.imread(img_path)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if img is None or gt_mask is None:
            continue
        pred_mask = predict(img)
        overlay = img.copy()
        overlay[pred_mask > 0] = (0, 0, 255)   # red = predicted defect
        cv2.imshow(f"Image {i:03d}", overlay)
        cv2.imshow(f"Pred mask {i:03d}", pred_mask)
        cv2.imshow(f"GT mask   {i:03d}", gt_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
