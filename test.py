import cv2
import numpy as np
import os
from model import build_reference_model, load_reference_model, predict

# ── Paths ──────────────────────────────────────────────────────────────────
GOOD_TRAIN_FOLDER = r"toothbrush\train\good"
DEFECTIVE_FOLDER  = r"toothbrush\train\defective"
GT_FOLDER         = r"toothbrush\ground_truth\defective"
MODEL_PATH        = "reference_model.pkl"


def compute_iou(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin   = (gt_mask  > 0).astype(np.uint8)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or (pred_bin, gt_bin).sum()
    return 1.0 if union == 0 else inter / union


# ── Load / build model ─────────────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    load_reference_model(MODEL_PATH)
else:
    print("No saved model found – building from good training images …")
    build_reference_model(GOOD_TRAIN_FOLDER, n_images=45, save_path=MODEL_PATH)

# ── Evaluate ───────────────────────────────────────────────────────────────
ious = []

for i in range(30):
    img_path = os.path.join(DEFECTIVE_FOLDER, f"{i:03d}.png")
    gt_path  = os.path.join(GT_FOLDER,        f"{i:03d}_mask.png")

    img     = cv2.imread(img_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if img is None or gt_mask is None:
        print(f"Skipping {i:03d} – file missing")
        continue

    pred_mask = predict(img)

    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

    iou = compute_iou(pred_mask, gt_mask)
    ious.append(iou)
    print(f"Image {i:03d}: IoU = {iou:.4f}")

    # ── visual: overlay predicted mask in red on original ──────────────────
    overlay = img.copy()
    overlay[pred_mask > 0] = (0, 0, 255)   # red pixels = predicted defect

    cv2.imshow("predict (red overlay)", overlay)
    cv2.imshow("predict mask", pred_mask)
    cv2.imshow("GT mask", gt_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

avg_iou = np.mean(ious)
print(f"\nAverage IoU over {len(ious)} images: {avg_iou:.4f}")
