import cv2
from model import predict
import numpy as np

def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union between two masks
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    
    if union == 0:
        return 1.0  # perfect match if both empty
    return intersection / union

# Paths to your defective images and GT masks
defective_folder = "toothbrush\\train\\defective"
gt_folder = "toothbrush\\ground_truth\\defective"

ious = []

for i in range(30):
    # Adjust filenames to match your naming convention
    img_path = f"{defective_folder}/{i:03d}.png"
    gt_path = f"{gt_folder}/{i:03d}_mask.png"
    
    img = cv2.imread(img_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or gt_mask is None:
        print(f"Skipping {i:03d}, file missing")
        continue
    
    pred_mask = predict(img)
    cv2.imshow("predict",pred_mask)
    cv2.imshow("GT",gt_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ensure GT and prediction have same shape
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
    
    iou = compute_iou(pred_mask, gt_mask)
    ious.append(iou)
    print(f"Image {i:03d}: IoU = {iou:.4f}")


avg_iou = np.mean(ious)
print(f"\nAverage IoU over {len(ious)} images: {avg_iou:.4f}")
