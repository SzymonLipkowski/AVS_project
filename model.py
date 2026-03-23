import cv2
import numpy as np
import os
import pickle

_reference_median = None
_reference_std = None


def _preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    return norm


def build_reference_model(good_image_folder, n_images=45, save_path="reference_model.pkl"):
    global _reference_median, _reference_std

    images = []
    for i in range(n_images):
        path = os.path.join(good_image_folder, f"{i:03d}.png")
        img = cv2.imread(path)
        if img is None:
            print(f"  Warning: could not read {path}")
            continue
        images.append(_preprocess(img))

    if not images:
        raise FileNotFoundError(f"No good images found in '{good_image_folder}'")

    stack = np.array(images, dtype=np.float32)
    _reference_median = np.median(stack, axis=0)
    _reference_std    = np.std(stack, axis=0)
    _reference_std    = np.clip(_reference_std, 5.0, None)

    with open(save_path, "wb") as f:
        pickle.dump({"median": _reference_median, "std": _reference_std}, f)
    print(f"Reference model built from {len(images)} images → saved to '{save_path}'.")


def load_reference_model(path="reference_model.pkl"):
    global _reference_median, _reference_std
    with open(path, "rb") as f:
        data = pickle.load(f)
    _reference_median = data["median"]
    _reference_std    = data["std"]
    print(f"Reference model loaded from '{path}'.")


def _filter_components(mask, min_area, max_area=None, min_aspect_ratio=None):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if min_aspect_ratio is not None:
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            h = stats[lbl, cv2.CC_STAT_HEIGHT]
            if h == 0 or w == 0:
                continue
            aspect = max(w, h) / min(w, h)
            if aspect < min_aspect_ratio:
                continue
        result[labels == lbl] = 255
    return result


def predict(image, threshold_sigma=3.5, min_defect_area=300):
    global _reference_median, _reference_std

    if _reference_median is None:
        raise RuntimeError("Call build_reference_model() or load_reference_model() first.")

    proc  = _preprocess(image)
    score = np.abs(proc - _reference_median) / _reference_std

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    # ── Track A: large defects ─────────────────────────────────────────────
    mask_a = (score > threshold_sigma).astype(np.uint8) * 255
    mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_OPEN,  k_open)
    mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_CLOSE, k_close)
    mask_a = _filter_components(mask_a, min_area=min_defect_area)
    mask_a = cv2.dilate(mask_a, k_dilate)

    # ── Track B: thin/scratch defects ─────────────────────────────────────
    mask_b = (score > 2.5).astype(np.uint8) * 255
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k_open)
    # close with narrow kernel — preserve elongated shape
    k_close_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, k_close_thin)
    # keep only elongated components (scratches/abrasion lines)
    # reject small round blobs which are normal bristle tips
    mask_b = _filter_components(mask_b, min_area=60, max_area=5000, min_aspect_ratio=3.0)
    # small dilate to thicken the line slightly
    k_dilate_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_b = cv2.dilate(mask_b, k_dilate_thin)

    result = cv2.bitwise_or(mask_a, mask_b)
    return result