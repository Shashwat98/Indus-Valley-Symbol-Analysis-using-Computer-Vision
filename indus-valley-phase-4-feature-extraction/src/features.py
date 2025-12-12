# src/features.py
from typing import Dict, Tuple
import numpy as np
import cv2


# ============= SIMPLE FEATURE SET (A) =============

def _hu_moments_from_gray(gray: np.ndarray) -> np.ndarray:
    """
    Compute 7 Hu invariant moments from a grayscale image.
    Returns a 1D array of length 7.
    """
    img = gray.astype(np.float32)
    m = cv2.moments(img)
    hu = cv2.HuMoments(m).flatten()
    return hu


def _binary_stats(bin_img: np.ndarray) -> Dict[str, float]:
    """
    Simple stats from a binary/clean view:
    - stroke_density: fraction of foreground pixels
    - bbox_area_ratio: area of tight bounding box / total area
    - bbox_aspect_ratio: w/h of tight bounding box (0 if no foreground)
    """
    h, w = bin_img.shape
    total_pixels = float(h * w)
    fg = bin_img > 0

    fg_count = float(np.count_nonzero(fg))
    stroke_density = fg_count / total_pixels if total_pixels > 0 else 0.0

    if fg_count == 0:
        return {
            "stroke_density": 0.0,
            "bbox_area_ratio": 0.0,
            "bbox_aspect_ratio": 0.0,
        }

    ys, xs = np.where(fg)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bbox_w = float(x2 - x1 + 1)
    bbox_h = float(y2 - y1 + 1)
    bbox_area = bbox_w * bbox_h

    bbox_area_ratio = bbox_area / total_pixels if total_pixels > 0 else 0.0
    bbox_aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0.0

    return {
        "stroke_density": stroke_density,
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_aspect_ratio": bbox_aspect_ratio,
    }


def _skeleton_stats(skel: np.ndarray) -> Dict[str, float]:
    """
    Simple skeleton-based stats:
    - skeleton_pixel_fraction: fraction of pixels that are skeleton
    - endpoint_count: # pixels with exactly 1 foreground neighbor
    - junction_count: # pixels with >= 3 foreground neighbors
    """
    h, w = skel.shape
    total_pixels = float(h * w)
    fg = skel > 0
    fg_count = int(np.count_nonzero(fg))

    if fg_count == 0:
        return {
            "skeleton_pixel_fraction": 0.0,
            "endpoint_count": 0.0,
            "junction_count": 0.0,
        }

    padded = np.pad(fg, pad_width=1, mode="constant", constant_values=False)
    endpoint_count = 0
    junction_count = 0

    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if not padded[y, x]:
                continue
            nbhd = padded[y - 1 : y + 2, x - 1 : x + 2]
            neighbors = int(np.count_nonzero(nbhd) - 1)
            if neighbors == 1:
                endpoint_count += 1
            elif neighbors >= 3:
                junction_count += 1

    skeleton_pixel_fraction = fg_count / total_pixels if total_pixels > 0 else 0.0

    return {
        "skeleton_pixel_fraction": skeleton_pixel_fraction,
        "endpoint_count": float(endpoint_count),
        "junction_count": float(junction_count),
    }


def extract_features_simple(views: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simple feature set A.

    Inputs:
      views: dict with keys at least {"gray", "clean", "skeleton"}
             each is a 2D uint8 array (size x size)

    Returns:
      feature_vector: 1D numpy array of floats
      feature_dict:   mapping name -> value (for CSV column names)
    """
    gray = views["gray"]
    clean = views["clean"]
    skeleton = views["skeleton"]

    # Hu moments (7)
    hu = _hu_moments_from_gray(gray)
    hu_dict = {f"hu_{i+1}": float(v) for i, v in enumerate(hu)}

    # Binary stats (3)
    bin_stats = _binary_stats(clean)

    # Skeleton stats (3)
    skel_stats = _skeleton_stats(skeleton)

    feat_dict: Dict[str, float] = {}
    feat_dict.update(hu_dict)
    feat_dict.update(bin_stats)
    feat_dict.update(skel_stats)

    feat_vec = np.array(list(feat_dict.values()), dtype=np.float32)
    return feat_vec, feat_dict


# ============= RICH FEATURE SET (B) =============

def _intensity_stats(img: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Basic intensity / contrast stats:
    - mean, std, p10, p90, dynamic_range (p90 - p10)
    """
    arr = img.astype(np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))
    dyn = p90 - p10

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_p10": p10,
        f"{prefix}_p90": p90,
        f"{prefix}_range_90_10": dyn,
    }


def _binary_stats_rich(bin_img: np.ndarray) -> Dict[str, float]:
    """
    Rich stats from binary/clean view:
    - stroke_density
    - bbox_area_ratio
    - bbox_aspect_ratio
    - fg_area (number of foreground pixels)
    - perimeter_total (sum of contour perimeters)
    - num_components (number of connected blobs)
    - compactness (perimeter^2 / (4*pi*area)) for the largest blob
    """
    h, w = bin_img.shape
    total_pixels = float(h * w)
    fg = bin_img > 0
    fg_count = float(np.count_nonzero(fg))

    stroke_density = fg_count / total_pixels if total_pixels > 0 else 0.0

    if fg_count == 0:
        return {
            "stroke_density": 0.0,
            "bbox_area_ratio": 0.0,
            "bbox_aspect_ratio": 0.0,
            "fg_area": 0.0,
            "perimeter_total": 0.0,
            "num_components": 0.0,
            "compactness": 0.0,
        }

    ys, xs = np.where(fg)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bbox_w = float(x2 - x1 + 1)
    bbox_h = float(y2 - y1 + 1)
    bbox_area = bbox_w * bbox_h

    bbox_area_ratio = bbox_area / total_pixels if total_pixels > 0 else 0.0
    bbox_aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0.0

    # Contours for perimeter / compactness
    bin_u8 = (fg.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(bin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    num_components = float(len(contours))

    perimeter_total = 0.0
    largest_area = 0.0
    largest_perim = 0.0

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        perim = float(cv2.arcLength(cnt, closed=True))
        perimeter_total += perim
        if area > largest_area:
            largest_area = area
            largest_perim = perim

    if largest_area > 0:
        compactness = (largest_perim ** 2) / (4.0 * np.pi * largest_area)
    else:
        compactness = 0.0

    return {
        "stroke_density": stroke_density,
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "fg_area": fg_count,
        "perimeter_total": perimeter_total,
        "num_components": num_components,
        "compactness": compactness,
    }


def _skeleton_stats_rich(skel: np.ndarray) -> Dict[str, float]:
    """
    Rich skeleton stats:
    - skeleton_pixel_fraction
    - endpoint_count
    - junction_count
    - skeleton_length (number of skeleton pixels)
    - endpoint_density (endpoints / skeleton pixels)
    - junction_density (junctions / skeleton pixels)
    """
    h, w = skel.shape
    total_pixels = float(h * w)
    fg = skel > 0
    fg_count = int(np.count_nonzero(fg))

    if fg_count == 0:
        return {
            "skeleton_pixel_fraction": 0.0,
            "endpoint_count": 0.0,
            "junction_count": 0.0,
            "skeleton_length": 0.0,
            "endpoint_density": 0.0,
            "junction_density": 0.0,
        }

    padded = np.pad(fg, pad_width=1, mode="constant", constant_values=False)
    endpoint_count = 0
    junction_count = 0

    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if not padded[y, x]:
                continue
            nbhd = padded[y - 1 : y + 2, x - 1 : x + 2]
            neighbors = int(np.count_nonzero(nbhd) - 1)
            if neighbors == 1:
                endpoint_count += 1
            elif neighbors >= 3:
                junction_count += 1

    skeleton_pixel_fraction = fg_count / total_pixels if total_pixels > 0 else 0.0
    endpoint_density = endpoint_count / fg_count if fg_count > 0 else 0.0
    junction_density = junction_count / fg_count if fg_count > 0 else 0.0

    return {
        "skeleton_pixel_fraction": skeleton_pixel_fraction,
        "endpoint_count": float(endpoint_count),
        "junction_count": float(junction_count),
        "skeleton_length": float(fg_count),
        "endpoint_density": float(endpoint_density),
        "junction_density": float(junction_density),
    }


def _edge_stats(edge_img: np.ndarray) -> Dict[str, float]:
    """
    Edge-based stats from 'edge' view:
    - edge_pixel_fraction
    - num_edge_components (contours)
    - edge_total_length (sum of contour lengths)
    - edge_mean_length (average contour length)
    """
    h, w = edge_img.shape
    total_pixels = float(h * w)
    fg = edge_img > 0
    edge_count = float(np.count_nonzero(fg))

    edge_pixel_fraction = edge_count / total_pixels if total_pixels > 0 else 0.0

    if edge_count == 0:
        return {
            "edge_pixel_fraction": 0.0,
            "num_edge_components": 0.0,
            "edge_total_length": 0.0,
            "edge_mean_length": 0.0,
        }

    bin_u8 = (fg.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(bin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    num_components = float(len(contours))

    total_length = 0.0
    for cnt in contours:
        total_length += float(cv2.arcLength(cnt, closed=False))

    mean_length = total_length / num_components if num_components > 0 else 0.0

    return {
        "edge_pixel_fraction": edge_pixel_fraction,
        "num_edge_components": num_components,
        "edge_total_length": total_length,
        "edge_mean_length": mean_length,
    }


def extract_features_rich(views: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Rich feature set B.

    Inputs:
      views: dict with keys:
        "gray", "clahe", "clean", "skeleton", "edge"

    Returns:
      feature_vector: 1D numpy array of floats
      feature_dict:   mapping name -> value
    """
    gray = views["gray"]
    clahe = views["clahe"]
    clean = views["clean"]
    skeleton = views["skeleton"]
    edge = views["edge"]

    feat_dict: Dict[str, float] = {}

    # 1) Hu moments from gray and CLAHE
    hu_gray = _hu_moments_from_gray(gray)
    hu_clahe = _hu_moments_from_gray(clahe)

    for i, v in enumerate(hu_gray):
        feat_dict[f"hu_gray_{i+1}"] = float(v)
    for i, v in enumerate(hu_clahe):
        feat_dict[f"hu_clahe_{i+1}"] = float(v)

    # 2) Intensity / contrast stats
    feat_dict.update(_intensity_stats(gray, prefix="gray"))
    feat_dict.update(_intensity_stats(clahe, prefix="clahe"))

    # 3) Binary / clean stats
    feat_dict.update(_binary_stats_rich(clean))

    # 4) Skeleton stats
    feat_dict.update(_skeleton_stats_rich(skeleton))

    # 5) Edge stats
    feat_dict.update(_edge_stats(edge))

    feat_vec = np.array(list(feat_dict.values()), dtype=np.float32)
    return feat_vec, feat_dict
