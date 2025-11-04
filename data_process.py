import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # Fallback if tqdm is not available

try:
    from skimage.morphology import thin as sk_thin
except Exception:
    sk_thin = None

try:
    from scipy.io import loadmat
except Exception as e:
    raise RuntimeError(
        "scipy is required to read BSDS500 ground truth .mat files. Please install it: pip install scipy"
    ) from e


def find_splits(dataset_root: Path) -> Dict[str, Dict[str, Path]]:
    """Locate image and groundTruth split directories under BSDS500/data.

    Returns a dict mapping split -> {"images": Path, "gt": Path}
    """
    base = dataset_root
    img_dir = base / "images"
    gt_dir = base / "groundTruth"
    splits = {}
    for split in ("train", "val", "test"):
        s_img = img_dir / split
        s_gt = gt_dir / split
        if s_img.exists() and s_gt.exists():
            splits[split] = {"images": s_img, "gt": s_gt}
        else:
            logging.warning("Split '%s' directories not found: %s or %s", split, s_img, s_gt)
    if not splits:
        raise FileNotFoundError(
            f"No valid splits found under {img_dir} and {gt_dir}. Ensure BSDS500/data layout is correct."
        )
    return splits


def list_image_files(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def load_gt_boundaries(gt_mat_path: Path) -> List[np.ndarray]:
    """Load list of boundary maps from BSDS500 groundTruth .mat file.

    Each annotator contributes a binary boundary map. Shapes match image size.
    """
    mat = loadmat(str(gt_mat_path))
    if "groundTruth" not in mat:
        raise ValueError(f"'groundTruth' key not found in {gt_mat_path}")
    gt = mat["groundTruth"]
    # gt is typically (1, N) object array, each item has fields 'Boundaries', 'Segmentation'
    boundaries = []
    # Normalize to iterable of items
    # Different scipy versions load struct arrays differently; handle common cases
    try:
        it = gt[0]
    except Exception:
        it = gt
    for item in it:
        try:
            b = item["Boundaries"][0, 0]
        except Exception:
            # Alternative access pattern
            # item may be a numpy.void with named fields
            b = item["Boundaries"]
            if isinstance(b, np.ndarray) and b.size == 1 and b.dtype == object:
                b = b.item()
        b = np.array(b, dtype=np.float32)
        boundaries.append(b)
    if not boundaries:
        raise ValueError(f"No boundaries found in {gt_mat_path}")
    return boundaries


def combine_boundaries(boundaries: List[np.ndarray], mode: str = "mean") -> np.ndarray:
    """Combine multiple annotator boundary maps into a single probability map.

    mode: 'mean' (default), 'max', 'or'
    Returns float32 array in [0, 1].
    """
    stack = np.stack(boundaries, axis=0).astype(np.float32)
    if mode == "mean":
        prob = stack.mean(axis=0)
    elif mode == "max":
        prob = stack.max(axis=0)
    elif mode == "or":
        prob = (stack.sum(axis=0) > 0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported combine mode: {mode}")
    prob = np.clip(prob, 0.0, 1.0)
    return prob


def apply_thinning(binary_map: np.ndarray) -> np.ndarray:
    if sk_thin is None:
        # Fallback: return input without thinning
        return binary_map
    # skimage.thin expects boolean
    thinned = sk_thin(binary_map.astype(bool))
    return thinned.astype(np.uint8)


def save_image(img_path: Path, out_path: Path, resize: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    img = Image.open(img_path).convert("RGB")
    if resize is not None:
        img = img.resize(resize, Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return img.size  # (width, height)


def save_edge_map(
    prob_map: np.ndarray,
    out_path: Path,
    edge_threshold: Optional[float] = None,
    thin: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if edge_threshold is not None:
        binary = (prob_map >= float(edge_threshold)).astype(np.uint8)
        if thin:
            binary = apply_thinning(binary)
        arr = (binary * 255).astype(np.uint8)
    else:
        arr = (prob_map * 255.0).round().astype(np.uint8)
    im = Image.fromarray(arr, mode="L")
    im.save(out_path)


def process_split(
    split_name: str,
    split_dirs: Dict[str, Path],
    out_root: Path,
    combine_mode: str,
    edge_threshold: Optional[float],
    thin: bool,
    resize: Optional[Tuple[int, int]],
) -> Dict:
    images_dir = split_dirs["images"]
    gt_dir = split_dirs["gt"]
    out_img_dir = out_root / "images" / split_name
    out_edge_dir = out_root / "edges" / split_name
    out_list_path = out_root / "splits" / f"{split_name}.txt"
    out_list_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    image_files = list_image_files(images_dir)
    for img_path in tqdm(image_files, desc=f"{split_name}"):
        stem = img_path.stem
        gt_path = gt_dir / f"{stem}.mat"
        if not gt_path.exists():
            logging.warning("Missing ground truth for %s: %s", img_path.name, gt_path)
            continue
        # Load and combine GT boundaries
        boundaries = load_gt_boundaries(gt_path)
        prob = combine_boundaries(boundaries, mode=combine_mode)

        # Optionally resize both image and prob map
        if resize is not None:
            # PIL uses (W, H); numpy uses (H, W)
            pil_prob = Image.fromarray((prob * 255).astype(np.uint8), mode="L").resize(resize, Image.NEAREST)
            prob = np.array(pil_prob, dtype=np.uint8).astype(np.float32) / 255.0

        out_img_path = out_img_dir / f"{stem}.png"
        out_edge_path = out_edge_dir / f"{stem}.png"

        width, height = save_image(img_path, out_img_path, resize=resize)
        save_edge_map(prob, out_edge_path, edge_threshold=edge_threshold, thin=thin)

        records.append(
            {
                "id": stem,
                "image": str(out_img_path),
                "edge": str(out_edge_path),
                "width": width,
                "height": height,
            }
        )

    # Write list file: image_path edge_path per line
    with open(out_list_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r['image']}\t{r['edge']}\n")

    return {
        "split": split_name,
        "count": len(records),
        "list_file": str(out_list_path),
        "images_dir": str(out_img_dir),
        "edges_dir": str(out_edge_dir),
    }


def parse_resize(resize_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not resize_str:
        return None
    try:
        w, h = resize_str.split(",")
        return int(w), int(h)
    except Exception:
        raise ValueError("--resize must be formatted as 'width,height', e.g., 320,320")


def main():
    parser = argparse.ArgumentParser(
        description="Process BSDS500 dataset to images and edge maps for edge detection training."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(Path(r"c:/Users/Administrator/Desktop/CV_Paper_2/BSDS500-master/BSDS500/data")),
        help="Path to BSDS500/data directory containing images/ and groundTruth/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(r"c:/Users/Administrator/Desktop/CV_Paper_2/BSDS500-master/processed")),
        help="Output directory for processed images, edge maps, and split files.",
    )
    parser.add_argument(
        "--combine-mode",
        type=str,
        default="mean",
        choices=["mean", "max", "or"],
        help="Strategy to combine multiple annotator boundary maps.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=None,
        help="Optional threshold in [0,1] to binarize edge probability maps.",
    )
    parser.add_argument(
        "--thin",
        action="store_true",
        help="Apply skeletonization/thinning to binary edge maps (requires scikit-image).",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Optional resize 'width,height' for both image and label (e.g., 320,320).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    resize = parse_resize(args.resize)

    logging.info("Dataset root: %s", dataset_root)
    logging.info("Output dir: %s", output_dir)
    logging.info("Combine mode: %s", args.combine_mode)
    logging.info("Edge threshold: %s", args.edge_threshold)
    logging.info("Thin: %s", args.thin)
    logging.info("Resize: %s", resize)

    splits = find_splits(dataset_root)
    meta = {"dataset_root": str(dataset_root), "output_dir": str(output_dir), "splits": []}

    for split_name, split_dirs in splits.items():
        info = process_split(
            split_name,
            split_dirs,
            output_dir,
            combine_mode=args.combine_mode,
            edge_threshold=args.edge_threshold,
            thin=args.thin,
            resize=resize,
        )
        meta["splits"].append(info)

    # Save metadata
    meta_path = output_dir / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logging.info("Wrote metadata: %s", meta_path)

    logging.info("Done. Processed splits: %s", ", ".join(sorted(splits.keys())))


if __name__ == "__main__":
    main()