import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt

from src.dataset import DocTamperDataset
from src.logger import get_logger

logger = get_logger("DocForge.Statistics")

def analyze_mask(mask: Image.Image) -> Dict[str, Any]:
    """Analyze pixel statistics for a single binary mask.

    Args:
        mask: The binary tampering mask (Pillow Image).
            Authentic pixels should be 0, tampered should be 255.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'total_pixels': Total number of pixels
            - 'tampered_pixels': Number of tampered pixels (value > 127)
            - 'authentic_pixels': Number of authentic pixels (value <= 127)
            - 'tampered_percentage': Percentage of tampered pixels (0.0 to 100.0)
    """
    # Convert mask to numpy array for fast pixel counting
    mask_arr = np.array(mask.convert("L"))
    
    total_pixels = mask_arr.size
    # Count pixels that are considered tampered (any value > 0)
    tampered_pixels = int(np.sum(mask_arr > 0))
    authentic_pixels = total_pixels - tampered_pixels
    tampered_percentage = (tampered_pixels / total_pixels) * 100.0
    
    return {
        "total_pixels": total_pixels,
        "tampered_pixels": tampered_pixels,
        "authentic_pixels": authentic_pixels,
        "tampered_percentage": tampered_percentage
    }

def compute_dataset_statistics(
    dataset: DocTamperDataset,
    sample_limit: Optional[int] = None,
    log_interval: int = 1000
) -> Dict[str, Any]:
    """Compute dataset-wide statistics.

    Iterates over the dataset to compute sample counts, resolution consistency,
    and tampered area metrics.

    Args:
        dataset: The DocTamperDataset instance.
        sample_limit: Maximum number of samples to process (useful for large datasets).
            If None, processes the entire dataset.
        log_interval: Number of samples between logging progress updates.

    Returns:
        Dict[str, Any]: Dictionary with dataset metrics.
    """
    total_available = len(dataset)
    num_to_process = total_available if sample_limit is None else min(sample_limit, total_available)
    
    if num_to_process == 0:
        logger.warning("No samples to compute statistics for.")
        return {}

    logger.info(f"Computing statistics for {num_to_process} samples (out of {total_available} total)...")

    first_idx = 0
    last_idx = num_to_process - 1

    image_sizes = set()
    mask_sizes = set()
    
    tampered_percentages = []
    tampered_pixel_counts = []
    
    total_authentic_pixels = 0
    total_tampered_pixels = 0
    
    # Ensure database is open
    dataset.open_database()
    
    for i in range(num_to_process):
        if (i + 1) % log_interval == 0 or i == num_to_process - 1:
            logger.info(f"Processing sample {i+1}/{num_to_process}...")
            
        try:
            image, mask = dataset.read_sample(i)
            
            # Record sizes
            image_sizes.add(image.size)
            mask_sizes.add(mask.size)
            
            # Analyze mask
            stats = analyze_mask(mask)
            
            tampered_percentages.append(stats["tampered_percentage"])
            tampered_pixel_counts.append(stats["tampered_pixels"])
            
            total_authentic_pixels += stats["authentic_pixels"]
            total_tampered_pixels += stats["tampered_pixels"]
            
        except Exception as e:
            logger.error(f"Error processing sample {i} during statistics: {e}")

    # Compute aggregation
    if not tampered_percentages:
        logger.error("No samples were successfully analyzed.")
        return {}
        
    avg_tampered_percentage = float(np.mean(tampered_percentages))
    min_tampered_percentage = float(np.min(tampered_percentages))
    max_tampered_percentage = float(np.max(tampered_percentages))

    avg_tampered_pixels = float(np.mean(tampered_pixel_counts))
    min_tampered_pixels = int(np.min(tampered_pixel_counts))
    max_tampered_pixels = int(np.max(tampered_pixel_counts))
    
    # We can check a few random ones at the end of the dataset to verify sizes if limit was set
    # but for simplicity, we report based on processed samples.
    
    results = {
        "dataset_name": dataset.db_path.name,
        "total_samples": total_available,
        "processed_samples": num_to_process,
        "first_index": first_idx,
        "last_index": total_available - 1,
        "image_resolutions": list(image_sizes),
        "mask_resolutions": list(mask_sizes),
        "avg_tampered_percentage": avg_tampered_percentage,
        "min_tampered_percentage": min_tampered_percentage,
        "max_tampered_percentage": max_tampered_percentage,
        "avg_tampered_pixels": avg_tampered_pixels,
        "min_tampered_pixels": min_tampered_pixels,
        "max_tampered_pixels": max_tampered_pixels,
        "total_authentic_pixels": total_authentic_pixels,
        "total_tampered_pixels": total_tampered_pixels,
        "tampered_percentages": tampered_percentages  # Keep raw values for plotting
    }
    
    logger.info(f"Finished computing statistics for {dataset.db_path.name}.")
    return results

def plot_tampered_percentage_histogram(
    stats: Dict[str, Any],
    save_path: Union[str, Path]
) -> None:
    """Plot and save a histogram of tampered pixel percentages.

    Args:
        stats: Statistics dictionary returned by compute_dataset_statistics.
        save_path: Path to save the generated plot.
    """
    percentages = stats.get("tampered_percentages", [])
    if not percentages:
        logger.warning("No tampered percentage data to plot.")
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot histogram
    ax.hist(percentages, bins=50, color="crimson", edgecolor="black", alpha=0.7)
    
    # Add title and labels
    dataset_name = stats.get("dataset_name", "DocTamper")
    ax.set_title(f"Distribution of Tampered Area ({dataset_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tampered Pixel Percentage (%)", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    
    # Add summary box
    summary_text = (
        f"Avg: {stats['avg_tampered_percentage']:.2f}%\n"
        f"Min: {stats['min_tampered_percentage']:.2f}%\n"
        f"Max: {stats['max_tampered_percentage']:.2f}%"
    )
    ax.text(
        0.95, 0.95, summary_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved tampered percentage histogram to {save_path}")

def format_statistics_table(stats: Dict[str, Any]) -> str:
    """Format a statistics dictionary into a clean, readable ASCII table.

    Args:
        stats: Statistics dictionary.

    Returns:
        str: Formatted ASCII table.
    """
    if not stats:
        return "No statistics data available."

    img_res_str = ", ".join(f"{w}x{h}" for w, h in stats["image_resolutions"])
    mask_res_str = ", ".join(f"{w}x{h}" for w, h in stats["mask_resolutions"])

    lines = [
        "+" + "-"*50 + "+",
        f"| {'Dataset Statistics Report':^48} |",
        "+" + "-"*50 + "+",
        f"| {'Dataset Name':<25} | {stats['dataset_name']:<20.20} |",
        f"| {'Total Samples':<25} | {stats['total_samples']:<20} |",
        f"| {'Processed Samples':<25} | {stats['processed_samples']:<20} |",
        f"| {'First Image Index':<25} | {stats['first_index']:<20} |",
        f"| {'Last Image Index':<25} | {stats['last_index']:<20} |",
        f"| {'Image Resolution':<25} | {img_res_str:<20.20} |",
        f"| {'Mask Resolution':<25} | {mask_res_str:<20.20} |",
        "+" + "-"*50 + "+",
        f"| {'Avg Tampered Area (%)':<25} | {stats['avg_tampered_percentage']:<20.2f} |",
        f"| {'Min Tampered Area (%)':<25} | {stats['min_tampered_percentage']:<20.2f} |",
        f"| {'Max Tampered Area (%)':<25} | {stats['max_tampered_percentage']:<20.2f} |",
        "+" + "-"*50 + "+",
        f"| {'Avg Tampered Pixels':<25} | {stats['avg_tampered_pixels']:<20.1f} |",
        f"| {'Min Tampered Pixels':<25} | {stats['min_tampered_pixels']:<20} |",
        f"| {'Max Tampered Pixels':<25} | {stats['max_tampered_pixels']:<20} |",
        "+" + "-"*50 + "+",
        f"| {'Total Authentic Pixels':<25} | {stats['total_authentic_pixels']:<20} |",
        f"| {'Total Tampered Pixels':<25} | {stats['total_tampered_pixels']:<20} |",
        "+" + "-"*50 + "+"
    ]
    return "\n".join(lines)

def generate_split_statistics_report(
    train_subset: Any,
    val_subset: Any,
    test_subset: Any,
    save_dir: Union[str, Path]
) -> Dict[str, Any]:
    """Compile comprehensive pipeline statistics for train/val/test splits and export JSON + CSV.

    Args:
        train_subset: PyTorch train Subset dataset.
        val_subset: PyTorch val Subset dataset.
        test_subset: PyTorch test Subset dataset.
        save_dir: Folder to save JSON and CSV reports.

    Returns:
        Dict[str, Any]: Compiled statistics dictionary.
    """
    import json
    import csv
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_size = len(train_subset)
    val_size = len(val_subset)
    test_size = len(test_subset)
    total_images = train_size + val_size + test_size
    
    logger.info(f"Generating split statistics for {total_images} total samples...")
    
    authentic_count = 0
    tampered_count = 0
    
    forgery_types: Dict[str, int] = {}
    widths = []
    heights = []
    
    mask_avail_count = 0
    total_bboxes = 0
    bbox_counts_per_sample = []
    
    subsets_dict = {
        "train": train_subset,
        "val": val_subset,
        "test": test_subset
    }
    
    for split_name, subset in subsets_dict.items():
        logger.info(f"Profiling {split_name} split ({len(subset)} samples)...")
        # Ensure database is open
        base_ds = subset.dataset
        base_ds.open_database()
        
        for idx in range(len(subset)):
            sample = subset[idx]
            
            # 1. Authentic / Tampered
            is_tampered = sample["tampering_label"] == 1
            if is_tampered:
                tampered_count += 1
            else:
                authentic_count += 1
                
            # 2. Forgery type
            ftype = sample["forgery_type"]
            forgery_types[ftype] = forgery_types.get(ftype, 0) + 1
            
            # 3. Resolutions
            widths.append(sample["width"])
            heights.append(sample["height"])
            
            # 4. Mask availability
            if sample["mask"] is not None:
                mask_avail_count += 1
                
            # 5. Bounding box stats
            bboxes = sample["bbox"]
            num_boxes = len(bboxes)
            if num_boxes > 0:
                total_bboxes += num_boxes
                bbox_counts_per_sample.append(num_boxes)

    # Aggregating resolution metrics
    if widths:
        avg_w = float(np.mean(widths))
        avg_h = float(np.mean(heights))
        min_w, max_w = int(np.min(widths)), int(np.max(widths))
        min_h, max_h = int(np.min(heights)), int(np.max(heights))
    else:
        avg_w = avg_h = min_w = max_w = min_h = max_h = 0
        
    # Aggregating bbox metrics
    if bbox_counts_per_sample:
        avg_bboxes = float(np.mean(bbox_counts_per_sample))
        min_bboxes = int(np.min(bbox_counts_per_sample))
        max_bboxes = int(np.max(bbox_counts_per_sample))
    else:
        avg_bboxes = min_bboxes = max_bboxes = 0

    stats_report = {
        "total_images": total_images,
        "train_count": train_size,
        "val_count": val_size,
        "test_count": test_size,
        "authentic_count": authentic_count,
        "tampered_count": tampered_count,
        "forgery_type_distribution": forgery_types,
        "avg_width": avg_w,
        "avg_height": avg_h,
        "min_width": min_w,
        "max_width": max_w,
        "min_height": min_h,
        "max_height": max_h,
        "mask_availability_count": mask_avail_count,
        "annotation_availability_count": total_images,
        "total_bounding_boxes": total_bboxes,
        "avg_bboxes_per_tampered_sample": avg_bboxes,
        "min_bboxes_per_tampered_sample": min_bboxes,
        "max_bboxes_per_tampered_sample": max_bboxes
    }
    
    # Save as JSON
    json_path = save_dir / "pipeline_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=4)
    logger.info(f"Saved pipeline JSON statistics to {json_path}")
    
    # Save as CSV (flattened representation)
    csv_path = save_dir / "pipeline_statistics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        
        for k, v in stats_report.items():
            if k != "forgery_type_distribution":
                writer.writerow([k, str(v)])
                
        for ftype, count in forgery_types.items():
            writer.writerow([f"forgery_type_{ftype}_count", str(count)])
            
    logger.info(f"Saved pipeline CSV statistics to {csv_path}")
    return stats_report
