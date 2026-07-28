import random
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

from src.dataset import DocTamperDataset
from src.logger import get_logger

logger = get_logger("DocForge.Visualization")

def overlay_mask(
    image: Image.Image,
    mask: Image.Image,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> Image.Image:
    """Blend a binary mask onto an image, highlighting the positive pixels in a custom color.

    Args:
        image: The original PIL Image.
        mask: The binary mask PIL Image (values of 255 represent tampered regions).
        alpha: Transparency value for the overlay (0.0 to 1.0).
        color: RGB tuple for the highlight color (defaults to Red: (255, 0, 0)).

    Returns:
        Image.Image: The combined image with overlay.
    """
    # Convert original to RGB
    original_rgb = image.convert("RGB")
    
    # Handle size mismatch if any
    if image.size != mask.size:
        logger.warning(
            f"Dimension mismatch between image {image.size} and mask {mask.size}. "
            f"Resizing mask to match image size."
        )
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    # Convert mask to grayscale (L)
    mask_l = mask.convert("L")
    
    # Scale mask values by alpha to set overlay opacity
    # 255 in mask (tampered) -> int(255 * alpha) opacity
    # 0 in mask (authentic) -> 0 opacity
    mask_alpha = mask_l.point(lambda p: int(p * alpha))
    
    # Create solid color overlay image
    color_overlay = Image.new("RGB", image.size, color)
    
    # Composite the color overlay onto the original image using the alpha mask
    blended = Image.composite(color_overlay, original_rgb, mask_alpha)
    return blended

def show_image(
    image: Image.Image,
    title: str = "Original Image",
    ax: Optional[plt.Axes] = None
) -> None:
    """Display a single PIL Image using Matplotlib.

    Args:
        image: The PIL Image to display.
        title: Title of the plot.
        ax: Optional Matplotlib axis to plot onto.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    
    ax.imshow(image)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

def show_mask(
    mask: Image.Image,
    title: str = "Binary Mask",
    ax: Optional[plt.Axes] = None
) -> None:
    """Display a binary mask PIL Image in grayscale using Matplotlib.

    Args:
        mask: The binary mask PIL Image to display.
        title: Title of the plot.
        ax: Optional Matplotlib axis to plot onto.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    
    ax.imshow(mask, cmap="gray")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

def show_overlay(
    image: Image.Image,
    mask: Image.Image,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
    title: str = "Overlay Highlight",
    ax: Optional[plt.Axes] = None
) -> None:
    """Display an overlay of a mask on an image.

    Args:
        image: The original PIL Image.
        mask: The binary mask PIL Image.
        alpha: Transparency value for the overlay.
        color: Highlight RGB color.
        title: Title of the plot.
        ax: Optional Matplotlib axis to plot onto.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        
    blended = overlay_mask(image, mask, alpha=alpha, color=color)
    ax.imshow(blended)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

def visualize_sample(
    image: Image.Image,
    mask: Image.Image,
    index: int,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """Plot a single sample containing Original, Mask, and Overlay in a 1x3 row.

    Args:
        image: Original image.
        mask: Binary mask.
        index: Sample index.
        alpha: Overlay opacity.
        color: Overlay RGB color.
        save_path: Path to save the figure if specified.
        show: Whether to display the plot (plt.show()).

    Returns:
        plt.Figure: The Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    show_image(image, title=f"Sample {index} - Original", ax=axes[0])
    show_mask(mask, title=f"Sample {index} - Mask", ax=axes[1])
    show_overlay(image, mask, alpha=alpha, color=color, title=f"Sample {index} - Overlay", ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved sample visualization to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def show_random_samples(
    dataset: DocTamperDataset,
    num_samples: int = 5,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True
) -> List[int]:
    """Select random samples from a dataset, plot and optionally save them.

    Args:
        dataset: The DocTamperDataset instance.
        num_samples: Number of random samples to display.
        alpha: Overlay opacity.
        color: Overlay RGB color.
        save_dir: Directory where the plotted figures should be saved.
        show: Whether to display the plots.

    Returns:
        List[int]: The list of randomly selected indices.
    """
    total_samples = len(dataset)
    if total_samples == 0:
        logger.warning("Empty dataset. Cannot visualize samples.")
        return []

    sampled_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    logger.info(f"Visualizing {len(sampled_indices)} random samples: {sampled_indices}")
    
    for idx in sampled_indices:
        try:
            image, mask = dataset.read_sample(idx)
            
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f"sample_{idx}_visualization.png"
                
            visualize_sample(
                image, 
                mask, 
                index=idx, 
                alpha=alpha, 
                color=color, 
                save_path=save_path, 
                show=show
            )
        except Exception as e:
            logger.error(f"Failed to visualize sample at index {idx}: {e}")
            
    return sampled_indices

def visualize_preprocessed_sample(
    sample: Dict[str, Any],
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """Visualize a preprocessed unified sample dictionary.

    Renders original image, mask, and overlay, drawing bounding boxes
    and metadata labels (sample ID, tampering label, forgery type, and prompt).

    Args:
        sample: Unified sample dict from DocTamperTorchDataset.
        alpha: Overlay blend opacity.
        color: RGB tuple highlight color.
        save_path: Path to save the Matplotlib figure.
        show: Whether to display the plot.

    Returns:
        plt.Figure: The generated Matplotlib figure.
    """
    # pyrefly: ignore [missing-import]
    import matplotlib.patches as patches
    
    img = sample["image"]
    mask = sample["mask"]
    bboxes = sample["bbox"]  # Absolute bboxes
    
    # Calculate scale factor between raw and processed sizes to scale bboxes
    orig_w, orig_h = sample["width"], sample["height"]
    proc_w, proc_h = img.size
    
    scale = min(proc_w / orig_w, proc_h / orig_h)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Generate overlay image
    blend = overlay_mask(img, mask, alpha=alpha, color=color)
    
    # Display images
    axes[0].imshow(img)
    axes[0].set_title("Original (Preprocessed)", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Tampering Mask (Preprocessed)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(blend)
    axes[2].set_title("Overlay Blend (Red)", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    
    # Draw scaled bounding boxes on all three subplots
    for ax in axes:
        for box in bboxes:
            xmin, ymin, xmax, ymax = box
            
            xmin_proc = int(round(xmin * scale))
            ymin_proc = int(round(ymin * scale))
            xmax_proc = int(round(xmax * scale))
            ymax_proc = int(round(ymax * scale))
            
            width = xmax_proc - xmin_proc
            height = ymax_proc - ymin_proc
            
            rect = patches.Rectangle(
                (xmin_proc, ymin_proc), width, height,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)

    # Compile overall figure title
    status_label = "TAMPERED" if sample["tampering_label"] == 1 else "AUTHENTIC"
    forgery_type = sample["forgery_type"]
    prompt = sample["prompt"]
    sample_id = sample["sample_id"]
    
    title_text = (
        f"Sample ID: {sample_id} | Status: {status_label} | Type: {forgery_type}\n"
        f"VLM Prompt: \"{prompt}\""
    )
    fig.suptitle(title_text, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved preprocessed sample visualization to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig
