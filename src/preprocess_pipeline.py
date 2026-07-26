import sys
from pathlib import Path

# Ensure the project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DatasetConfig
from src.logger import get_logger
from src.validation import run_prepipeline_validation
from src.dataset_loader import get_dataloaders
from src.statistics import generate_split_statistics_report
from src.visualization import visualize_preprocessed_sample

logger = get_logger("DocForge.Pipeline")

def run_preprocessing_pipeline(sample_limit: int = 500) -> None:
    """Execute the Phase 1 Preprocessing and Data Preparation pipeline.

    Args:
        sample_limit: Maximum number of samples to process from the database
            to constrain resource usage and runtime.
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING & DATA PREPARATION PIPELINE (PHASE 1)")
    logger.info("=" * 60)

    # 1. Load configurations
    config = DatasetConfig()
    config.ensure_output_dirs()

    # 2. Pre-pipeline dataset validation
    # Verifies folder existence, readability, dimensions, and mask binarity.
    logger.info("Step 1: Running dataset structure and sample validation...")
    validation_passed = run_prepipeline_validation(config, sample_limit=min(100, sample_limit))
    if not validation_passed:
        logger.error("Pre-pipeline validation checks failed. Aborting preprocessing pipeline.")
        sys.exit(1)
    logger.info("Validation completed. Pipeline is READY.")

    # 3. Create PyTorch DataLoaders
    logger.info("Step 2: Splitting dataset and constructing PyTorch DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config.training_set,
        config=config,
        sample_limit=sample_limit
    )

    # 4. Generate split statistics
    logger.info("Step 3: Compiling dataset split metrics and exporting statistics reports...")
    # Extract subsets from dataloaders
    train_subset = train_loader.dataset
    val_subset = val_loader.dataset
    test_subset = test_loader.dataset
    
    stats_report = generate_split_statistics_report(
        train_subset,
        val_subset,
        test_subset,
        config.statistics_dir
    )
    
    logger.info("Pipeline statistics successfully generated.")
    logger.info(f"Total processed samples: {stats_report['total_images']}")
    logger.info(f"Authentic samples: {stats_report['authentic_count']} | Tampered samples: {stats_report['tampered_count']}")
    logger.info(f"Extracted bounding boxes: {stats_report['total_bounding_boxes']}")

    # 5. Visualize and save preprocessed samples
    logger.info("Step 4: Generating and saving visualization overlays for preprocessed samples...")
    # Visualize 3 samples from the augmented training subset
    # Make sure database is open
    train_subset.dataset.open_database()
    
    visualized = 0
    for i in range(len(train_subset)):
        if visualized >= 3:
            break
        try:
            sample = train_subset[i]
            # Save overlays of preprocessed images
            save_path = config.overlays_dir / f"preprocessed_train_sample_{sample['metadata']['index']}_overlay.png"
            visualize_preprocessed_sample(
                sample,
                alpha=0.5,
                color=(255, 0, 0),
                save_path=save_path,
                show=False
            )
            visualized += 1
        except Exception as e:
            logger.error(f"Error visualizing preprocessed sample at index {i}: {e}")

    # Close dataset database connections
    train_subset.dataset.close_database()
    logger.info("Step 5: Database connections closed safely.")
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY.")
    logger.info("=" * 60)

if __name__ == "__main__":
    limit = 500
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass
    run_preprocessing_pipeline(sample_limit=limit)
