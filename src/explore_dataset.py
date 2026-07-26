import sys
from pathlib import Path
from typing import Optional

# Ensure the project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DatasetConfig
from src.dataset import DocTamperDataset
from src.logger import get_logger
from src.statistics import compute_dataset_statistics, format_statistics_table, plot_tampered_percentage_histogram
from src.validation import verify_dataset, save_validation_report
from src.visualization import show_random_samples

logger = get_logger("DocForge.Explorer")

def run_explorer(sample_limit: int = 1000) -> None:
    """Run Phase 0 exploration on all available DocTamper datasets.

    Args:
        sample_limit: Limit of samples to process for statistics & validation
            to run within a reasonable time.
    """
    logger.info("Initializing Dataset Explorer...")
    
    # Initialize configuration and ensure output directories exist
    config = DatasetConfig()
    config.ensure_output_dirs()
    
    logger.info(f"Dataset configurations: {config}")

    subsets = ["TrainingSet", "TestingSet", "SCD", "FCD"]
    
    for subset in subsets:
        try:
            logger.info("=" * 60)
            logger.info(f"Processing Subset: {subset}")
            logger.info("=" * 60)
            
            # Resolve path for the subset
            subset_path = config.get_subset_path(subset)
            if not (subset_path / "data.mdb").exists():
                logger.warning(f"Subset {subset} not found at {subset_path}. Skipping.")
                continue

            # Load dataset
            dataset = DocTamperDataset(subset_path, config=config)
            
            # 1. Compute and print statistics
            stats = compute_dataset_statistics(dataset, sample_limit=sample_limit, log_interval=500)
            
            if stats:
                table_str = format_statistics_table(stats)
                print(table_str)
                print("\n")
                
                # Save statistics report
                stats_report_path = config.statistics_dir / f"{subset}_statistics_report.txt"
                with open(stats_report_path, "w", encoding="utf-8") as f:
                    f.write(table_str)
                logger.info(f"Saved text statistics report to {stats_report_path}")
                
                # Generate and save histogram
                histogram_path = config.statistics_dir / f"{subset}_tampered_percentage_histogram.png"
                plot_tampered_percentage_histogram(stats, histogram_path)
            
            # 2. Run integrity checks and save report
            logger.info(f"Running integrity verification for {subset}...")
            report = verify_dataset(dataset, sample_limit=sample_limit, log_interval=500)
            
            report_path = config.reports_dir / f"{subset}_validation_report.md"
            save_validation_report(report, report_path)
            logger.info(f"Completed verification. Passed: {report['passed']}.")
            
            # 3. Visualize and save random samples
            logger.info(f"Generating and saving random overlays for {subset}...")
            # We save 3 random samples from each dataset
            show_random_samples(
                dataset, 
                num_samples=3, 
                alpha=0.5, 
                color=(255, 0, 0), 
                save_dir=config.overlays_dir, 
                show=False
            )
            
            # Close dataset database to free up memory/file handles
            dataset.close_database()
            
        except Exception as e:
            logger.error(f"Error exploring subset {subset}: {e}", exc_info=True)

if __name__ == "__main__":
    # Allow passing sample limit from CLI
    limit = 1000
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass
    run_explorer(sample_limit=limit)
