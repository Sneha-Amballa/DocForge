import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
from src.dataset import DocTamperDataset, DatasetError
from src.logger import get_logger

logger = get_logger("DocForge.Validation")

def verify_dataset(
    dataset: DocTamperDataset,
    sample_limit: Optional[int] = None,
    log_interval: int = 1000
) -> Dict[str, Any]:
    """Verify the integrity of a dataset (images exist, masks exist, sizes match, binary check).

    Args:
        dataset: The DocTamperDataset instance.
        sample_limit: Maximum number of samples to check. If None, checks the whole dataset.
        log_interval: Interval for logging status.

    Returns:
        Dict[str, Any]: Verification metrics and lists of failing indices.
    """
    total_available = len(dataset)
    num_to_check = total_available if sample_limit is None else min(sample_limit, total_available)
    
    logger.info(f"Starting dataset integrity checks for {num_to_check} samples of {dataset.db_path.name}...")

    missing_images = []
    missing_masks = []
    dimension_mismatches = []
    corrupted_images = []
    corrupted_masks = []
    non_binary_masks = []

    # Ensure database is open
    dataset.open_database()

    for i in range(num_to_check):
        if (i + 1) % log_interval == 0 or i == num_to_check - 1:
            logger.info(f"Verifying sample {i+1}/{num_to_check}...")

        # 1. Check Image Reading
        image_ok = False
        mask_ok = False
        img_size = None
        msk_size = None

        try:
            image = dataset.read_image(i)
            img_size = image.size
            image_ok = True
        except KeyError:
            # Key not found in DB
            missing_images.append(i)
        except Exception as e:
            # Corrupted / PIL load fail
            corrupted_images.append((i, str(e)))

        # 2. Check Mask Reading
        try:
            mask = dataset.read_mask(i, normalize=False)
            msk_size = mask.size
            mask_ok = True
            
            # Verify if mask is strictly binary (values subset of {0, 1, 255})
            mask_arr = np.array(mask.convert("L"))
            unique_vals = np.unique(mask_arr)
            is_binary = np.all(np.isin(unique_vals, [0, 1, 255]))
            if not is_binary:
                non_binary_masks.append((i, [int(v) for v in unique_vals]))
        except KeyError:
            missing_masks.append(i)
        except Exception as e:
            corrupted_masks.append((i, str(e)))

        # 3. Check Dimension Match
        if image_ok and mask_ok:
            if img_size != msk_size:
                dimension_mismatches.append((i, img_size, msk_size))

    # Determine if verification passed
    passed = len(missing_images) == 0 and \
             len(missing_masks) == 0 and \
             len(dimension_mismatches) == 0 and \
             len(corrupted_images) == 0 and \
             len(corrupted_masks) == 0 and \
             len(non_binary_masks) == 0

    results = {
        "dataset_name": dataset.db_path.name,
        "total_samples": total_available,
        "checked_samples": num_to_check,
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "dimension_mismatches": dimension_mismatches,
        "corrupted_images": corrupted_images,
        "corrupted_masks": corrupted_masks,
        "non_binary_masks": non_binary_masks,
        "passed": passed
    }

    logger.info(f"Verification completed for {dataset.db_path.name}. Passed: {passed}")
    return results

def save_validation_report(
    report: Dict[str, Any],
    save_path: Union[str, Path]
) -> None:
    """Save the validation results as a Markdown report file.

    Args:
        report: Dict returned by verify_dataset.
        save_path: Path to save the Markdown report.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    status_emoji = "✅ PASSED" if report["passed"] else "❌ FAILED"
    
    md_content = [
        f"# Dataset Integrity Validation Report - {report['dataset_name']}",
        "",
        f"**Overall Status:** {status_emoji}",
        "",
        "## Summary Metrics",
        "",
        f"- **Dataset Subset:** {report['dataset_name']}",
        f"- **Total Samples in DB:** {report['total_samples']}",
        f"- **Verified Samples:** {report['checked_samples']}",
        f"- **Pass Rate:** {((report['checked_samples'] - (len(report['missing_images']) + len(report['missing_masks']) + len(report['corrupted_images']) + len(report['corrupted_masks']) + len(report['dimension_mismatches']) + len(report['non_binary_masks']))) / report['checked_samples'] * 100):.2f}%",
        "",
        "## Detailed Checks",
        "",
        "| Check | Status | Failures Count |",
        "| --- | --- | --- |",
        f"| **Image Existence** | {'✅ OK' if not report['missing_images'] else '❌ Missing'} | {len(report['missing_images'])} |",
        f"| **Mask Existence** | {'✅ OK' if not report['missing_masks'] else '❌ Missing'} | {len(report['missing_masks'])} |",
        f"| **Image Readability** | {'✅ OK' if not report['corrupted_images'] else '❌ Corrupt'} | {len(report['corrupted_images'])} |",
        f"| **Mask Readability** | {'✅ OK' if not report['corrupted_masks'] else '❌ Corrupt'} | {len(report['corrupted_masks'])} |",
        f"| **Dimension Agreement** | {'✅ OK' if not report['dimension_mismatches'] else '❌ Mismatch'} | {len(report['dimension_mismatches'])} |",
        f"| **Mask Binarity (0/255)** | {'✅ OK' if not report['non_binary_masks'] else '❌ Non-Binary'} | {len(report['non_binary_masks'])} |",
        "",
    ]

    # Append lists of failures if any
    if report["missing_images"]:
        md_content.append("### Missing Images (First 10 Indices)")
        md_content.append(", ".join(map(str, report["missing_images"][:10])))
        md_content.append("")
        
    if report["missing_masks"]:
        md_content.append("### Missing Masks (First 10 Indices)")
        md_content.append(", ".join(map(str, report["missing_masks"][:10])))
        md_content.append("")

    if report["corrupted_images"]:
        md_content.append("### Corrupted Images (First 10)")
        for idx, err in report["corrupted_images"][:10]:
            md_content.append(f"- **Sample {idx}**: {err}")
        md_content.append("")

    if report["corrupted_masks"]:
        md_content.append("### Corrupted Masks (First 10)")
        for idx, err in report["corrupted_masks"][:10]:
            md_content.append(f"- **Sample {idx}**: {err}")
        md_content.append("")

    if report["dimension_mismatches"]:
        md_content.append("### Dimension Mismatches (First 10)")
        for idx, img_sz, msk_sz in report["dimension_mismatches"][:10]:
            md_content.append(f"- **Sample {idx}**: Image size {img_sz} != Mask size {msk_sz}")
        md_content.append("")

    if report["non_binary_masks"]:
        md_content.append("### Non-Binary Masks (First 10)")
        for idx, unique_vals in report["non_binary_masks"][:10]:
            md_content.append(f"- **Sample {idx}**: Contains non-binary values: {unique_vals}")
        md_content.append("")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    logger.info(f"Validation report saved to {save_path}")

def validate_dataset_folders(config: Any) -> Dict[str, Any]:
    """Verify that all required dataset folders exist and contain data.mdb.

    Args:
        config: DatasetConfig instance.

    Returns:
        Dict[str, Any]: Folder validation results.
    """
    subsets = {
        "TrainingSet": config.training_set,
        "TestingSet": config.testing_set,
        "SCD": config.scd_set,
        "FCD": config.fcd_set
    }
    
    results = {}
    all_ok = True
    
    for name, path in subsets.items():
        exists = path.exists()
        has_data = (path / "data.mdb").exists() if exists else False
        has_lock = (path / "lock.mdb").exists() if exists else False
        
        ok = exists and has_data
        if not ok:
            all_ok = False
            
        results[name] = {
            "path": str(path),
            "exists": exists,
            "has_data_mdb": has_data,
            "has_lock_mdb": has_lock,
            "ok": ok
        }
        
    results["all_ok"] = all_ok
    return results

def run_prepipeline_validation(config: Any, sample_limit: int = 100) -> bool:
    """Run all folder validation and high-level checks before starting the preprocessing pipeline.

    Generates a Markdown validation report.

    Args:
        config: DatasetConfig instance.
        sample_limit: Number of samples to verify in the training dataset for integrity checks.

    Returns:
        bool: True if folder structures are OK, False otherwise.
    """
    logger.info("Running pre-pipeline dataset validation...")
    
    # 1. Folder validation
    folder_report = validate_dataset_folders(config)
    
    # 2. Database validation (on TrainingSet as proxy)
    db_ok = True
    db_report = {}
    
    if folder_report["TrainingSet"]["ok"]:
        try:
            # Import here to avoid circular dependencies
            from src.dataset import DocTamperDataset
            dataset = DocTamperDataset(config.training_set, config=config)
            db_report = verify_dataset(dataset, sample_limit=sample_limit)
            db_ok = db_report["passed"]
        except Exception as e:
            logger.error(f"Error checking database integrity: {e}")
            db_ok = False
            db_report = {"passed": False, "error": str(e)}
    else:
        logger.error("Training Set folder not found or corrupted. Skipping database integrity check.")
        db_ok = False
        db_report = {"passed": False, "error": "Folder not found."}

    # Write combined pre-pipeline report
    report_path = config.reports_dir / "prepipeline_validation_report.md"
    config.ensure_output_dirs()

    status_emoji = "✅ READY" if (folder_report["all_ok"] and db_ok) else "❌ NOT READY"
    
    md_lines = [
        "# Pre-Pipeline Dataset Validation Report",
        "",
        f"**Overall Pipeline Readiness:** {status_emoji}",
        "",
        "## 1. Directory Structure Check",
        "",
        "| Subset | Path | Directory Exists | data.mdb Exists | Status |",
        "| --- | --- | --- | --- | --- |"
    ]
    
    for name in ["TrainingSet", "TestingSet", "SCD", "FCD"]:
        item = folder_report[name]
        status = "✅ OK" if item["ok"] else "❌ MISSING"
        md_lines.append(
            f"| {name} | `{item['path']}` | {'Yes' if item['exists'] else 'No'} | "
            f"{'Yes' if item['has_data_mdb'] else 'No'} | {status} |"
        )
        
    md_lines.extend([
        "",
        "## 2. Sample Integrity Check",
        "",
        f"- **Subset verified:** TrainingSet",
        f"- **Checked sample count:** {sample_limit}",
        f"- **Passed:** {'Yes' if db_ok else 'No'}",
        ""
    ])
    
    if db_ok and "total_samples" in db_report:
        md_lines.extend([
            "Detailed database validation completed successfully.",
            f"- Missing images: {len(db_report['missing_images'])}",
            f"- Missing masks: {len(db_report['missing_masks'])}",
            f"- Corrupted images: {len(db_report['corrupted_images'])}",
            f"- Corrupted masks: {len(db_report['corrupted_masks'])}",
            f"- Dimension mismatches: {len(db_report['dimension_mismatches'])}"
        ])
    elif "error" in db_report:
        md_lines.append(f"❌ **Validation Error:** {db_report['error']}")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    logger.info(f"Pre-pipeline validation report saved to {report_path}")
    return folder_report["all_ok"] and db_ok
