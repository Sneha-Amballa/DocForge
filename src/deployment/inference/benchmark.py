import time
import torch
import torch.nn as nn
from typing import Dict, Any, List

from src.logger import get_logger
from src.deployment.inference.predictor import DocForgePredictor
from src.training.utils import get_gpu_memory_usage

logger = get_logger("DocForge.DeployBenchmark")

def run_predictor_benchmark(
    predictor: DocForgePredictor,
    num_iterations: int = 5,
    batch_sizes: List[int] = [1, 2, 4]
) -> Dict[str, Any]:
    """Profile latency, throughput, and memory consumption of the inference engine.

    Args:
        predictor: DocForgePredictor instance.
        num_iterations: Number of warmup/timing runs to execute.
        batch_sizes: List of batch sizes to benchmark.

    Returns:
        Dict[str, Any]: Benchmark results details.
    """
    logger.info("=" * 60)
    logger.info("STARTING INFERENCE ENGINE BENCHMARK")
    logger.info("=" * 60)
    
    # Create a solid dummy white image for testing
    from PIL import Image
    dummy_image = Image.new("RGB", (800, 600), color="white")
    prompt = "Assess if this document contains any tampered or forged areas."
    
    results = {}
    
    for bs in batch_sizes:
        logger.info(f"Benchmarking batch size: {bs}...")
        
        # Warmup iteration
        try:
            for _ in range(2):
                for _ in range(bs):
                    _ = predictor.predict_image(dummy_image, prompt)
        except Exception as e:
            logger.error(f"Warmup prediction failed: {e}")
            continue
            
        latencies = []
        
        # Timing runs
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            # Predict bs images sequentially (simulates concurrent REST requests)
            for _ in range(bs):
                _ = predictor.predict_image(dummy_image, prompt)
                
            elapsed = (time.perf_counter() - start_time) * 1000.0  # ms
            latencies.append(elapsed)
            
        avg_batch_latency = float(np.mean(latencies))
        avg_single_latency = avg_batch_latency / bs
        throughput = (bs * 1000.0) / avg_batch_latency  # samples per second
        
        # Profile memory
        gpu_mem = get_gpu_memory_usage()
        
        logger.info(
            f"  Batch size {bs} | "
            f"Avg Batch Latency: {avg_batch_latency:.2f} ms | "
            f"Latency per sample: {avg_single_latency:.2f} ms | "
            f"Throughput: {throughput:.2f} samples/sec | "
            f"Memory: {gpu_mem}"
        )
        
        results[f"batch_size_{bs}"] = {
            "batch_size": bs,
            "avg_batch_latency_ms": avg_batch_latency,
            "latency_per_sample_ms": avg_single_latency,
            "throughput_samples_per_sec": throughput,
            "memory_usage": gpu_mem
        }
        
    return results

import numpy as np
