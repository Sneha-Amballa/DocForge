# Document Forgery & Tampering Detection using Fine-Tuned Vision-Language Models

A deep learning and computer vision project focused on detecting, localizing, and explaining document forgery using open-source vision-language models (VLMs). The project is designed around real-world document verification workflows, where identity documents, receipts, invoices, and contracts may be manipulated through copy-move edits, splicing, text replacement, or print-based alterations.

## Overview

Document forgery is a major challenge in digital identity verification, KYC automation, and financial document processing. Fraudulent edits can be subtle, visually persuasive, and hard to detect when documents are compressed, re-photographed, or captured under poor lighting conditions.

This repository presents a practical research and engineering pipeline that uses a fine-tuned Vision-Language Model to:

- determine whether a document is authentic or tampered,
- localize the forged region using bounding boxes or masks,
- predict the forgery type,
- generate a short natural language explanation for the detected manipulation.

Rather than relying on commercial APIs, the approach emphasizes local model fine-tuning using parameter-efficient methods such as LoRA/PEFT.

## Why This Project Fits the Problem Space

This project is aligned with the core themes of document understanding, OCR enhancement, and image forgery detection. The work goes beyond a generic VLM demo by combining:

- multimodal document reasoning,
- image-level forgery classification,
- region localization,
- adversarial robustness testing,
- deployment-oriented optimization.

The goal is not only to report validation accuracy, but to demonstrate behavior under noisy and real-world-like document capture conditions.

## Project Objectives

The primary objectives of the project are to:

1. Train a VLM for document forgery detection and localization.
2. Improve robustness under real-world degradations.
3. Build a structured adversarial evaluation benchmark.
4. Analyze failure modes instead of only reporting aggregate accuracy.
5. Study deployment trade-offs through model compression and quantization.

## Core Features

- Binary authentication prediction
  - Authentic
  - Tampered

- Forgery localization
  - Bounding box
  - Pixel mask when available

- Forgery type classification
  - Copy-move
  - Splicing
  - Text replacement
  - Text insertion
  - Text deletion
  - Photo replacement
  - Print-based editing

- Natural language explanation
  - Example: “The date of birth field appears to have been digitally modified.”

- Adversarial robustness evaluation
  - JPEG recompression
  - Motion blur
  - Camera recapture
  - Low-light conditions
  - Partial occlusion
  - Distribution shift

## Problem Statement

Given an image of a document, the model predicts whether the document is authentic or tampered and, if needed, identifies the manipulated region and explanation.

Input:

- Document image

Output:

```json
{
  "tampered": true,
  "region": [x, y, w, h],
  "type": "Text Replacement",
  "confidence": 0.96,
  "explanation": "The date of birth field appears to have been digitally modified."
}
```

## Motivation

Identity verification systems process thousands to millions of document images each day. Fraudulent edits commonly involve:

- manipulated Aadhaar or ID cards,
- altered PAN documents,
- fake passports or driver’s licenses,
- modified receipts and invoices,
- forged certificates and employment documents.

Automated detection of such manipulations can reduce fraud, improve verification pipelines, and increase the reliability of downstream KYC and compliance systems.

## Datasets

### Training Data

- DocTamper
  - Large public document forgery dataset
  - Pixel-level annotations
  - Multiple tampering categories
  - Approximately 170K images

- MIDV-2020
  - Identity document dataset used for synthetic forgery generation
  - Supports synthetic edits such as name replacement, DOB modification, address editing, and photo swapping

### Out-of-Distribution Evaluation Data

The following datasets are intentionally held out from training to evaluate generalization and failure robustness:

- AIForge-Doc
- MixTamper
- Receipt Forgery Dataset
- T-SROIE
- Tampered-IC13

This split provides a stronger benchmark story than relying only on a clean validation set.

## Model Architecture

The system is built around a multimodal vision-language backbone that jointly reasons over document images and textual supervision.

```text
Document Image
      |
      v
Vision Encoder (VLM)
      |
      v
Cross-Modal Reasoning
      |
      v
Language Decoder / Structured Output Head
      |
      +--> Tampered / Authentic
      +--> Forgery Region
      +--> Forgery Type
      +--> Explanation
```

### Supported Base Models

The project supports a range of open-weight VLMs, including:

- Qwen2-VL
- LLaVA-NeXT
- InternVL2

The primary implementation targets Qwen2-VL-2B because it offers a practical balance between capability and GPU efficiency.

## Fine-Tuning Strategy

Instead of full-parameter training, the project uses parameter-efficient fine-tuning.

### Primary Method

- LoRA / PEFT

### Advantages

- Lower GPU memory usage
- Faster training iteration cycles
- Smaller checkpoints
- Easier deployment and portability

### Libraries

- PyTorch
- Transformers
- PEFT
- TRL
- Accelerate
- OpenCV
- Pillow
- NumPy
- Pandas
- scikit-learn

## Synthetic Data Generation Pipeline

Real-world document forgery data is limited and often imbalanced. To address this, the repository includes a synthetic tampering generation workflow.

### Pipeline

1. Start from an authentic document image.
2. Detect editable regions or OCR text fields.
3. Apply random tampering operations.
4. Match font and visual style where possible.
5. Blend manipulated content into the document.
6. Introduce compression, noise, and photographic artifacts.
7. Produce a forged sample with structured annotations.

### Example Forge Types

- Copy-move
- Splicing
- Text replacement
- Name editing
- Date editing
- Price editing
- Signature replacement
- Photo swapping

## Training Pipeline

```text
Dataset
  |
  v
Preprocessing
  |
  v
Image Augmentation
  |
  v
Tokenizer / Vision Processor
  |
  v
LoRA Fine-Tuning
  |
  v
Validation and Checkpointing
```

## Evaluation Strategy

The project emphasizes evaluation beyond clean validation performance.

### Clean Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 score
- IoU
- mAP

### Adversarial Evaluation Conditions

The system is evaluated under progressively harder image perturbations, including:

- JPEG compression
- Motion blur
- Gaussian noise
- low-resolution conditions
- camera recapture
- occlusion
- perspective distortion

### Key Evaluation Principle

The real deliverable is not just a single accuracy number. The project reports a degradation table showing how model performance changes as document quality degrades.

## Failure Analysis

Instead of hiding mistakes, the project explicitly studies failure cases. Examples include:

- incorrect bounding box localization,
- missed tiny edits,
- false positives triggered by compression artifacts,
- missed photo replacement,
- OCR confusion,
- over-reliance on JPEG artifacts.

Each failure is documented with:

- the input image,
- ground truth annotation,
- the model’s prediction,
- a hypothesis describing the likely cause,
- a proposed improvement.

## Deployment and Optimization

For deployment-oriented evaluation, the fine-tuned model is tested under different inference settings.

### Methods

- AWQ quantization
- GGUF conversion

### Trade-off Analysis

The project compares:

- accuracy,
- latency,
- memory usage,
- compression effects.

This provides a production-aware perspective that balances quality with inference cost.

## Project Structure

```text
DocForge/
├── datasets/
│   ├── doctamper/
│   ├── midv2020/
│   ├── aiforge_doc/
│   └── synthetic/
├── preprocessing/
│   ├── generate_synthetic.py
│   ├── augmentations.py
│   └── preprocess.py
├── training/
│   ├── train.py
│   ├── trainer.py
│   ├── config.py
│   └── lora_config.py
├── evaluation/
│   ├── evaluate.py
│   ├── robustness.py
│   ├── metrics.py
│   └── visualization.py
├── deployment/
│   ├── quantize.py
│   └── inference.py
├── notebooks/
├── outputs/
├── checkpoints/
├── reports/
├── README.md
├── requirements.txt
└── LICENSE
```

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Multimodal Modeling | Transformers |
| PEFT | LoRA, PEFT |
| Training | Accelerate |
| Vision Processing | OpenCV, Pillow |
| OCR | EasyOCR, Tesseract |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Experiment Tracking | Weights & Biases |
| Evaluation | scikit-learn |
| Quantization | AWQ, GGUF |

## Expected Outcomes

This project is designed to demonstrate practical end-to-end competence in:

- Vision-Language Model fine-tuning,
- document understanding,
- OCR-aware multimodal reasoning,
- synthetic data generation,
- adversarial benchmarking,
- failure analysis,
- deployment optimization.

## Future Improvements

- multilingual document support,
- segmentation of tampered regions using SAM2,
- stronger OCR-aware reasoning,
- preference optimization using DPO,
- multi-document reasoning,
- active learning for hard examples,
- distillation for edge deployment,
- real-time inference APIs.

## Acknowledgements

This project builds on publicly available research datasets and open-source models, including DocTamper, MIDV-2020, AIForge-Doc, MixTamper, Qwen2-VL, LLaVA-NeXT, InternVL2, PyTorch, Hugging Face Transformers, and the PEFT ecosystem.

## License

This project is intended for educational and research purposes only. Please review and comply with the individual licenses and usage terms of all datasets and pretrained models before using them for commercial or production applications.
