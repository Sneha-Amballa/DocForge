import json

def build_qwen_prompt(sample, image_obj):
    """
    Builds the multimodal conversation format for Qwen2-VL.
    Args:
        sample: A dictionary from the unified DocForge dataset.
        image_obj: A PIL Image or valid image path expected by Qwen-VL.
    Returns:
        messages: A list of message dictionaries.
    """
    tampered = sample.get("tampered", False)
    bboxes = sample.get("bounding_boxes", [])
    forgery = sample.get("forgery_type", "unknown")
    
    # We map [x_min, y_min, x_max, y_max] to [x, y, w, h] for the output
    regions = []
    for box in bboxes:
        if len(box) == 4:
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            regions.append([int(x_min), int(y_min), int(w), int(h)])
            
    # For simplicity, if multiple boxes, we just return the first one or a list.
    # The prompt asks for "region" (singular or list). Let's provide a list if there are any.
    output_dict = {
        "tampered": tampered,
        "region": regions[0] if len(regions) > 0 else (None if not tampered else []),
        "type": forgery if tampered else None
    }
    
    ground_truth_json = json.dumps(output_dict)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj, "max_pixels": 256 * 256},
                {"type": "text", "text": "Analyze this document for tampering. Output only JSON containing 'tampered' (bool), 'region' ([x,y,w,h]), and 'type' (string)."},
            ],
        },
        {
            "role": "assistant",
            "content": ground_truth_json
        }
    ]
    
    return messages
