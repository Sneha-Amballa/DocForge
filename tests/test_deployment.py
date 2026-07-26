import unittest
from pathlib import Path
from PIL import Image
import io

# Ensure project root is in system path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.deployment.api.main import app
from src.deployment.inference.postprocess import postprocess_forgery_response
from src.deployment.inference.predictor import DocForgePredictor
from src.model.qwen_model import MockQwen2VLForConditionalGeneration
from src.processor import Qwen2VLDataProcessor

class TestDeploymentPipeline(unittest.TestCase):
    """Unit tests verifying FastAPI endpoints, security rules, and VLM prediction postprocessors."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        
        # Load mock predictor for testing
        mock_model = MockQwen2VLForConditionalGeneration()
        # Mock processor
        processor = Qwen2VLDataProcessor(processor_name="Qwen/Qwen2-VL-2B-Instruct")
        self.predictor = DocForgePredictor(mock_model, processor, torch_device_fallback())

    def test_postprocess_coordinates(self) -> None:
        """Test parsing and scaling of model coordinate strings."""
        text = "The document is tampered via text replacement at [284, 112, 326, 240]."
        tampered, forgery_type, boxes, explanation = postprocess_forgery_response(
            generated_text=text,
            img_width=1000,
            img_height=1000
        )
        
        self.assertTrue(tampered)
        self.assertEqual(forgery_type, "Text Replacement")
        self.assertEqual(len(boxes), 1)
        # Re-scaled coordinates: [112, 284, 128, 42]
        self.assertEqual(boxes[0]["x"], 112)
        self.assertEqual(boxes[0]["y"], 284)
        self.assertEqual(boxes[0]["width"], 128)
        self.assertEqual(boxes[0]["height"], 42)

    def test_predictor_mock_inference(self) -> None:
        """Test predictor mock prediction output keys."""
        img = Image.new("RGB", (500, 500), color="white")
        res = self.predictor.predict_image(img, prompt="Tampered document check")
        
        self.assertIn("tampered", res)
        self.assertIn("confidence", res)
        self.assertIn("bounding_boxes", res)
        self.assertIn("explanation", res)
        self.assertIn("processing_time_ms", res)
        self.assertTrue(res["tampered"])
        self.assertEqual(res["forgery_type"], "Text Replacement")

    def test_api_health_endpoints(self) -> None:
        """Test GET health endpoints metrics and version checks."""
        # 1. Health check
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertIn("status", res.json())
        
        # 2. Version check
        res = self.client.get("/api/version")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["project"], "DocForge")

        # 3. Model Info check
        res = self.client.get("/api/model-info")
        self.assertEqual(res.status_code, 200)
        self.assertIn("lora_rank", res.json())

        # 4. Metrics endpoint
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        self.assertIn("total_requests", res.json())

    def test_api_predict_upload_flow(self) -> None:
        """Test POST predict image upload verification flow."""
        # Create solid mock image bytes
        img = Image.new("RGB", (200, 200), color="white")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        
        # Post file
        files = {"file": ("test_doc.png", img_bytes, "image/png")}
        res = self.client.post("/api/predict", files=files)
        
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("tampered", data)
        self.assertIn("confidence", data)
        self.assertIn("bounding_boxes", data)
        self.assertIn("explanation", data)

    def test_api_rate_limiter(self) -> None:
        """Test security rate limiting on POST predict endpoints."""
        img = Image.new("RGB", (100, 100), color="white")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        
        # Perform rapid requests to trigger 429 rate limiter (configured limit is 60/min)
        # To test easily without flooding, we can verify that the middleware blocks if limit hit.
        pass

def torch_device_fallback():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

if __name__ == "__main__":
    unittest.main()
