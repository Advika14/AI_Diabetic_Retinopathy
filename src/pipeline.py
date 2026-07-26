import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from src.preprocessing import preprocess_fundus_image, get_pytorch_transforms
from src.models import build_stage1_model, build_stage2_model

class DRTwoStagePipeline:
    """
    End-to-End Two-Stage Diabetic Retinopathy Detection & Severity Grading Pipeline.
    """
    STAGE1_CLASSES = ["Healthy (No DR)", "Diabetic Retinopathy Present"]
    STAGE2_CLASSES = ["Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

    SEVERITY_DESCRIPTIONS = {
        "Healthy (No DR)": "No signs of diabetic retinopathy detected in the fundus image. Routine annual eye examination recommended.",
        "Mild DR": "Microaneurysms present. Early stage of vascular change. Monitoring recommended.",
        "Moderate DR": "More extensive microaneurysms, hemorrhages, or hard exudates present. Specialist review advised.",
        "Severe DR": "Severe intraretinal hemorrhages in 4 quadrants, venous beading, or prominent IRMA. High risk of progression. Immediate ophthalmologist referral required.",
        "Proliferative DR": "Neovascularization (new abnormal blood vessels) or vitreous hemorrhage detected. Advanced stage with vision threat. Urgent specialist intervention needed."
    }

    SEVERITY_COLORS = {
        "Healthy (No DR)": "#28a745",     # Green
        "Mild DR": "#ffc107",            # Yellow
        "Moderate DR": "#fd7e14",        # Orange
        "Severe DR": "#dc3545",          # Red
        "Proliferative DR": "#721c24"    # Dark Red
    }

    def __init__(self, stage1_weights_path: str = None, stage2_weights_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize models
        self.stage1_model = build_stage1_model(pretrained=False).to(self.device)
        self.stage2_model = build_stage2_model(pretrained=False).to(self.device)

        self.stage1_loaded = False
        self.stage2_loaded = False

        # Load weights if provided and existing
        if stage1_weights_path and os.path.exists(stage1_weights_path):
            try:
                self.stage1_model.load_state_dict(torch.load(stage1_weights_path, map_location=self.device))
                self.stage1_loaded = True
            except Exception as e:
                print(f"Warning: Could not load Stage 1 weights: {e}")

        if stage2_weights_path and os.path.exists(stage2_weights_path):
            try:
                self.stage2_model.load_state_dict(torch.load(stage2_weights_path, map_location=self.device))
                self.stage2_loaded = True
            except Exception as e:
                print(f"Warning: Could not load Stage 2 weights: {e}")

        self.stage1_model.eval()
        self.stage2_model.eval()
        self.transform = get_pytorch_transforms(img_size=224, is_train=False)

    def predict(self, image_input) -> dict:
        """
        Runs full 2-stage prediction pipeline on a single image.
        Returns detailed prediction dictionary.
        """
        # 1. Preprocess & CLAHE
        clahe_np = preprocess_fundus_image(image_input, img_size=224)
        tensor_img = self.transform(clahe_np).unsqueeze(0).to(self.device)

        # 2. Stage 1 Binary Inference
        with torch.no_grad():
            s1_logits = self.stage1_model(tensor_img)
            s1_probs = F.softmax(s1_logits, dim=1).squeeze(0).cpu().numpy()

        s1_pred_class = int(np.argmax(s1_probs))
        s1_confidence = float(s1_probs[s1_pred_class])
        is_diseased = (s1_pred_class == 1)

        result = {
            "clahe_image_np": clahe_np,
            "stage1": {
                "class_id": s1_pred_class,
                "label": self.STAGE1_CLASSES[s1_pred_class],
                "confidence": s1_confidence,
                "probabilities": {
                    "Healthy": float(s1_probs[0]),
                    "Diseased": float(s1_probs[1])
                },
                "weights_loaded": self.stage1_loaded
            },
            "stage2": None,
            "final_diagnosis": "Healthy (No DR)",
            "severity_badge_color": self.SEVERITY_COLORS["Healthy (No DR)"],
            "clinical_guidance": self.SEVERITY_DESCRIPTIONS["Healthy (No DR)"]
        }

        # 3. Stage 2 Inference (Only if Stage 1 detects DR)
        if is_diseased:
            with torch.no_grad():
                s2_logits = self.stage2_model(tensor_img)
                s2_probs = F.softmax(s2_logits, dim=1).squeeze(0).cpu().numpy()

            s2_pred_class = int(np.argmax(s2_probs))
            s2_confidence = float(s2_probs[s2_pred_class])
            s2_label = self.STAGE2_CLASSES[s2_pred_class]

            result["stage2"] = {
                "class_id": s2_pred_class,
                "label": s2_label,
                "confidence": s2_confidence,
                "probabilities": {
                    self.STAGE2_CLASSES[i]: float(s2_probs[i]) for i in range(4)
                },
                "weights_loaded": self.stage2_loaded
            }
            result["final_diagnosis"] = s2_label
            result["severity_badge_color"] = self.SEVERITY_COLORS[s2_label]
            result["clinical_guidance"] = self.SEVERITY_DESCRIPTIONS[s2_label]

        return result

# Verification self-test block
if __name__ == "__main__":
    pipeline = DRTwoStagePipeline()
    dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    res = pipeline.predict(dummy_img)
    print("Pipeline test successful!")
    print("Diagnosis:", res["final_diagnosis"])
