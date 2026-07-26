import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_fundus_image, get_pytorch_transforms

class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 Blindness Detection dataset.
    Supports Stage 1 (Binary: Healthy vs. DR Present) and Stage 2 (4-Class Severity).
    """
    SEVERITY_MAP = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }

    STAGE2_MAP = {
        1: 0, # Mild -> Class 0
        2: 1, # Moderate -> Class 1
        3: 2, # Severe -> Class 2
        4: 3  # Proliferative -> Class 3
    }

    STAGE2_CLASSES = ["Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

    def __init__(self, df: pd.DataFrame, img_dir: str, stage: int = 1, transform=None, img_size: int = 224, apply_clahe_preprocessing: bool = True):
        self.df = df.copy().reset_index(drop=True)
        self.img_dir = img_dir
        self.stage = stage
        self.img_size = img_size
        self.apply_clahe_preprocessing = apply_clahe_preprocessing

        # Setup labels based on Stage
        if self.stage == 1:
            # Stage 1: Binary (0 = Healthy/No DR, 1 = Diseased/DR Present)
            self.df['stage_label'] = self.df['diagnosis'].apply(lambda x: 0 if x == 0 else 1)
        elif self.stage == 2:
            # Stage 2: Filter out 0 (Healthy), keep only 1, 2, 3, 4
            self.df = self.df[self.df['diagnosis'] > 0].reset_index(drop=True)
            self.df['stage_label'] = self.df['diagnosis'].map(self.STAGE2_MAP)
        else:
            raise ValueError("Stage must be 1 or 2.")

        self.transform = transform if transform is not None else get_pytorch_transforms(img_size=img_size, is_train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = str(row['id_code'])
        if not image_name.endswith('.png') and not image_name.endswith('.jpg'):
            image_name += '.png'
            
        img_path = os.path.join(self.img_dir, image_name)

        if not os.path.exists(img_path):
            # Fallback if image file is not found (for dummy dataset testing)
            raw_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        else:
            raw_img = Image.open(img_path).convert("RGB")

        if self.apply_clahe_preprocessing:
            processed_img_np = preprocess_fundus_image(raw_img, img_size=self.img_size)
        else:
            processed_img_np = np.array(raw_img.resize((self.img_size, self.img_size)))

        tensor_img = self.transform(processed_img_np)
        label = torch.tensor(row['stage_label'], dtype=torch.long)

        return tensor_img, label
