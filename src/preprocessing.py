import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crops black borders surrounding retinal fundus images.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_any = mask.any(1)
        if not check_any.any():
            return img
        else:
            img1 = img[np.ix_(mask.any(1), mask.any(0), [True, True, True])]
            return img1
    return img

def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the 
    green channel (or LAB lightness channel) of a retinal image to highlight blood vessels.
    """
    # Convert RGB to LAB color space
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    limg = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

def preprocess_fundus_image(image_input, img_size: int = 224) -> np.ndarray:
    """
    Pipeline: PIL/Numpy input -> Crop borders -> CLAHE enhancement -> Resize.
    Returns RGB Numpy array (uint8).
    """
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img_np = image_input.copy()
        if img_np.shape[-1] == 4: # RGBA -> RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError("Unsupported image input format. Must be PIL Image or NumPy array.")

    # 1. Auto-crop black borders
    cropped = crop_image_from_gray(img_np)

    # 2. Apply CLAHE contrast enhancement
    clahe_enhanced = apply_clahe(cropped)

    # 3. Resize to target dimensions
    resized = cv2.resize(clahe_enhanced, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return resized

def get_pytorch_transforms(img_size: int = 224, is_train: bool = False):
    """
    Returns PyTorch transformation pipeline.
    """
    # ImageNet standard normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    return transform
