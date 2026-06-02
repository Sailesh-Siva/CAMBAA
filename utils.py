import os
import cv2
import joblib
import torch
import torch.nn as nn
import numpy as np

from ultralytics import YOLO
from timm import create_model
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from pytorch_grad_cam import EigenCAM

from utils_model2 import model2
# =========================================================
# SETTINGS
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 512
CONF_THRES = 0.25
PAD_RATIO = 0.20

RESULT_FOLDER = "static/results"

os.makedirs(RESULT_FOLDER, exist_ok=True)

# =========================================================
# PATHS
# =========================================================

YOLO_WEIGHTS = hf_hub_download(
    repo_id="saileshsiva/CAMBAA",
    filename="best.pt"
)

MODEL_PATH = hf_hub_download(
    repo_id="saileshsiva/CAMBAA",
    filename="evcc_weights_only.pth"
)

SVR_ABOVE_8_PATH = hf_hub_download(
    repo_id="saileshsiva/CAMBAA",
    filename="svr_above_8.pkl"
)

SVR_BELOW_8_PATH = hf_hub_download(
    repo_id="saileshsiva/CAMBAA",
    filename="svr_below_8.pkl"
)

# =========================================================
# LOAD YOLO
# =========================================================

yolo_model = YOLO(YOLO_WEIGHTS)

print("YOLO model loaded")


# =========================================================
# CBAM
# =========================================================

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        B, C, H, W = x.shape

        avg_pool = torch.mean(x, dim=(2, 3))

        max_pool, _ = torch.max(
            x.reshape(B, C, -1),
            dim=2
        )

        attn = self.mlp(avg_pool) + self.mlp(max_pool)

        attn = self.sigmoid(attn).view(B, C, 1, 1)

        return x * attn


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_pool = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_pool, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        concat = torch.cat(
            [avg_pool, max_pool],
            dim=1
        )

        attn = self.sigmoid(self.conv(concat))

        return x * attn


class CBAM(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.channel_att = ChannelAttention(in_channels)

        self.spatial_att = SpatialAttention()

    def forward(self, x):

        identity = x

        x = self.channel_att(x)

        x = self.spatial_att(x)

        return x + identity


# =========================================================
# MAIN MODEL
# =========================================================

class MaxViTBoneAgeRegressor(nn.Module):

    def __init__(
        self,
        model_name="maxvit_base_tf_512",
        dropout=0.1
    ):
        super().__init__()

        self.backbone = create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool=""
        )

        self.feature_dim = self.backbone.num_features

        self.cbam = CBAM(self.feature_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.gender_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.LayerNorm(32)
        )

        self.regressor = nn.Sequential(

            nn.Linear(self.feature_dim + 32, 512),

            nn.GELU(),

            nn.LayerNorm(512),

            nn.Dropout(dropout),

            nn.Linear(512, 128),

            nn.GELU(),

            nn.LayerNorm(128),

            nn.Dropout(dropout),

            nn.Linear(128, 1)
        )

    # -----------------------------------------------------
    # FEATURE EXTRACTION FOR SVR
    # -----------------------------------------------------

    def extract_features(self, x, gender):

        x = self.backbone(x)

        x = self.cbam(x)

        x = self.pool(x).flatten(1)

        gender = gender.unsqueeze(1)

        gender_feat = self.gender_fc(gender)

        x = torch.cat(
            [x, gender_feat],
            dim=1
        )

        return x

    # -----------------------------------------------------
    # FORWARD FOR CAM
    # -----------------------------------------------------

    def forward_for_cam(self, x, gender):

        x = self.backbone(x)

        x = self.cbam(x)

        x = self.pool(x).flatten(1)

        gender = gender.unsqueeze(1)

        gender_feat = self.gender_fc(gender)

        x = torch.cat(
            [x, gender_feat],
            dim=1
        )

        x = self.regressor(x)

        return x.squeeze(1)


# =========================================================
# REGRESSION TARGET
# =========================================================

class RegressionTarget:

    def __call__(self, model_output):
        return model_output


# =========================================================
# MODEL WRAPPER FOR EIGENCAM
# =========================================================

class ModelWrapper(nn.Module):

    def __init__(self, model, gender_tensor):
        super().__init__()

        self.model = model
        self.gender_tensor = gender_tensor

    def forward(self, x):

        return self.model.forward_for_cam(
            x,
            self.gender_tensor
        )


# =========================================================
# LOAD MAIN MODEL
# =========================================================

model = MaxViTBoneAgeRegressor().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.eval()

print("Main model loaded")


# =========================================================
# LOAD SVR MODELS
# =========================================================

svr_above_8 = joblib.load(SVR_ABOVE_8_PATH)

svr_below_8 = joblib.load(SVR_BELOW_8_PATH)

print("SVR models loaded")


# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# MERGE BOXES
# =========================================================

def merge_boxes(boxes, img_w, img_h, pad_ratio=0.1):

    x1 = boxes[:, 0].min().item()
    y1 = boxes[:, 1].min().item()
    x2 = boxes[:, 2].max().item()
    y2 = boxes[:, 3].max().item()

    pad_w = (x2 - x1) * pad_ratio
    pad_h = (y2 - y1) * pad_ratio

    x1 -= pad_w
    y1 -= pad_h
    x2 += pad_w
    y2 += pad_h

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w, int(x2))
    y2 = min(img_h, int(y2))

    return x1, y1, x2, y2


# =========================================================
# YOLO CROP FUNCTION
# =========================================================

def crop_hand_with_yolo(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]

    results = yolo_model(
        img,
        conf=CONF_THRES,
        verbose=False
    )

    if len(results[0].boxes) == 0:
        crop = img

    else:

        boxes = results[0].boxes.xyxy.cpu()

        areas = (
            (boxes[:, 2] - boxes[:, 0]) *
            (boxes[:, 3] - boxes[:, 1])
        )

        boxes = boxes[areas > 500]

        if boxes.shape[0] == 0:
            crop = img

        else:

            x1, y1, x2, y2 = merge_boxes(
                boxes,
                w,
                h,
                PAD_RATIO
            )

            crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("Empty crop produced")

    if crop.shape[0] > IMG_SIZE or crop.shape[1] > IMG_SIZE:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    crop_resized = cv2.resize(
        crop,
        (IMG_SIZE, IMG_SIZE),
        interpolation=interp
    )

    return crop_resized


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================

def predict_bone_age_model1(
    image_path,
    gender_value,
    svr_type="above_8",
    output_filename="eigencam_overlay.png"
):

    crop = crop_hand_with_yolo(image_path)

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    image_pil = Image.fromarray(crop_rgb)

    image_tensor = transform(image_pil)

    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    gender = torch.tensor(
        [gender_value],
        dtype=torch.float32
    ).to(DEVICE)

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    with torch.no_grad():

        features = model.extract_features(
            input_tensor,
            gender
        )

    features = features.cpu().numpy()

    # =====================================================
    # SVR PREDICTION
    # =====================================================

    if svr_type == "above_8":

        prediction = svr_above_8.predict(features)[0]

        prediction = np.clip(
            prediction,
            96,
            228
        )

    elif svr_type == "below_8":

        prediction = svr_below_8.predict(features)[0]

        prediction = np.clip(
            prediction,
            0,
            95
        )

    else:

        raise ValueError(
            "svr_type must be either "
            "'above_8' or 'below_8'"
        )

    prediction = round(prediction)

    # =====================================================
    # EIGENCAM
    # =====================================================

    wrapped_model = ModelWrapper(
        model,
        gender
    )

    wrapped_model.eval()

    target_layers = [
        model.backbone.stages[-1]
    ]

    cam_extractor = EigenCAM(
        model=wrapped_model,
        target_layers=target_layers
    )

    targets = [RegressionTarget()]

    cam = cam_extractor(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # =====================================================
    # UNNORMALIZE IMAGE
    # =====================================================

    img = image_tensor.permute(1, 2, 0).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = std * img + mean

    img = np.clip(img, 0, 1)

    # =====================================================
    # RESIZE CAM
    # =====================================================

    cam = cv2.resize(
        cam,
        (img.shape[1], img.shape[0])
    )

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    heatmap = heatmap.astype(np.float32) / 255.0

    # =====================================================
    # OVERLAY
    # =====================================================

    overlay = 0.6 * img + 0.4 * heatmap

    overlay = np.clip(overlay, 0, 1)

    # =====================================================
    # SAVE RESULT
    # =====================================================

    save_img = (overlay * 255).astype(np.uint8)

    result_path = os.path.join(
        RESULT_FOLDER,
        output_filename
    )

    cv2.imwrite(
        result_path,
        cv2.cvtColor(
            save_img,
            cv2.COLOR_RGB2BGR
        )
    )

    return prediction, result_path

def predict_bone_age_model2(
    image_path,
    gender_value,
    svr_type="above_8",
    output_filename="eigencam_overlay.png"
):

    crop = crop_hand_with_yolo(image_path)

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    image_pil = Image.fromarray(crop_rgb)

    image_tensor = transform(image_pil)

    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    gender = torch.tensor(
        [gender_value],
        dtype=torch.float32
    ).to(DEVICE)

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    with torch.no_grad():

        features = model.extract_features(
            input_tensor,
            gender
        )

    features = features.cpu().numpy()

    # =====================================================
    # MODEL2 PREDICTION
    # =====================================================

    features = features.flatten()
    
    dl_feature_dict = {
        f"feature_{i}": float(v)
        for i, v in enumerate(features)
    }

    prediction = model2(

        image_path=image_path,

        gender=gender_value,

        age_group=svr_type,

        dl_feature_dict=dl_feature_dict
    )

    prediction = round(prediction["predicted_boneage_months"])

    # =====================================================
    # EIGENCAM
    # =====================================================

    wrapped_model = ModelWrapper(
        model,
        gender
    )

    wrapped_model.eval()

    target_layers = [
        model.backbone.stages[-1]
    ]

    cam_extractor = EigenCAM(
        model=wrapped_model,
        target_layers=target_layers
    )

    targets = [RegressionTarget()]

    cam = cam_extractor(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # =====================================================
    # UNNORMALIZE IMAGE
    # =====================================================

    img = image_tensor.permute(1, 2, 0).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = std * img + mean

    img = np.clip(img, 0, 1)

    # =====================================================
    # RESIZE CAM
    # =====================================================

    cam = cv2.resize(
        cam,
        (img.shape[1], img.shape[0])
    )

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    heatmap = heatmap.astype(np.float32) / 255.0

    # =====================================================
    # OVERLAY
    # =====================================================

    overlay = 0.6 * img + 0.4 * heatmap

    overlay = np.clip(overlay, 0, 1)

    # =====================================================
    # SAVE RESULT
    # =====================================================

    save_img = (overlay * 255).astype(np.uint8)

    result_path = os.path.join(
        RESULT_FOLDER,
        output_filename
    )

    cv2.imwrite(
        result_path,
        cv2.cvtColor(
            save_img,
            cv2.COLOR_RGB2BGR
        )
    )

    return prediction, result_path