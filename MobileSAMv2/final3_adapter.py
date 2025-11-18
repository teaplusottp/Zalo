"""
Integrated Inference Pipeline with MobileViT Adapter + Kalman Tracking
CORRECTED & OPTIMIZED VERSION - Device-consistent, GPU-accelerated

Features:
  - ObjectAwareModel (YOLO) for detection
  - MobileSAMv2 for segmentation
  - MobileViT + Weight Adapter for similarity matching
  - Kalman Filter for object tracking
  - CORRECT: Template scores [Q,N,K], Instance scores [Q,N], MAX aggregation
  - Flexible video_id input
"""

import os
import cv2
import json
import time
import sys
import torch
import numpy as np
import timm
from torch import nn
from PIL import Image
from torchvision import transforms as T
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# MobileSAMv2 imports
from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import SamPredictor
from mobilesamv2.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer, Sam
from tinyvit.tiny_vit import TinyViT


# ====================== CONFIGURATION ======================
class Config:
    """Centralized configuration - modify these for different objects"""
    
    def __init__(self, video_id="CardboardBox_1"):
        self.video_id = video_id
        
        # Base paths (keep on your machine)
        self.data_base = r"D:\code\detect\public_test\public_test\samples"
        self.segment_base = r"segment_objects"
        self.results_base = "results"
        
        # Video and template paths (auto-generated from video_id)
        self.video_path = os.path.join(self.data_base, video_id, "drone_video.mp4")
        self.template_paths = [
            os.path.join(self.segment_base, video_id, "mask_images", f"img_{i}.png")
            for i in range(1, 4)
        ]
        
        # Output paths
        os.makedirs(os.path.join(self.results_base, video_id), exist_ok=True)
        self.output_video = os.path.join(self.results_base, video_id, 
                                         f"track_adapter_kalman_{video_id}.mp4")
        self.output_json = os.path.join(self.results_base, video_id, 
                                        f"predictions_adapter_kalman_{video_id}.json")
        
        # Model paths (weights)
        self.adapter_checkpoint = "./weight/adapter_best.pth"
        self.sam_checkpoint = './weight/mobile_sam.pt'
        self.yolo_model = './weight/ObjectAwareModel.pt'
        
        # Hyperparameters
        self.SIM_THRESHOLD = 0.9
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dim = 512
        self.img_size = 224
        
        # Video processing
        self.start_sec = 0
        self.end_sec = 1000000
        self.conf_thres = 0.3
        self.iou_thres = 0.9
        self.min_area_ratio = 0.0005
        self.max_area_ratio = 0.05
        
        # Kalman tracking
        self.MAX_AGE = 10
        self.MIN_HITS = 3
        self.IOU_THRESHOLD = 0.3
    
    def validate(self):
        """Check if all required files exist"""
        errors = []
        
        if not os.path.exists(self.video_path):
            errors.append(f"Video not found: {self.video_path}")
        
        for i, path in enumerate(self.template_paths):
            if not os.path.exists(path):
                errors.append(f"Template {i+1} not found: {path}")
        
        if not os.path.exists(self.adapter_checkpoint):
            errors.append(f"Adapter checkpoint not found: {self.adapter_checkpoint}")
        
        if errors:
            print("\nConfiguration Errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True


# ====================== FEATURE EXTRACTOR WITH ADAPTER ======================

class ClipStyleMobileViT(nn.Module):
    """MobileViT backbone with global average pooling"""
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = timm.create_model("mobilevit_s.cvnets_in1k", pretrained=True)
        
        # Global average pooling + head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(self.backbone.num_features, embedding_dim)
        self.embedding_dim = embedding_dim
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.flatten(1)
        features = self.head(x)
        return features


class WeightAdapter(nn.Module):
    """Trained Weight Adapter for feature refinement"""
    def __init__(self, feature_dim=512, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, max(feature_dim // reduction, 32), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(feature_dim // reduction, 32), feature_dim, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        weights = self.fc(features)
        return features * weights


class SimilarityModel:
    """
    CORRECTED VERSION - Following Paper's Exact Matching Approach
    
    Matching Pipeline:
    1. Extract template features [K, D]
    2. Extract proposal features [Q, D]
    3. Compute similarity matrix [Q, K] → MAX aggregation [Q]
    4. Stable matching to assign instance IDs
    """
    
    def __init__(self, config):
        self.device = config.device
        self.img_size = config.img_size
        
        # Load backbone
        print("Loading MobileViT backbone...")
        self.backbone = ClipStyleMobileViT(embedding_dim=config.embedding_dim).to(self.device)
        self.backbone.eval()
        
        # Load adapter
        print(f"Loading Weight Adapter from {config.adapter_checkpoint}...")
        self.adapter = WeightAdapter(feature_dim=config.embedding_dim, reduction=4).to(self.device)
        
        try:
            checkpoint = torch.load(config.adapter_checkpoint, map_location=self.device)
            self.adapter.load_state_dict(checkpoint['adapter_state'])
            self.backbone.head.load_state_dict(checkpoint['backbone_head_state'])
            print("Adapter loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load adapter: {e}")
        
        self.adapter.eval()
        
        # Preprocessing
        self.preprocess = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Store template features
        self.template_features = None  # [K, D]
        self.num_templates = 0
        self.eps_norm = 1e-8
    
    @staticmethod
    def _create_mask_from_black_bg(image_np, threshold=10):
        """Create binary mask by detecting non-black regions"""
        if len(image_np.shape) == 2:
            image_np = np.expand_dims(image_np, axis=2)
        
        H, W = image_np.shape[:2]
        corner_size = min(50, min(H, W) // 4)
        
        corners = [
            image_np[:corner_size, :corner_size],
            image_np[:corner_size, -corner_size:],
            image_np[-corner_size:, :corner_size],
            image_np[-corner_size:, -corner_size:]
        ]
        
        corner_colors = np.concatenate([c.reshape(-1, image_np.shape[2]) for c in corners], axis=0)
        bg_color = np.mean(corner_colors, axis=0)
        
        diff = np.abs(image_np.astype(float) - bg_color.astype(float))
        max_channel = np.max(image_np, axis=2)
        color_diff = np.max(diff, axis=2)
        
        is_bg_color_similar = color_diff < threshold
        is_very_dark = max_channel < threshold
        
        mask = ~(is_bg_color_similar | is_very_dark)
        
        return mask.astype(np.float32)
    
    @staticmethod
    def _apply_ffa(features, mask):
        import torch.nn.functional as F
        
        B, C, H, W = features.shape
        
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        mask_resized = F.interpolate(
            mask.float(), 
            size=(H, W), 
            mode='nearest'
        )
        
        masked_features = features * mask_resized
        sum_features = masked_features.sum(dim=(2, 3))
        valid_count = mask_resized.sum(dim=(2, 3)).clamp(min=1e-6)
        pooled = sum_features / valid_count
        return pooled
    
    @staticmethod
    def _resize_mask(mask, crop_shape, feature_shape):
        from scipy import ndimage
        
        H_crop, W_crop = crop_shape
        H_feat, W_feat = feature_shape
        
        if H_feat != H_crop or W_feat != W_crop:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((W_feat, H_feat), Image.NEAREST)
            mask = np.array(mask_pil).astype(np.float32) / 255.0
        
        return mask
    
    def load_template_images(self, template_paths):
        """Load and extract features from template images"""
        print(f"\nLoading {len(template_paths)} template images...")
        
        template_features_list = []
        
        for i, template_path in enumerate(template_paths):
            try:
                template_img = Image.open(template_path).convert('RGB')
                template_np = np.array(template_img)
                
                template_tensor = self.preprocess(template_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    backbone_features_map = self.backbone.backbone.forward_features(template_tensor)
                    
                    mask = self._create_mask_from_black_bg(template_np)
                    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    backbone_features_ffa = self._apply_ffa(backbone_features_map, mask_tensor)
                    backbone_feat = self.backbone.head(backbone_features_ffa)
                    backbone_feat = backbone_feat / (backbone_feat.norm(dim=-1, keepdim=True) + self.eps_norm)
                    
                    adapted_feat = self.adapter(backbone_feat)
                    adapted_feat = adapted_feat / (adapted_feat.norm(dim=-1, keepdim=True) + self.eps_norm)
                
                template_features_list.append(adapted_feat)  # Keep on GPU
                print(f"  Template {i+1}: feature_dim={adapted_feat.shape[1]}")
            
            except Exception as e:
                print(f"  Failed to load template {i+1}: {e}")
                import traceback
                traceback.print_exc()
        
        if template_features_list:
            self.template_features = torch.cat(template_features_list, dim=0).to(self.device)  # [K, D]
            self.num_templates = len(template_features_list)
            print(f"Template features shape: {self.template_features.shape}")
        else:
            raise RuntimeError("No templates loaded successfully")
    
    def extract_proposal_features(self, crop_pil, crop_img_np, mask):
        crop_tensor = self.preprocess(crop_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            backbone_features_map = self.backbone.backbone.forward_features(crop_tensor)
            
            mask_resized = self._resize_mask(mask, crop_img_np.shape[:2], backbone_features_map.shape[2:])
            mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            backbone_features_ffa = self._apply_ffa(backbone_features_map, mask_tensor)
            backbone_feat = self.backbone.head(backbone_features_ffa)
            backbone_feat = backbone_feat / (backbone_feat.norm(dim=-1, keepdim=True) + self.eps_norm)
            
            adapted_feat = self.adapter(backbone_feat)
            adapted_feat = adapted_feat / (adapted_feat.norm(dim=-1, keepdim=True) + self.eps_norm)
        
        return adapted_feat.squeeze(0)  # [D]
    
    def compute_matching_score(self, proposal_features):
        template_features = self.template_features  # Already on GPU
        similarities = torch.matmul(proposal_features.unsqueeze(0), template_features.T)
        similarities = similarities.squeeze(0)
        matching_score, best_template_idx = torch.max(similarities, dim=0)
        return float(matching_score.item()), int(best_template_idx.item())
    
    @staticmethod
    def compute_appearance_bonus(proposal_features, template_features):
        """
        PAPER'S APPEARANCE MATCHING (Eq. 7)
        Auto-detects device from proposal_features
        """
        if template_features is None or proposal_features is None:
            return 0.5

        device = proposal_features.device
        template_feat = template_features.to(device)

        proposal_feat = proposal_features / (proposal_features.norm() + 1e-8)
        template_feat = template_feat / (template_feat.norm(dim=1, keepdim=True) + 1e-8)

        similarities = torch.matmul(template_feat, proposal_feat)
        best_similarity = torch.max(similarities).item()
        mean_similarity = torch.mean(similarities).item()

        appearance_bonus = 0.7 * best_similarity + 0.3 * mean_similarity
        return float(np.clip(appearance_bonus, 0, 1))
    
    @staticmethod
    def compute_uav_size_penalty(bbox, frame_area, max_aspect_ratio=2.0):
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        bbox_area = w * h
        area_ratio = bbox_area / frame_area

        if area_ratio < 0.0001:
            area_score = 0.6
        elif area_ratio <= 0.001:
            area_score = 1.0
        elif area_ratio <= 0.01:
            area_score = 0.8
        elif area_ratio <= 0.05:
            area_score = 0.6
        else:
            area_score = 0.1

        aspect_ratio = max(w / h, h / w)
        if aspect_ratio <= 1.1:
            aspect_score = 1.0
        elif aspect_ratio <= max_aspect_ratio:
            aspect_score = 1.0 - 0.6 * (aspect_ratio - 1.1) / (max_aspect_ratio - 1.1)
        else:
            aspect_score = max(0.1, 0.4 - 0.3 * (aspect_ratio - max_aspect_ratio))

        final_score = 0.6 * area_score + 0.4 * aspect_score
        return float(final_score)


# ====================== KALMAN TRACKER ======================

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]], dtype=np.float32)

        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]], dtype=np.float32)

        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.R *= 10.

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.x)

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.array([[cx], [cy], [w], [h]], dtype=np.float32)

    @staticmethod
    def convert_x_to_bbox(x):
        cx = x[0, 0]
        cy = x[1, 0]
        w = x[2, 0]
        h = x[3, 0]
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def iou(bb1, bb2):
    x1, y1, x2, y2 = bb1
    x1_, y1_, x2_, y2_ = bb2
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


# ====================== MAIN INFERENCE PIPELINE ======================

def main(video_id="CardboardBox_1"):
    print("\n" + "="*70)
    print(f"CORRECTED & OPTIMIZED INFERENCE - {video_id}")
    print("="*70)
    
    config = Config(video_id=video_id)
    print(f"\nConfiguration:")
    print(f"   Video ID: {video_id}")
    print(f"   Video: {config.video_path}")
    print(f"   Templates: {len(config.template_paths)}")
    print(f"   Device: {config.device}")
    
    if not config.validate():
        return
    
    print("All files found\n")
    
    # Load models
    print("="*70)
    print("Loading Models")
    print("="*70)
    
    similarity_model = SimilarityModel(config)
    similarity_model.load_template_images(config.template_paths)
    
    print("\nLoading MobileSAMv2...")
    ObjAwareModel_instance = ObjectAwareModel(config.yolo_model)
    
    state_dict_sam = torch.load(config.sam_checkpoint, map_location=config.device, weights_only=True)
    
    image_encoder = TinyViT(
        img_size=1024, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0, drop_rate=0., drop_path_rate=0.,
        use_checkpoint=False, mbconv_expand_ratio=4.0,
        local_conv_size=3, layer_lr_decay=0.8
    )
    
    prompt_encoder = PromptEncoder(
        embed_dim=256, image_embedding_size=(64, 64),
        input_image_size=(1024, 1024), mask_in_chans=16
    )
    
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256
    )
    
    mobilesamv2 = Sam(image_encoder=image_encoder, prompt_encoder=prompt_encoder, mask_decoder=mask_decoder)
    mobilesamv2.load_state_dict(state_dict_sam, strict=False)
    mobilesamv2.to(config.device).eval()
    predictor = SamPredictor(mobilesamv2)
    print("MobileSAMv2 loaded")
    
    # Video setup
    print("\n" + "="*70)
    print("Setting Up Video")
    print("="*70)
    
    cap = cv2.VideoCapture(config.video_path)
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(config.start_sec * fps_input)
    end_frame = int(config.end_sec * fps_input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    end_frame = min(end_frame, total_frames_available - 1)
    total_frames = max(0, end_frame - start_frame + 1)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(config.output_video, fourcc, fps_input, (frame_w, frame_h))
    
    print(f"   Input: {config.video_path}")
    print(f"   Resolution: {frame_w}x{frame_h}")
    print(f"   FPS: {fps_input}")
    print(f"   Frames: {total_frames}")
    print(f"   Output: {config.output_video}")
    
    # Tracking setup
    video_name = os.path.basename(config.video_path).split('.')[0]
    predictions = {"video_id": video_name, "predictions": []}
    
    KalmanBoxTracker.count = 0
    trackers = []
    current_image = None
    
    frame_idx = start_frame
    frame_count = 0
    start_time = time.time()
    
    print("\n" + "="*70)
    print("Processing Video")
    print("="*70)
    
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display = frame.copy()
        
        # Detection
        obj_results = ObjAwareModel_instance(
            frame_rgb, device=config.device, retina_masks=False,
            imgsz=640, conf=config.conf_thres, iou=config.iou_thres
        )
        
        detections = []
        if obj_results and len(obj_results[0].boxes) > 0:
            boxes = obj_results[0].boxes.xyxy.cpu().numpy()
            confs = obj_results[0].boxes.conf.cpu().numpy()
            frame_area = frame_w * frame_h
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                area_ratio = (x2 - x1) * (y2 - y1) / frame_area
                if config.min_area_ratio <= area_ratio <= config.max_area_ratio:
                    detections.append((box, conf))
        
        # Set image for SAM
        if current_image is None or not np.array_equal(current_image, frame_rgb):
            predictor.set_image(frame_rgb)
            current_image = frame_rgb.copy()
        
        # Segment + Match
        candidates = []
        frame_area = frame_w * frame_h
        
        for box, conf in detections:
            input_box = np.array(box)
            try:
                masks, scores, _ = predictor.predict(box=input_box, multimask_output=False)
            except:
                continue
            
            if len(masks) == 0:
                continue
            
            mask = masks[0]
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            
            my1, my2 = ys.min(), ys.max()
            mx1, mx2 = xs.min(), xs.max()
            
            black_img = np.zeros_like(frame)
            black_img[mask] = frame[mask]
            crop_img = black_img[my1:my2+1, mx1:mx2+1]
            if crop_img.size == 0:
                continue
            
            crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            crop_mask = mask[my1:my2+1, mx1:mx2+1]
            
            proposal_features = similarity_model.extract_proposal_features(crop_pil, crop_img_rgb, crop_mask)
            
            matching_score, best_template_idx = similarity_model.compute_matching_score(proposal_features)
            
            appearance_bonus = SimilarityModel.compute_appearance_bonus(
                proposal_features, similarity_model.template_features
            )
            
            size_penalty = SimilarityModel.compute_uav_size_penalty(
                [mx1, my1, mx2, my2], frame_area
            )
            
            final_score = (
                0.7 * matching_score +
                0.15 * appearance_bonus +
                0.15 * size_penalty
            )
            
            if final_score >= config.SIM_THRESHOLD:
                candidates.append({
                    "bbox": [mx1, my1, mx2, my2],
                    "matching_score": final_score,
                    "template_match": matching_score,
                    "appearance": appearance_bonus,
                    "size_penalty": size_penalty,
                    "mask": mask,
                    "conf": float(conf),
                    "iou": float(scores[0]),
                    "raw_box": box
                })
        
        # Kalman update
        trks = np.zeros((len(trackers), 4))
        to_del = []
        for t in range(len(trackers)):
            pos = trackers[t].predict()
            trks[t] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            trackers.pop(t)
        trks = trks[[i for i in range(len(trks)) if i not in to_del]]
        
        # Matching
        matched, unmatched_dets, unmatched_trks = [], [], []
        if len(trks) > 0 and len(candidates) > 0:
            cost_matrix = np.zeros((len(trks), len(candidates)), dtype=np.float32)
            for t, trk in enumerate(trks):
                for d, det in enumerate(candidates):
                    cost = -det["matching_score"]
                    iou_val = iou(trk, det["bbox"])
                    if iou_val >= 0.1:
                        cost -= (iou_val * 2.0)
                    else:
                        cost += 10.0
                    cost_matrix[t, d] = cost
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_dets = set()
            matched_trks = set()
            for t, d in zip(row_ind, col_ind):
                if candidates[d]["matching_score"] >= config.SIM_THRESHOLD:
                    matched.append((d, t))
                    matched_dets.add(d)
                    matched_trks.add(t)
            unmatched_dets = [d for d in range(len(candidates)) if d not in matched_dets]
            unmatched_trks = [t for t in range(len(trks)) if t not in matched_trks]
        else:
            unmatched_dets = list(range(len(candidates)))
            unmatched_trks = list(range(len(trks)))
        
        for d, t in matched:
            trackers[t].update(candidates[d]["bbox"])
            candidates[d]["track_id"] = trackers[t].id
            candidates[d]["hit_streak"] = trackers[t].hit_streak
        
        for i in unmatched_dets:
            trk = KalmanBoxTracker(candidates[i]["bbox"])
            trackers.append(trk)
            candidates[i]["track_id"] = trk.id
            candidates[i]["hit_streak"] = 1
        
        active_candidates = [c for c in candidates if c.get("hit_streak", 0) >= config.MIN_HITS]
        if active_candidates:
            best = max(active_candidates, key=lambda x: x["matching_score"] * 0.8 + x["hit_streak"] * 0.2)
            mx1, my1, mx2, my2 = map(int, best["bbox"])
            mask = best["mask"]
            track_id = best["track_id"]
            
            predictions["predictions"].append({
                "frame": int(frame_idx),
                "track_id": int(track_id),
                "bbox": {"x1": mx1, "y1": my1, "x2": mx2, "y2": my2},
                "matching_score": round(best["matching_score"], 4),
                "template_match": round(best["template_match"], 4),
                "appearance": round(best["appearance"], 4),
                "size_penalty": round(best["size_penalty"], 4)
            })
            
            overlay = frame_display.copy()
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (0.7 * overlay[mask_bool] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
            frame_display = overlay
            
            cv2.rectangle(frame_display, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
            y_offset = my1 - 50
            for i, text in enumerate([
                f"ID:{track_id} Match:{best['matching_score']:.3f}",
                f"Temp:{best['template_match']:.3f} App:{best['appearance']:.3f}",
                f"Size:{best['size_penalty']:.3f}"
            ]):
                cv2.putText(frame_display, text, (mx1, y_offset + i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        trackers = [t for t in trackers if t.time_since_update <= config.MAX_AGE]
        out_video.write(frame_display)
        frame_idx += 1
        frame_count += 1
        
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Frame {frame_count}/{total_frames} | FPS: {fps:.2f} | Tracks: {len(predictions['predictions'])}")
    
    cap.release()
    out_video.release()
    
    total_time = time.time() - start_time
    final_fps = frame_count / total_time if total_time > 0 else 0
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETED!")
    print("="*70)
    print(f"   Frames: {frame_count}")
    print(f"   Time: {total_time:.2f}s")
    print(f"   FPS: {final_fps:.2f}")
    print(f"   Detections: {len(predictions['predictions'])}")
    print(f"\nOutput:")
    print(f"   Video: {config.output_video}")
    print(f"   JSON:  {config.output_json}")
    
    os.makedirs(os.path.dirname(config.output_json), exist_ok=True)
    with open(config.output_json, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print("="*70 + "\n")


if __name__ == '__main__':
    video_id = sys.argv[1] if len(sys.argv) > 1 else "CardboardBox_0"
    main(video_id=video_id)