#!/usr/bin/env python3
"""
AeroEyes Streaming Prediction Module for Zalo AI Challenge 2025
Implements streaming interface for drone-based object detection
"""

import sys
import os
import torch
import random
import numpy as np
import json
import cv2
from time import time
from PIL import Image
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# Add code path to system
sys.path.append('/code')

# Import model libraries
try:
    from transformers import AutoImageProcessor, AutoModel
    from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
    from mobilesamv2 import SamPredictor
    from mobilesamv2.modeling import Sam
    from tinyvit.tiny_vit import TinyViT
    from mobilesamv2.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer
except ImportError as e:
    print(f"⚠️ Import Warning: {e}. Ensure 'mobilesamv2' and 'tinyvit' folders are in /code/")

# Fixed seed for reproducibility (as per BTC requirement)
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
print("✅ Environment Ready & Seed Fixed (42)")

# ====================== CONFIG ======================
class StreamingConfig:
    def __init__(self):
        self.data_base = "/code/detect/public_test/public_test/samples"
        self.results_base = "/result"

        # MODEL PATHS
        self.dino_model_id = "./weight/DINO"
        self.sam_checkpoint = './weight/mobile_sam.pt'
        self.yolo_model = './weight/ObjectAwareModel.pt'

        # THRESHOLDS
        self.SCORE_THRESHOLD = 0.50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_thres = 0.25
        self.min_area_ratio = 0.0005
        self.max_area_ratio = 0.15
        
        # TRACKING
        self.IOU_THRESHOLD = 0.3
        self.MAX_AGE = 10
        self.MIN_HITS = 3

# ====================== FEATURE EXTRACTION ======================
class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, model_id, device):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_id, local_files_only=True).to(device)
        self.model.eval()
        self.patch_size = getattr(self.model.config, "patch_size", 14)
        self.device = device

    def forward(self, img_pil):
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state 
        pixel_values = inputs['pixel_values']
        h_grid = pixel_values.shape[2] // self.patch_size
        w_grid = pixel_values.shape[3] // self.patch_size
        num_patches = h_grid * w_grid
        
        patch_tokens = last_hidden_state[:, 1:, :][:, :num_patches, :]
        B, N, C = patch_tokens.shape
        
        patch_tokens = patch_tokens.permute(0, 2, 1).view(B, C, h_grid, w_grid)
        return patch_tokens

class FFAProcessor:
    @staticmethod
    def apply_ffa(feat_map, mask):
        target_size = feat_map.shape[-2:]
        mask_resized = F.interpolate(mask.float(), size=target_size, mode='nearest')
        masked_feat = feat_map * mask_resized
        sum_feat = masked_feat.sum(dim=(2, 3))
        sum_mask = mask_resized.sum(dim=(2, 3)) + 1e-6
        return sum_feat / sum_mask

# ====================== AUTO SEGMENTATION ======================
def segment_background_then_object(image_bgr, mobilesamv2, device):
    """
    Segment nền trước, sau đó lấy inverse để tìm object lớn nhất.
    Returns: mask của object (binary)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    predictor = SamPredictor(mobilesamv2)
    predictor.set_image(image_rgb)
    
    # === SEGMENT NỀN: Dùng 4 góc ảnh làm background prompts ===
    margin = 50
    background_points = np.array([
        [margin, margin],
        [w - margin, margin],
        [margin, h - margin],
        [w - margin, h - margin]
    ])
    background_labels = np.array([1, 1, 1, 1])
    
    masks, scores, _ = predictor.predict(
        point_coords=background_points,
        point_labels=background_labels,
        multimask_output=False
    )
    
    background_mask = masks[0]
    foreground_mask = ~background_mask
    
    # === TÌM CONNECTED COMPONENTS ===
    foreground_uint8 = foreground_mask.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground_uint8, connectivity=8
    )
    
    if num_labels <= 1:
        return None
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component_idx = np.argmax(areas) + 1
    object_mask = (labels == largest_component_idx)
    
    return object_mask

class SimilarityModel:
    def __init__(self, cfg, mobilesamv2=None):
        self.device = cfg.device
        self.extractor = DinoV3FeatureExtractor(cfg.dino_model_id, self.device)
        self.template_features = None
        self.mobilesamv2 = mobilesamv2

    def load_templates(self, img_dir):
        """
        Load templates từ object_images folder.
        Tự động segment mỗi ảnh để tạo mask.
        """
        feats = []
        print("📝 Processing templates từ object_images...")
        
        # Tìm tất cả ảnh jpg/jpeg
        jpg_files = sorted([f for f in os.listdir(img_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg'))])
        
        if not jpg_files:
            print(f"❌ Không tìm thấy ảnh trong {img_dir}")
            return False
        
        for idx, img_file in enumerate(jpg_files[:3], 1):
            img_path = os.path.join(img_dir, img_file)
            print(f"  Processing template {idx}: {img_file}")
            
            try:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f"    ⚠️ Không thể đọc ảnh: {img_file}")
                    continue
                
                # === AUTO SEGMENT ===
                if self.mobilesamv2 is not None:
                    object_mask = segment_background_then_object(img_bgr, self.mobilesamv2, self.device)
                    if object_mask is None:
                        print(f"    ⚠️ Segment thất bại cho {img_file}")
                        continue
                else:
                    print(f"    ⚠️ SAM model chưa load")
                    return False
                
                # === EXTRACT FEATURES ===
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                feat = self.extract_features(img_pil, object_mask.astype(np.float32))
                feats.append(feat)
                print(f"    ✓ Template {idx} encoded")
                
            except Exception as e:
                print(f"    ❌ Lỗi xử lý {img_file}: {e}")
                continue
            
        if not feats:
            print("❌ Không thể load bất kỳ template nào!")
            return False
            
        self.template_features = torch.stack(feats).to(self.device)
        self.template_features = F.normalize(self.template_features, p=2, dim=1)
        print(f"✅ Load thành công {len(feats)} template features")
        return True

    def extract_features(self, img_pil, mask_np):
        m = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        feat_map = self.extractor(img_pil)
        feat = FFAProcessor.apply_ffa(feat_map, m)
        return F.normalize(feat, p=2, dim=1).squeeze(0)

    def compute_scores(self, feat):
        sims = torch.matmul(feat.unsqueeze(0), self.template_features.T).squeeze(0)
        avg = sims.mean().item()
        best = sims.max().item()
        app_bonus = 0.7 * best + 0.3 * avg
        return 0, 0, 0, avg, app_bonus

    @staticmethod
    def size_penalty(bbox, area):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ratio = (w * h) / area
        if ratio < 0.0001:
            s = 0.8
        elif ratio <= 0.001:
            s = 1.0
        elif ratio <= 0.01:
            s = 0.9
        elif ratio <= 0.05:
            s = 0.8
        else:
            s = 0.2
        return s

# ====================== TRACKING UTILITIES ======================
def iou_batch(bb_test, bb_gt):
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]))
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + 
              (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]], dtype=np.float32)
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]], dtype=np.float32)
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.R[2:,2:] *= 10.
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hit_streak = 0
        self.last_score = 0.0

    def update(self, bbox, score=0.0):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.last_score = score

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.x)[0]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        return np.array([x, y, w*h, w/float(h)]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = np.array(linear_sum_assignment(-iou_matrix)).T
    else:
        matched_indices = np.empty((0,2))
    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:,0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:,1]]
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# ====================== STREAMING PREDICTOR CLASS ======================
class StreamingObjectPredictor:
    """
    Streaming object predictor for drone-based detection.
    Implements predict_streaming(frame_rgb_np, frame_idx) interface.
    """
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = StreamingConfig()
        self.cfg = cfg
        self.device = cfg.device
        
        print("⏳ Loading Models...")
        
        # 1. Load SAM trước (cần cho segmentation templates)
        self.sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                                  embed_dims=[64,128,160,320], depths=[2,2,6,2],
                                  num_heads=[2,4,5,10], window_sizes=[7,7,14,7]),
            prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(64,64),
                                         input_image_size=(1024,1024), mask_in_chans=16),
            mask_decoder=MaskDecoder(num_multimask_outputs=3,
                                     transformer=TwoWayTransformer(depth=2, embedding_dim=256, 
                                                                   mlp_dim=2048, num_heads=8),
                                     transformer_dim=256)
        )
        self.sam.load_state_dict(torch.load(cfg.sam_checkpoint, map_location=cfg.device), 
                                 strict=False)
        self.sam.to(cfg.device).eval()
        self.predictor = SamPredictor(self.sam)
        
        # 2. Load DINOv3 Feature Extractor (với SAM)
        self.sim_model = SimilarityModel(cfg, mobilesamv2=self.sam)
        self.templates_loaded = False
        
        # 3. Load YOLO Model
        self.yolo = ObjectAwareModel(cfg.yolo_model)
        
        print("✅ All Models Loaded Successfully!")
        
        # State tracking
        self.trackers = []
        self.frame_height = None
        self.frame_width = None
        self.prev_frame_rgb = None
        
    def initialize_templates(self, template_img_dir):
        """Initialize template features for similarity matching"""
        self.templates_loaded = self.sim_model.load_templates(template_img_dir)
        return self.templates_loaded
    
    def predict_streaming(self, frame_rgb_np, frame_idx):
        """
        Streaming prediction interface for drone deployment.
        
        Args:
            frame_rgb_np: RGB frame as numpy array (H, W, 3)
            frame_idx: Frame index (incrementing)
        
        Returns:
            [x1, y1, x2, y2] if object detected, None otherwise
        """
        if frame_rgb_np is None:
            return None
        
        # Update frame dimensions on first call
        if self.frame_height is None:
            self.frame_height, self.frame_width = frame_rgb_np.shape[:2]
        
        best_obj = None
        
        # Only process if templates are loaded
        if self.templates_loaded:
            H, W = self.frame_height, self.frame_width
            
            # A. Detect candidates using YOLO
            results = self.yolo(frame_rgb_np, conf=self.cfg.conf_thres, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
            
            high_score_candidates = []
            if len(boxes) > 0:
                self.predictor.set_image(frame_rgb_np)
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Filter by area ratio
                    area_ratio = ((x2-x1)*(y2-y1)) / (W*H)
                    if not (self.cfg.min_area_ratio <= area_ratio <= self.cfg.max_area_ratio):
                        continue
                    
                    # SAM Segmentation
                    masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
                    if len(masks) == 0:
                        continue
                    mask = masks[0]
                    
                    # Crop and extract features
                    yy, xx = np.where(mask)
                    if len(yy) == 0:
                        continue
                    y1b, y2b, x1b, x2b = yy.min(), yy.max()+1, xx.min(), xx.max()+1
                    crop = frame_rgb_np[y1b:y2b, x1b:x2b]
                    crop_mask = mask[y1b:y2b, x1b:x2b]
                    
                    # Score computation
                    feat = self.sim_model.extract_features(Image.fromarray(crop), crop_mask)
                    _, _, _, avg_match, app_bonus = self.sim_model.compute_scores(feat)
                    size_pen = self.sim_model.size_penalty([x1b, y1b, x2b, y2b], W * H)
                    
                    final_score = 0.7 * avg_match + 0.25 * app_bonus + 0.05 * size_pen
                    
                    if final_score > self.cfg.SCORE_THRESHOLD:
                        high_score_candidates.append({
                            'bbox': np.array([x1b, y1b, x2b, y2b]),
                            'score': final_score
                        })
            
            # B. Update trackers using SORT
            if len(high_score_candidates) > 0:
                dets = np.array([c['bbox'] for c in high_score_candidates])
            else:
                dets = np.empty((0, 4))
            
            trks = np.zeros((len(self.trackers), 4))
            to_del = []
            for t, trk in enumerate(self.trackers):
                pos = trk.predict()
                trks[t, :] = [pos[0], pos[1], pos[2], pos[3]]
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            for t in reversed(to_del):
                self.trackers.pop(t)

            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
                dets, trks, iou_threshold=self.cfg.IOU_THRESHOLD
            )

            for m in matched:
                self.trackers[m[1]].update(dets[m[0]], score=high_score_candidates[m[0]]['score'])
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i])
                trk.update(dets[i], score=high_score_candidates[i]['score'])
                self.trackers.append(trk)
                
            # C. Select best object
            active_trackers = []
            for trk in self.trackers:
                if (trk.time_since_update < 1) and (trk.hit_streak >= self.cfg.MIN_HITS or frame_idx <= 5):
                    active_trackers.append({
                        "bbox": trk.get_state()[0],
                        "score": trk.last_score,
                        "id": trk.id
                    })
            
            if active_trackers:
                best_obj = max(active_trackers, key=lambda x: x['score'])
        
        # Return format: [x1, y1, x2, y2] or None
        if best_obj:
            bbox = best_obj['bbox'].astype(int)
            return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        return None


# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    import glob
    import time as time_module
    
    print("🚀 Starting Streaming Prediction Pipeline...")
    
    # Initialize predictor
    cfg = StreamingConfig()
    predictor = StreamingObjectPredictor(cfg)
    
    # Find all test cases
    search_path = os.path.join(cfg.data_base, "*")
    test_cases = [os.path.basename(p) for p in glob.glob(search_path) if os.path.isdir(p)]
    print(f"🔎 Found {len(test_cases)} videos: {test_cases}")
    
    all_results = {}
    all_times = []
    
    for video_id in test_cases:
        print(f"\n▶️ Processing: {video_id}")
        
        # Reset tracker for new video
        predictor.trackers = []
        KalmanBoxTracker.count = 0
        
        # Load templates for this video
        template_img_dir = os.path.join(cfg.data_base, video_id, "object_images")
        predictor.initialize_templates(template_img_dir)
        
        # Open video
        video_path = os.path.join(cfg.data_base, video_id, "drone_video.mp4")
        cap = cv2.VideoCapture(video_path)
        
        frame_idx = 0
        video_predictions = []
        t_start = time_module.time()
        
        # Process all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get prediction for this frame
            bbox = predictor.predict_streaming(frame_rgb, frame_idx)
            
            video_predictions.append({
                "frame_id": frame_idx,
                "box": bbox if bbox else [],
                "score": 0.0  # Score not returned in streaming mode
            })
            
            frame_idx += 1
        
        cap.release()
        t_end = time_module.time()
        elapsed_ms = int((t_end - t_start) * 1000)
        
        all_results[video_id] = video_predictions
        all_times.append({"id": video_id, "answer": "processed", "time": elapsed_ms})
        
        print(f"✅ Processed {frame_idx} frames in {elapsed_ms}ms")
    
    # Save results with new schema
    os.makedirs(cfg.results_base, exist_ok=True)
    
    # Convert to new format
    final_results = []
    for video_id, predictions in all_results.items():
        # Extract bbox data from predictions
        bboxes = []
        for pred in predictions:
            if pred['box']:  # Only if box is not empty
                bboxes.append({
                    "frame": pred["frame_id"],
                    "x1": pred["box"][0],
                    "y1": pred["box"][1],
                    "x2": pred["box"][2],
                    "y2": pred["box"][3]
                })
        
        video_result = {
            "video_id": video_id,
            "detections": [{"bboxes": bboxes}] if bboxes else []
        }
        final_results.append(video_result)
    
    output_json = os.path.join(cfg.results_base, "submission.json")
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save time submission
    df_time = pd.DataFrame(all_times)
    time_csv = os.path.join(cfg.results_base, "time_submission.csv")
    df_time.to_csv(time_csv, index=False)
    
    print(f"\n✅ DONE! Results saved:")
    print(f"   - {output_json}")
    print(f"   - {time_csv}")
    print(f"📊 Summary:")
    for item in all_times:
        print(f"  {item['id']}: {item['time']}ms")
