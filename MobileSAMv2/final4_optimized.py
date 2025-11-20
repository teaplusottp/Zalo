"""
ZALO 2025 - DINOv3 + FFA + SORT TRACKING (Hungarian + Kalman)
✅ Login Added (Fix 401 Error)
✅ Backbone: DINOv3 (ViT)
✅ Feature Extraction: FFA (Feature-Frequency Alignment)
✅ Tracking: SORT (Hungarian Algorithm + Kalman Filter)
✅ Logic: Only track if Final Score > 0.5
"""
import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from datetime import datetime
from torch import nn
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ✅ THÊM ĐOẠN NÀY ĐỂ LOGIN HUGGING FACE
from huggingface_hub import login
# Token của bạn
login(token="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")

# Transformers for DINOv3
from transformers import AutoImageProcessor, AutoModel

# MobileSAMv2
from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import SamPredictor
from mobilesamv2.modeling import Sam
from tinyvit.tiny_vit import TinyViT
from mobilesamv2.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer


# ====================== CONFIG ======================
class Config:
    def __init__(self, video_id="CardboardBox_0", start_time=0.0, end_time=None):
        self.video_id = video_id
        self.start_time = float(start_time)
        self.end_time = float(end_time) if end_time else None

        # PATHS
        self.data_base = r"D:\code\detect\public_test\public_test\samples"
        self.segment_base = r"./segment_objects"
        self.results_base = "results_final_tracking"

        self.video_path = os.path.join(self.data_base, video_id, "drone_video.mp4")
        self.template_img_dir = os.path.join(self.segment_base, video_id, "original_images")
        self.template_mask_dir = os.path.join(self.segment_base, video_id, "mask_images")

        os.makedirs(os.path.join(self.results_base, video_id), exist_ok=True)
        suffix = f"_t{int(self.start_time)}" + (f"-{int(self.end_time)}" if self.end_time else "")
        self.output_video = os.path.join(self.results_base, video_id, f"output_{video_id}{suffix}.mp4")
        self.output_json = self.output_video.replace('.mp4', '.json')

        # MODEL CONFIG
        # Model gây lỗi lúc nãy, giờ đã có login nên sẽ tải được
       # self.dino_model_id = "./weight/DINO" 
        self.dino_model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"  
       
        self.sam_checkpoint = './weight/mobile_sam.pt'
        self.yolo_model = './weight/ObjectAwareModel.pt'

        # THRESHOLDS
        self.SCORE_THRESHOLD = 0.50  # ✅ Hard threshold theo yêu cầu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_thres = 0.25
        self.min_area_ratio = 0.0005
        self.max_area_ratio = 0.15
        
        # TRACKING CONFIG
        self.IOU_THRESHOLD = 0.3
        self.MAX_AGE = 10      # Số frame mất dấu tối đa trước khi xóa track
        self.MIN_HITS = 3      # Số frame liên tiếp cần để confirm track

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_root = os.path.join(self.results_base, video_id, f"debug_{timestamp}")


# ====================== DINOv3 EXTRACTOR ======================
class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, model_id, device):
        super().__init__()
        print(f"⏳ Loading DINO model: {model_id}...")
        self.device = device
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(device)
        except Exception as e:
            print(f"⚠️ Error loading HF model: {e}")
            raise e
        self.model.eval()
        self.patch_size = getattr(self.model.config, "patch_size", 14)
        for param in self.model.parameters(): param.requires_grad = False

    def forward(self, img_pil):
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        pixel_values = inputs['pixel_values']
        h_img, w_img = pixel_values.shape[2], pixel_values.shape[3]
        
        h_grid = h_img // self.patch_size
        w_grid = w_img // self.patch_size
        num_patches = h_grid * w_grid

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        last_hidden_state = outputs.last_hidden_state 
        tokens_no_cls = last_hidden_state[:, 1:, :]
        patch_tokens = tokens_no_cls[:, :num_patches, :]
        
        B, N, C = patch_tokens.shape
        if N != num_patches:
            patch_tokens = patch_tokens.permute(0, 2, 1).view(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
            patch_tokens = F.interpolate(patch_tokens, size=(h_grid, w_grid), mode='bilinear')
            return patch_tokens

        feat_map = patch_tokens.reshape(B, h_grid, w_grid, C).permute(0, 3, 1, 2)
        return feat_map


# ====================== FFA PROCESSOR ======================
class FFAProcessor:
    @staticmethod
    def apply_ffa(feat_map, mask):
        target_size = feat_map.shape[-2:]
        mask_resized = F.interpolate(mask.float(), size=target_size, mode='nearest')
        masked_feat = feat_map * mask_resized
        sum_feat = masked_feat.sum(dim=(2, 3))
        sum_mask = mask_resized.sum(dim=(2, 3)) + 1e-6
        return sum_feat / sum_mask


# ====================== SIMILARITY MODEL ======================
class SimilarityModel:
    def __init__(self, cfg):
        self.device = cfg.device
        self.extractor = DinoV3FeatureExtractor(cfg.dino_model_id, self.device)
        self.template_features = None

    def load_templates(self, img_dir, mask_dir):
        feats = []
        print("📝 Processing templates...")
        print(cfg.yolo_model)
        for i in range(1, 4):
            img_path = os.path.join(img_dir, f"img_{i}.jpg")
            mask_path = os.path.join(mask_dir, f"img_{i}.png")
            if not (os.path.exists(img_path) and os.path.exists(mask_path)): continue
            img = Image.open(img_path).convert('RGB')
            mask = np.array(Image.open(mask_path).convert('L')) > 128
            feat = self.extract_features(img, mask.astype(np.float32))
            feats.append(feat)
            print(f"  ✓ Template {i} encoded.")
            
        if not feats: raise RuntimeError("No template loaded!")
        self.template_features = torch.stack(feats).to(self.device)
        self.template_features = F.normalize(self.template_features, p=2, dim=1)

    def extract_features(self, img_pil, mask_np):
        m = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        feat_map = self.extractor(img_pil)
        feat = FFAProcessor.apply_ffa(feat_map, m)
        return F.normalize(feat, p=2, dim=1).squeeze(0)

    def compute_scores(self, feat):
        sims = torch.matmul(feat.unsqueeze(0), self.template_features.T).squeeze(0)
        s1, s2, s3 = sims.cpu().numpy()
        avg = sims.mean().item()
        best = sims.max().item()
        app_bonus = 0.7 * best + 0.3 * avg
        return s1, s2, s3, avg, app_bonus
    
    @staticmethod
    def size_penalty(bbox, area):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ratio = (w * h) / area
        if ratio < 0.0001: s = 0.8
        elif ratio <= 0.001: s = 1.0
        elif ratio <= 0.01: s = 0.9
        elif ratio <= 0.05: s = 0.8
        else: s = 0.2
        return s


# ====================== TRACKING SYSTEM (SORT) ======================
def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    ✅ FIXED: Handle empty inputs correctly
    """
    # Nếu một trong hai tập rỗng, trả về ma trận rỗng ngay
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
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return o

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], dtype=np.float32)
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], dtype=np.float32)
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.R[2:,2:] *= 10.
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_score = 0.0

    def update(self, bbox, score=0.0):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.last_score = score

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

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
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(matched_indices).T
    else:
        matched_indices = np.empty((0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

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


# ====================== MAIN PIPELINE ======================
def main(object):
    start_time_real = datetime.now()
    cfg = Config(object, start_time=40, end_time=50) # Chỉnh lại start_time nếu cần
    
    if not all(os.path.exists(p) for p in [cfg.video_path, cfg.template_img_dir, cfg.template_mask_dir]):
        print("❌ Missing data!"); return

    # --- Load Models ---
    print("📂 Loading models...")
    sim_model = SimilarityModel(cfg)
    sim_model.load_templates(cfg.template_img_dir, cfg.template_mask_dir)

    yolo = ObjectAwareModel(cfg.yolo_model)
    sam = Sam(image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                                    embed_dims=[64,128,160,320], depths=[2,2,6,2],
                                    num_heads=[2,4,5,10], window_sizes=[7,7,14,7]),
              prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(64,64),
                                           input_image_size=(1024,1024), mask_in_chans=16),
              mask_decoder=MaskDecoder(num_multimask_outputs=3,
                                       transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                                       transformer_dim=256))
    sam.load_state_dict(torch.load(cfg.sam_checkpoint, map_location=cfg.device), strict=False)
    sam.to(cfg.device).eval()
    predictor = SamPredictor(sam)

    # --- Video Setup ---
    cap = cv2.VideoCapture(cfg.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(cfg.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cfg.start_time * fps))
    
    # --- Tracking Variables ---
    frame_idx = int(cfg.start_time * fps)
    frame_count = 0
    
    trackers = [] # List of KalmanBoxTracker objects
    final_predictions = [] # List for JSON output

    print(f"🎬 Processing... Min Score: {cfg.SCORE_THRESHOLD}")
    
    while True:
        ret, frame = cap.read()
        if not ret or (cfg.end_time and frame_idx >= cfg.end_time * fps): break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        disp = frame.copy()
        
        # 1. Detect candidates
        results = yolo(rgb, conf=cfg.conf_thres, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []

        high_score_candidates = []
        
        if len(boxes) > 0:
            predictor.set_image(rgb)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                if not (cfg.min_area_ratio <= (x2-x1)*(y2-y1)/(W*H) <= cfg.max_area_ratio): continue

                masks, _, _ = predictor.predict(box=box, multimask_output=False)
                if len(masks) == 0: continue
                mask = masks[0]
                yy, xx = np.where(mask)
                if len(yy) == 0: continue
                
                y1b, y2b, x1b, x2b = yy.min(), yy.max()+1, xx.min(), xx.max()+1
                crop = rgb[y1b:y2b, x1b:x2b]
                crop_mask = mask[y1b:y2b, x1b:x2b]
                
                # Feature Extraction & Scoring
                feat = sim_model.extract_features(Image.fromarray(crop), crop_mask)
                s1, s2, s3, avg_match, app_bonus = sim_model.compute_scores(feat)
                size_pen = sim_model.size_penalty([x1b, y1b, x2b, y2b], W * H)
                
                final_score = 0.7 * avg_match + 0.25 * app_bonus + 0.05 * size_pen

                # ✅ LOGIC: Chỉ giữ candidate có score > 0.5
                if final_score > cfg.SCORE_THRESHOLD:
                    high_score_candidates.append({
                        'bbox': np.array([x1b, y1b, x2b, y2b]), 
                        'score': final_score, 
                        'details': {'s1':s1, 's2':s2, 's3':s3, 'avg':avg_match, 'bonus':app_bonus}
                    })

        # 2. SORT Tracking Logic (FIXED)
        # ✅ FIX: Xử lý trường hợp không có candidate nào
        if len(high_score_candidates) > 0:
            dets_for_track = np.array([c['bbox'] for c in high_score_candidates])
        else:
            dets_for_track = np.empty((0, 4))
            
        trks_for_track = np.zeros((len(trackers), 4))
        
        to_del = []
        for t, trk in enumerate(trackers):
            pos = trk.predict()[0]
            trks_for_track[t, :] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)): to_del.append(t)
        
        for t in reversed(to_del): trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_for_track, trks_for_track, iou_threshold=cfg.IOU_THRESHOLD)

        for m in matched:
            trackers[m[1]].update(dets_for_track[m[0]], score=high_score_candidates[m[0]]['score'])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets_for_track[i])
            trk.update(dets_for_track[i], score=high_score_candidates[i]['score'])
            trackers.append(trk)

        # Manage Trackers & Visualize
        i = len(trackers)
        active_trackers = []
        for trk in reversed(trackers):
            i -= 1
            if trk.time_since_update > cfg.MAX_AGE:
                trackers.pop(i)
            else:
                # Visualize & Save ONLY if hit streak is sufficient or just updated
                if (trk.time_since_update < 1) and (trk.hit_streak >= cfg.MIN_HITS or frame_idx <= cfg.start_time*fps + 5):
                    bbox = trk.get_state()[0]
                    active_trackers.append({
                        "bbox": bbox.astype(int).tolist(),
                        "score": trk.last_score,
                        "id": trk.id
                    })

        # 3. Choose BEST Tracker for JSON (Single Target)
        best_obj = None
        if active_trackers:
            best_obj = max(active_trackers, key=lambda x: x['score'])
            
            # Draw Best Object
            x1, y1, x2, y2 = best_obj['bbox']
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(disp, f"ID:{best_obj['id']} S:{best_obj['score']:.3f}", (x1, y1-10), 0, 0.7, (0, 255, 0), 2)

        # 4. Save JSON format
        frame_res = {
            "frame_id": frame_idx,
            "box": best_obj['bbox'] if best_obj else [],
            "score": float(best_obj['score']) if best_obj else 0.0,
            "track_id": int(best_obj['id']) if best_obj else -1
        }
        final_predictions.append(frame_res)

        if not active_trackers:
            cv2.putText(disp, "NO TARGET (>0.5)", (50, 100), 0, 1.5, (0, 0, 255), 3)

        out.write(disp)
        if frame_idx % 30 == 0:
            status = f"ID:{best_obj['id']} S:{best_obj['score']:.3f}" if best_obj else "Searching..."
            print(f"Frame {frame_idx} | {status}")
        
        frame_idx += 1
        frame_count += 1

    cap.release()
    out.release()
    
    # ✅ WRITE JSON
    json_output = {
        "video_id": cfg.video_id,
        "predictions": final_predictions
    }
    with open(cfg.output_json, 'w') as f:
        json.dump(json_output, f, indent=2)
        
    print(f"\n✅ Done! Results saved to:\n - {cfg.output_video}\n - {cfg.output_json}")

if __name__ == "__main__":
    main("LifeJacket_0")