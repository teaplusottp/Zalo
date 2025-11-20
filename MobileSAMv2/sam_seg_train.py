import cv2
import numpy as np
import torch
import os
import json
import threading
import queue
import time
from mobilesamv2 import SamPredictor
from mobilesamv2.modeling import Sam
from collections import defaultdict

# ===================================================================
# DILATE MASK
# ===================================================================
def dilate_mask(mask, kernel_size=21, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
    return dilated.astype(bool)

# ===================================================================
# TẠO MODEL
# ===================================================================
def create_mobilesamv2_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = '/home/qud2hc/Desktop/Test/MobileSAM/MobileSAMv2/weight/mobile_sam.pt'
    state_dict = torch.load(checkpoint_path, map_location=device)

    from tinyvit.tiny_vit import TinyViT
    image_encoder = TinyViT(
        img_size=1024, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4., drop_rate=0., drop_path_rate=0.0,
        use_checkpoint=False, mbconv_expand_ratio=4.0,
        local_conv_size=3, layer_lr_decay=0.8
    )

    from mobilesamv2.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    mobilesamv2 = Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    mobilesamv2.load_state_dict(state_dict, strict=False)
    mobilesamv2.to(device=device)
    mobilesamv2.eval()
    return mobilesamv2, device


# ===================================================================
# SINGLE INFERENCE (AN TOÀN, DỄ DEBUG)
# ===================================================================
def single_inference(frame_rgb, bbox, model, device):  
    """  
    Segment trên toàn bộ frame với bbox làm prompt  
    """  
    predictor = SamPredictor(model)  
    predictor.set_image(frame_rgb)  
      
    # Dùng bbox gốc làm prompt (không cần padding)  
    x1, y1, x2, y2 = bbox  
    input_box = np.array([x1, y1, x2, y2])  
      
    # Sử dụng predict() thay vì manual workflow  
    masks, scores, _ = predictor.predict(  
        point_coords=None,  
        point_labels=None,  
        box=input_box,  
        multimask_output=False  
    )  
      
    return masks[0]  # Trả về mask trên toàn bộ frame  
  
  
def process_video_single(video_path, annotations, output_dir, model, device):
    if not os.path.exists(video_path):
        print(f"[SKIP] Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # Tạo thư mục con
    mask_dir = os.path.join(output_dir, "mask")
    img_dir  = os.path.join(output_dir, "image")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    frame_to_data = defaultdict(list)
    for ann in annotations:
        for bbox in ann['bboxes']:
            frame_num = bbox['frame']
            orig_bbox = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
            frame_to_data[frame_num].append(orig_bbox)

    processed = 0
    for frame_num in sorted(frame_to_data.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bboxes = frame_to_data[frame_num]

        for bbox in bboxes:
            mask = single_inference(frame_rgb, bbox, model, device)

            ys, xs = np.where(mask)
            if len(ys) == 0:
                print(f"[WARN] Frame {frame_num}: Không tìm thấy mask → bỏ qua")
                continue

            # Tính bbox từ mask
            my1, my2 = ys.min(), ys.max()
            mx1, mx2 = xs.min(), xs.max()

            # ====== 1) LƯU MASK (đen-trắng) ======
            cropped_mask = mask[my1:my2+1, mx1:mx2+1]
            mask_uint8 = (cropped_mask.astype(np.uint8) * 255)

            mask_path = os.path.join(mask_dir, f"mask_{frame_num}.png")
            cv2.imwrite(mask_path, mask_uint8)

            # ====== 2) LƯU ẢNH GỐC CROP THEO BBOX MASK ======
            cropped_img = frame_bgr[my1:my2+1, mx1:mx2+1]
            img_path = os.path.join(img_dir, f"image_{frame_num}.png")
            cv2.imwrite(img_path, cropped_img)

            processed += 1
            if processed % 10 == 0:
                print(f"  → Đã lưu {processed} mask + image")

    cap.release()
    print(f"Hoàn thành: {processed} mask + image → {output_dir}")


# ===================================================================
# MAIN
# ===================================================================
def main(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    model, device = create_mobilesamv2_model()
    print(f"[INFO] Model loaded on {device.upper()}")

    for video in data:
        video_id = video['video_id']
        video_path = f"./train/samples/{video_id}/drone_video.mp4"
        output_dir = f"./segment2/{video_id}"
        print(f"\n[START] {video_id} → Single inference (ổn định)")
        start = time.time()

        process_video_single(video_path, video['annotations'], output_dir, model, device)

        print(f"[DONE] Time: {time.time() - start:.2f}s | Output: {output_dir}")

if __name__ == "__main__":
    annotation_file = '/home/qud2hc/Desktop/Test/MobileSAM/MobileSAMv2/train/annotations/annotations.json'
    main(annotation_file)