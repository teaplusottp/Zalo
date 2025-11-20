import cv2  
import numpy as np  
import torch  
import os  
from mobilesamv2 import SamPredictor  
from mobilesamv2.modeling import Sam  
from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel  
from glob import glob  
from tqdm import tqdm  
  
def create_mobilesamv2_model():  
    device = "cuda" if torch.cuda.is_available() else "cpu"  
      
    # Load ObjectAwareModel (YOLO)  
    obj_model_path = './weight/ObjectAwareModel.pt'  
    ObjAwareModel_instance = ObjectAwareModel(obj_model_path)  
      
    # Load MobileSAMv2  
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
      
    return mobilesamv2, ObjAwareModel_instance, device  
  
def segment_background_then_object(image_bgr, mobilesamv2, device):  
    """  
    Segment nền trước, sau đó lấy inverse để tìm object lớn nhất  
    """  
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  
    h, w = image_rgb.shape[:2]  
      
    predictor = SamPredictor(mobilesamv2)  
    predictor.set_image(image_rgb)  
      
    # === SEGMENT NỀN: Dùng 4 góc ảnh làm background prompts ===  
    # Giả định: nền thường ở 4 góc ảnh  
    margin = 50  # pixels từ mép  
    background_points = np.array([  
        [margin, margin],              # Góc trên-trái  
        [w - margin, margin],          # Góc trên-phải  
        [margin, h - margin],          # Góc dưới-trái  
        [w - margin, h - margin]       # Góc dưới-phải  
    ])  
    background_labels = np.array([1, 1, 1, 1])  # Tất cả là foreground (nền)  
      
    # Segment nền  
    masks, scores, _ = predictor.predict(  
        point_coords=background_points,  
        point_labels=background_labels,  
        multimask_output=False  
    )  
      
    background_mask = masks[0]  
      
    # === INVERSE MASK: Lấy foreground (objects) ===  
    foreground_mask = ~background_mask  
      
    # === TÌM CONNECTED COMPONENTS ===  
    # Chuyển sang uint8 cho OpenCV  
    foreground_uint8 = foreground_mask.astype(np.uint8) * 255  
      
    # Tìm các connected components  
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(  
        foreground_uint8, connectivity=8  
    )  
      
    if num_labels <= 1:  # Chỉ có background (label 0)  
        print("[WARN] No foreground objects found")  
        return None  
      
    # === CHỌN COMPONENT LỚN NHẤT (bỏ qua label 0 = background) ===  
    # stats[:, cv2.CC_STAT_AREA] chứa diện tích của mỗi component  
    areas = stats[1:, cv2.CC_STAT_AREA]  # Bỏ label 0  
    largest_component_idx = np.argmax(areas) + 1  # +1 vì đã bỏ label 0  
      
    # Tạo mask cho object lớn nhất  
    object_mask = (labels == largest_component_idx)  
      
    # === TÌM BBOX VÀ CROP ===  
    ys, xs = np.where(object_mask)  
    if len(ys) == 0:  
        print("[WARN] Empty object mask")  
        return None  
      
    my1, my2 = ys.min(), ys.max()  
    mx1, mx2 = xs.min(), xs.max()  
      
    # Crop vùng object  
    cropped_region = image_bgr[my1:my2+1, mx1:mx2+1]  
    cropped_mask = object_mask[my1:my2+1, mx1:mx2+1]  
      
    # Tạo output với nền đen  
    output_image = np.zeros_like(cropped_region)  
    output_image[cropped_mask] = cropped_region[cropped_mask]  
      
    return output_image  
  
  
def process_all_images(samples_dir="./train/samples", output_base="./segment"):  
    mobilesamv2, _, device = create_mobilesamv2_model()  # Không cần YOLO nữa  
    print(f"[INFO] Model loaded on {device.upper()}")  
      
    video_dirs = [d for d in glob(os.path.join(samples_dir, "*")) if os.path.isdir(d)]  
    print(f"[INFO] Tìm thấy {len(video_dirs)} thư mục video_id")  
      
    total = 0  
    for video_dir in tqdm(video_dirs, desc="Videos"):  
        video_id = os.path.basename(video_dir)  
        input_dir = os.path.join(video_dir, "object_images")  
        output_dir = os.path.join(output_base, video_id, "object_images")  
        os.makedirs(output_dir, exist_ok=True)  
          
        images = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.jpeg"))  
        if not images:  
            continue  
          
        for img_path in images:  
            img = cv2.imread(img_path)  
            if img is None:  
                continue  
              
            result = segment_background_then_object(img, mobilesamv2, device)  
            if result is None:  
                continue  
              
            name = os.path.basename(img_path).rsplit(".", 1)[0] + ".png"  
            cv2.imwrite(os.path.join(output_dir, name), result)  
            total += 1  
      
    print(f"\nHOÀN THÀNH! Đã segment {total} ảnh → {output_base}")
  
if __name__ == "__main__":  
    process_all_images()