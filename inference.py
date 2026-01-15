import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import matplotlib.pyplot as plt
import glob
import random
import zipfile
import numpy as np

# ==========================================
# 0. 한글 폰트 설정
# ==========================================
FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

def check_font():
    if not os.path.exists(FONT_PATH):
        print("⚠️ 폰트가 없습니다. (로컬 환경이면 폰트 경로를 수정하세요)")
        return False
    return True

if check_font():
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(FONT_PATH)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 설정
# ==========================================
# 주의: 로컬이나 다른 환경에서 실행할 때는 경로를 환경에 맞게 변경해야 합니다.
YOLO_PATH = '/content/drive/MyDrive/멋사 프로젝트 1/2step_detection/detection_11s_try1_best.pt'
CLS_PATH = '/content/drive/MyDrive/멋사 프로젝트 1/2step_detection/best_efficientnet_5class.pth'
ZIP_DATA_DIR = '/content/drive/MyDrive/멋사 프로젝트 1/Validation/01.원천데이터'
OUTPUT_DIR = '/content/drive/MyDrive/멋사 프로젝트 1/Output'

CLASS_NAMES = ['잎끝마름', '정상', '역병', '시들음병', '황화'] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def draw_korean_text(img, text, position, color=(0, 255, 0), font_size=20):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# ==========================================
# 3. 모델 로드 함수
# ==========================================
def load_models():
    print("⏳ Loading YOLO model...")
    try:
        yolo_model = YOLO(YOLO_PATH)
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        return None, None
    
    print("⏳ Loading Classification model...")
    cls_model = models.efficientnet_b0(weights=None)
    in_features = cls_model.classifier[1].in_features
    cls_model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    
    try:
        state_dict = torch.load(CLS_PATH, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(CLS_PATH, map_location=device)
    except Exception as e:
        print(f"❌ Classification 모델 로드 실패: {e}")
        return yolo_model, None

    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    
    cls_model.load_state_dict(state_dict, strict=False)
    cls_model.to(device)
    cls_model.eval()
    return yolo_model, cls_model

# ==========================================
# 4. 추론 파이프라인
# ==========================================
preprocess_cls = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def analyze_image_from_memory(img_rgb, filename, yolo_model, cls_model):
    pil_img = Image.fromarray(img_rgb)
    vis_img = img_rgb.copy()
    results = yolo_model(img_rgb, verbose=False)
    detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
            if (x2 - x1) < 10 or (y2 - y1) < 10: continue

            cropped_img = pil_img.crop((x1, y1, x2, y2))
            input_tensor = preprocess_cls(cropped_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = cls_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                cls_score, cls_idx = torch.max(probs, 1)
                pred_class = CLASS_NAMES[cls_idx.item()]
                pred_conf = cls_score.item()
            
            detections.append((pred_class, pred_conf))
            color = (0, 255, 0) if "정상" in pred_class else (255, 0, 0)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
            label = f"{pred_class} ({pred_conf:.2f})"
            vis_img = draw_korean_text(vis_img, label, (x1, y1 - 25), color=color, font_size=20)

    return vis_img, detections

# ==========================================
# 5. Zip 처리 및 메인
# ==========================================
def get_random_samples_from_zips(base_dir, num_samples=5):
    all_zips = glob.glob(os.path.join(base_dir, "*.zip"))
    if not all_zips: return []
    samples = []
    random.shuffle(all_zips)
    count = 0
    for zip_path in all_zips:
        if count >= num_samples: break
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                image_names = [x for x in z.namelist() if x.lower().endswith(('.jpg', '.png', '.jpeg')) and not x.startswith('__MACOSX')]
                if not image_names: continue
                target_img_name = random.choice(image_names)
                img_data = z.read(target_img_name)
                nparr = np.frombuffer(img_data, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    samples.append({'image': img_rgb, 'filename': target_img_name, 'zipname': os.path.basename(zip_path)})
                    count += 1
        except: continue
    return samples

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 Using device: {device}")
    
    yolo_net, cls_net = load_models()
    if yolo_net is None or cls_net is None: return

    print(f"\n🔍 랜덤 샘플 추출 중...")
    sample_data = get_random_samples_from_zips(ZIP_DATA_DIR, num_samples=5)
    
    if not sample_data:
        print("❌ 분석할 이미지가 없습니다.")
        return

    print(f"📸 {len(sample_data)}장 분석 시작!")
    for i, item in enumerate(sample_data):
        img_rgb = item['image']
        fname = item['filename']
        zname = item['zipname']
        result_img, dets = analyze_image_from_memory(img_rgb, fname, yolo_net, cls_net)
        
        save_name = f"result_{os.path.splitext(zname)[0]}_{os.path.basename(fname)}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"[{i+1}] {zname} -> 저장 완료 (객체 {len(dets)}개)")

if __name__ == "__main__":
    main()
