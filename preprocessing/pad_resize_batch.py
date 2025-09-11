import cv2
import os
from pathlib import Path
from tqdm import tqdm
import shutil

# ========== 需要你改這裡 ==========
SRC_ROOT = r"C:\database\tzuchi\20250905\output\20250905\label"                # 母資料夾（底下直接是各子資料夾）
DST_ROOT = r"C:\database\tzuchi\20250905\output\20250905\label_gray_he_244"   # 輸出的母資料夾
TARGET_SIZE = 244
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PAD_VALUE = (0, 0, 0)   # 黑邊；想用白邊可改成 (255, 255, 255)
# ==================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_image(p: Path):
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def pad_and_resize(img, size=TARGET_SIZE, pad_value=PAD_VALUE):
    h, w = img.shape[:2]
    # 等比縮放
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # 置中 padding 成正方形
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left
    result = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=pad_value
    )
    return result

def process_one(src_path: Path, dst_path: Path):
    # 讀灰階
    gray = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return False
    # 直方圖均衡（HE）
    gray_eq = cv2.equalizeHist(gray)
    # 轉 3ch（方便之後用到預訓練分類網路）
    img_3ch = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    # pad + resize 到 TARGET_SIZE
    out = pad_and_resize(img_3ch, size=TARGET_SIZE, pad_value=PAD_VALUE)
    # 輸出
    ensure_dir(dst_path.parent)
    return cv2.imwrite(str(dst_path), out)

def main():
    src_root = Path(SRC_ROOT)
    dst_root = Path(DST_ROOT)

    ensure_dir(dst_root)

    # 直接遞迴抓取所有影像檔
    all_imgs = [p for p in src_root.rglob("*") if is_image(p)]
    if not all_imgs:
        print(f"❌ 在 {SRC_ROOT} 底下找不到影像。請確認路徑與副檔名。")
        return

    # 統計第一層子資料夾張數（方便你確認）
    per_top_count = {}

    print(f"共找到 {len(all_imgs)} 張影像，開始處理...")
    for p in tqdm(all_imgs, ncols=90):
        rel = p.relative_to(src_root)           # 例如 classA/img001.png 或 classA/sub/x.png
        dst_path = dst_root / rel

        ok = process_one(p, dst_path)
        if not ok:
            # 若處理失敗就直接複製原圖，避免漏檔
            ensure_dir(dst_path.parent)
            shutil.copy2(p, dst_path)

        # 統計（以第一層資料夾名為 key）
        top = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        per_top_count[top] = per_top_count.get(top, 0) + 1

    print(f"\n✅ 完成！已輸出到：{dst_root}")
    print("各資料夾統計：")
    for k in sorted(per_top_count):
        print(f"  {k}: {per_top_count[k]}")

if __name__ == "__main__":
    main()
