import os
from pathlib import Path
import cv2
from tqdm import tqdm
import shutil

# ===== 修改成你的路徑 =====
SRC_ROOT = r"C:\database\tzuchi\20250905\output\20250905\train_test_split_ori"        # 你目前有 train/ val/ test 的資料夾
DST_ROOT = r"C:\database\tzuchi\20250905\output\20250905\train_test_split_gray_he"     # 產生的第二份資料夾（相同結構）
# =========================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SAVE_3CH = True      # True: 輸出成 3 通道(灰階複製成3ch，利於用預訓練模型)；False: 單通道
USE_CLAHE = False    # 若想用 CLAHE，把這個改成 True（clipLimit 與格子大小可調）

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def he_process(gray):
    # 直方圖均衡
    return cv2.equalizeHist(gray)

def clahe_process(gray, clip=2.0, grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray)

def process_one(src_path: Path, dst_path: Path):
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    # HE 或 CLAHE
    if USE_CLAHE:
        img_eq = clahe_process(img)
    else:
        img_eq = he_process(img)

    # 是否存成 3 通道（灰階複製成 BGR 3ch）
    if SAVE_3CH:
        img_out = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)
    else:
        img_out = img_eq

    # 寫出（覆蓋同名檔案）
    ensure_dir(dst_path.parent)
    ok = cv2.imwrite(str(dst_path), img_out)
    return ok

def main():
    src_root = Path(SRC_ROOT)
    dst_root = Path(DST_ROOT)

    # 複製資料夾骨架（train/val/test 與類別夾路徑會在寫檔時自動建立）
    ensure_dir(dst_root)

    # 遍歷 train/val/test
    splits = [d for d in src_root.iterdir() if d.is_dir() and d.name in {"train","val","test"}]
    if not splits:
        print("找不到 train/val/test，請確認 SRC_ROOT。")
        return

    for split_dir in splits:
        # 類別資料夾
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        for cls_dir in class_dirs:
            imgs = [p for p in cls_dir.rglob("*") if is_image(p)]
            rel_base = cls_dir.relative_to(src_root)
            print(f"[{split_dir.name}/{cls_dir.name}] 處理 {len(imgs)} 張")

            for p in tqdm(imgs, ncols=90):
                rel = p.relative_to(src_root)                    # e.g. train/CLASS/img.jpg
                dst_path = dst_root / rel
                ok = process_one(p, dst_path)
                if not ok:
                    # 若讀不到/寫失敗 → 直接複製原圖過去以保持資料完整
                    ensure_dir(dst_path.parent)
                    shutil.copy2(p, dst_path)

    print(f"\n✅ 完成！已輸出到：{dst_root}")
    print(f"    設定：USE_CLAHE={USE_CLAHE}, SAVE_3CH={SAVE_3CH}")

if __name__ == "__main__":
    main()
