import os
import shutil
import random
from pathlib import Path
from typing import List

# ======= 需要你改這裡 =======
SRC_ROOT = r"C:\database\tzuchi\20250905\output\20250905\label"     # 原始資料夾（每個子資料夾一個類別）
DST_ROOT = r"C:\database\tzuchi\20250905\output\20250905\train_test_split"  # 產出的新資料夾
# ===========================

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
TARGET_PER_CLASS = 100
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

random.seed(SEED)

def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    src = Path(SRC_ROOT)
    dst = Path(DST_ROOT)
    ensure_dir(dst)

    # 找出所有類別資料夾
    class_dirs = [d for d in src.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError("找不到任何類別資料夾，請確認 SRC_ROOT 路徑是否正確。")

    # 建立輸出資料夾結構
    for split in SPLITS:
        ensure_dir(dst / split)

    for cls_dir in sorted(class_dirs):
        cls_name = cls_dir.name
        imgs = list_images(cls_dir)

        if len(imgs) == 0:
            print(f"[WARN] 類別 '{cls_name}' 沒有影像，跳過。")
            continue

        # 隨機抽樣 100 張（不足就有放回抽樣）
        if len(imgs) >= TARGET_PER_CLASS:
            chosen = random.sample(imgs, TARGET_PER_CLASS)
        else:
            # 有放回抽樣
            chosen = [random.choice(imgs) for _ in range(TARGET_PER_CLASS)]

        # 打亂並依比例切分
        random.shuffle(chosen)
        n = len(chosen)
        n_train = int(n * SPLITS["train"])
        n_val   = int(n * SPLITS["val"])
        n_test  = n - n_train - n_val

        split_map = {
            "train": chosen[:n_train],
            "val":   chosen[n_train:n_train+n_val],
            "test":  chosen[n_train+n_val:]
        }

        # 建立類別子資料夾並複製檔案
        for split, files in split_map.items():
            out_dir = dst / split / cls_name
            ensure_dir(out_dir)

            # 為了避免檔名重覆（尤其是有放回抽樣的複製件），重名就加尾綴
            name_count = {}
            for i, src_path in enumerate(files):
                base = src_path.stem
                ext = src_path.suffix.lower()
                # 若這個來源檔名已經複製過，就加上 dup 序號
                key = (base, ext)
                name_count[key] = name_count.get(key, 0) + 1
                if name_count[key] > 1:
                    out_name = f"{base}_dup{name_count[key]-1:04d}{ext}"
                else:
                    out_name = f"{base}{ext}"

                dst_path = out_dir / out_name
                shutil.copy2(src_path, dst_path)

        print(f"[OK] 類別 '{cls_name}' 完成：train {n_train} / val {n_val} / test {n_test}")

    print(f"\n✅ 完成！新的資料集位於：{dst}")

if __name__ == "__main__":
    main()
