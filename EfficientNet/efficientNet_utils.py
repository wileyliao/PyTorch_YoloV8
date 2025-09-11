from __future__ import annotations
import json
from pathlib import Path
import torch

def derive_classes_from_imagefolder(train_dir: Path) -> list[str]:
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"在 {train_dir} 底下找不到任何類別資料夾。")
    return classes

def get_num_classes_from_ckpt(ckpt_path: Path) -> int:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    for k in ("classifier.1.weight", "classifier.1.bias"):
        if k in state:
            return state["classifier.1.weight"].shape[0]
    for k in ("module.classifier.1.weight", "module.classifier.1.bias"):
        if k in state:
            return state["module.classifier.1.weight"].shape[0]
    raise RuntimeError("無法從權重檔推斷輸出類別數。")

def write_class_json(train_dir: Path, ckpt_path: Path) -> Path:
    """
    根據 ckpt_path 推導 JSON 檔名，並寫入類別資訊。
    e.g. checkpoints/xxx.pth -> checkpoints/xxx.json
    """
    out_json = ckpt_path.with_suffix(".json")
    classes = derive_classes_from_imagefolder(train_dir)
    if ckpt_path.exists():
        num_ckpt = get_num_classes_from_ckpt(ckpt_path)
        if len(classes) != num_ckpt:
            raise RuntimeError(
                f"類別數不一致：資料夾={len(classes)}，權重輸出維度={num_ckpt}。"
            )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已寫入 {out_json}，類別順序如下：\n{classes}")
    return out_json

if __name__ == "__main__":
    # 手動測試：改成你的路徑
    TRAIN_DIR = Path(r"C:\path\to\train")
    CKPT_PATH = Path(r"checkpoints\efficientnet_b0_best_244.pth")
    write_class_json(TRAIN_DIR, CKPT_PATH)
