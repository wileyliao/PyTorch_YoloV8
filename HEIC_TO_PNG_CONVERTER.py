import os
from PIL import Image
import pillow_heif

# 設定來源資料夾路徑
input_folder = r"C:\database\tzuchi\20250905採樣"
output_folder = os.path.join(input_folder, "output")

# 如果沒有 output 資料夾就建立
os.makedirs(output_folder, exist_ok=True)

# 遍歷資料夾內所有檔案
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(input_folder, filename)

        # 讀取 HEIC
        heif_file = pillow_heif.read_heif(heic_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw"
        )

        # 轉換檔名與存檔
        base_name = os.path.splitext(filename)[0] + ".png"
        png_path = os.path.join(output_folder, base_name)
        image.save(png_path, "PNG")

        print(f"已轉換: {filename} -> {png_path}")
