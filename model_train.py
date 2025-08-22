import os
import yaml
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import multiprocessing
import threading

class TrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 訓練參數設定")

        # 參數變數
        self.yaml_path_var = tk.StringVar()
        self.project_name_var = tk.StringVar(value='project')
        self.image_size_var = tk.StringVar(value='640')
        self.batch_size_var = tk.StringVar(value='4')
        self.epoch_size_var = tk.StringVar(value='400')
        self.early_stop_var = tk.StringVar(value='30')
        self.status_var = tk.StringVar(value='等待操作...')

        self.build_form()

    def build_form(self):
        padding = {'padx': 10, 'pady': 5}
        row = 0

        # 檔案選擇
        tk.Label(self.root, text="YAML 檔案路徑:").grid(row=row, column=0, sticky='w', **padding)
        tk.Entry(self.root, textvariable=self.yaml_path_var, width=50).grid(row=row, column=1, sticky='w', **padding)
        tk.Button(self.root, text="選擇檔案", command=self.select_file).grid(row=row, column=2, sticky='w', **padding)
        row += 1

        # 參數設定
        for label, var in [
            ("Project Name:", self.project_name_var),
            ("Image Size:", self.image_size_var),
            ("Batch Size:", self.batch_size_var),
            ("Epochs:", self.epoch_size_var),
            ("Early Stop (Patience):", self.early_stop_var)
        ]:
            tk.Label(self.root, text=label).grid(row=row, column=0, sticky='w', **padding)
            tk.Entry(self.root, textvariable=var).grid(row=row, column=1, sticky='w', **padding)
            row += 1

        # 訓練按鈕
        tk.Button(self.root, text="開始訓練", command=self.start_training_thread).grid(row=row, column=1, sticky='w', pady=15)
        row += 1

        # 狀態欄
        self.status_label = tk.Label(self.root, textvariable=self.status_var, fg='blue')
        self.status_label.grid(row=row, column=0, columnspan=3, sticky='w', padx=10, pady=(0,10))

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if path:
            self.yaml_path_var.set(path)

    def start_training_thread(self):
        thread = threading.Thread(target=self.start_training)
        thread.start()

    def start_training(self):
        try:
            self.status_var.set("🚧 訓練中，請稍候...")
            yaml_path = self.yaml_path_var.get()
            if not os.path.exists(yaml_path):
                raise FileNotFoundError("找不到指定的 YAML 檔案")

            base_dir = os.path.dirname(yaml_path)
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            data['train'] = os.path.join(base_dir, 'train')
            data['val'] = os.path.join(base_dir, 'valid')
            data['test'] = os.path.join(base_dir, 'test')

            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, sort_keys=False)

            image_size = int(self.image_size_var.get())
            batch_size = int(self.batch_size_var.get())
            epoch_size = int(self.epoch_size_var.get())
            early_stop = int(self.early_stop_var.get())
            project_name = self.project_name_var.get()

            model = YOLO('yolov8n.pt')
            model.train(
                data=yaml_path,
                imgsz=image_size,
                epochs=epoch_size,
                patience=early_stop,
                batch=batch_size,
                project=project_name,
                name=f'model_size_{image_size}_batch_{batch_size}',
                augment=True,
                cache='ram'
            )

            self.status_var.set("✅ 訓練已結束，請查看終端機輸出。")
        except Exception as e:
            self.status_var.set("❌ 訓練失敗")
            messagebox.showerror("錯誤", str(e))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = TrainApp(root)
    root.mainloop()
