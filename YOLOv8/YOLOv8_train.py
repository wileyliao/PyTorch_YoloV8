import os
import yaml
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import multiprocessing
import threading

def class_check():
    model = YOLO('model_v1.pt')
    for class_id, class_name in model.names.items():
        print(f"ID: {class_id}, Class: {class_name}")

class TrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 訓練參數設定")

        # 參數變數
        self.yaml_path_var = tk.StringVar()
        self.project_name_var = tk.StringVar(value='project')
        self.image_size_var = tk.StringVar(value='256')   # 依你前述建議，預設 256
        self.batch_size_var = tk.StringVar(value='32')
        self.epoch_size_var = tk.StringVar(value='150')
        self.early_stop_var = tk.StringVar(value='5')

        # 新增：優化相關
        self.optimizer_var = tk.StringVar(value='AdamW')  # 'auto'/'SGD'/'Adam'/'AdamW'
        self.lr0_var = tk.StringVar(value='0.0015')
        self.momentum_var = tk.StringVar(value='0.9')     # 只對 SGD 有效
        self.weight_decay_var = tk.StringVar(value='0.01')
        self.coslr_var = tk.BooleanVar(value=True)
        self.warmup_epochs_var = tk.StringVar(value='5')

        # （可選）模型選擇
        self.model_path_var = tk.StringVar(value='yolov8s.pt')

        self.status_var = tk.StringVar(value='等待操作...')
        self.build_form()

    def build_form(self):
        padding = {'padx': 10, 'pady': 5}
        row = 0

        # 檔案選擇
        tk.Label(self.root, text="Data YAML 路徑:").grid(row=row, column=0, sticky='w', **padding)
        tk.Entry(self.root, textvariable=self.yaml_path_var, width=50).grid(row=row, column=1, sticky='w', **padding)
        tk.Button(self.root, text="選擇檔案", command=self.select_file).grid(row=row, column=2, sticky='w', **padding)
        row += 1

        # 模型權重
        tk.Label(self.root, text="模型權重:").grid(row=row, column=0, sticky='w', **padding)
        tk.Entry(self.root, textvariable=self.model_path_var, width=50).grid(row=row, column=1, sticky='w', **padding)
        row += 1

        # 基本參數
        for label, var in [
            ("Project Name:", self.project_name_var),
            ("Image Size:", self.image_size_var),
            ("Batch Size:", self.batch_size_var),
            ("Epochs:", self.epoch_size_var),
            ("Early Stop (Patience):", self.early_stop_var),
        ]:
            tk.Label(self.root, text=label).grid(row=row, column=0, sticky='w', **padding)
            tk.Entry(self.root, textvariable=var).grid(row=row, column=1, sticky='w', **padding)
            row += 1

        # Optimizer 區塊
        tk.Label(self.root, text="Optimizer:").grid(row=row, column=0, sticky='w', **padding)
        opt_menu = tk.OptionMenu(self.root, self.optimizer_var, 'auto', 'SGD', 'Adam', 'AdamW')
        opt_menu.grid(row=row, column=1, sticky='w', **padding)
        row += 1

        # LR / Momentum / Weight Decay
        for label, var in [
            ("Initial LR (lr0):", self.lr0_var),
            ("Momentum (SGD):", self.momentum_var),
            ("Weight Decay:", self.weight_decay_var),
            ("Warmup Epochs:", self.warmup_epochs_var),
        ]:
            tk.Label(self.root, text=label).grid(row=row, column=0, sticky='w', **padding)
            tk.Entry(self.root, textvariable=var).grid(row=row, column=1, sticky='w', **padding)
            row += 1

        # Cosine LR
        tk.Checkbutton(self.root, text="使用 Cosine LR 衰減", variable=self.coslr_var).grid(row=row, column=1, sticky='w', **padding)
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
        thread = threading.Thread(target=self.start_training, daemon=True)
        thread.start()

    def start_training(self):
        try:
            self.status_var.set("🚧 訓練中，請稍候...")
            yaml_path = self.yaml_path_var.get()
            if not os.path.exists(yaml_path):
                raise FileNotFoundError("找不到指定的 Data YAML 檔案")

            base_dir = os.path.dirname(yaml_path)
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            # 依你的既有習慣，補 train/val/test 路徑
            data['train'] = os.path.join(base_dir, 'train')
            data['val'] = os.path.join(base_dir, 'valid')
            data['test'] = os.path.join(base_dir, 'test')

            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, sort_keys=False)

            # 讀 UI 參數
            image_size = int(self.image_size_var.get())
            batch_size = int(self.batch_size_var.get())
            epoch_size = int(self.epoch_size_var.get())
            early_stop = int(self.early_stop_var.get())
            project_name = self.project_name_var.get()

            optimizer = self.optimizer_var.get()  # 'auto'/'SGD'/'Adam'/'AdamW'
            lr0 = float(self.lr0_var.get())
            momentum = float(self.momentum_var.get())
            weight_decay = float(self.weight_decay_var.get())
            warmup_epochs = float(self.warmup_epochs_var.get())
            cos_lr = bool(self.coslr_var.get())

            model_path = self.model_path_var.get()
            model = YOLO(model_path)

            # 組裝 train 參數（若 optimizer=auto，則不強制套 lr/momentum）
            train_kwargs = dict(
                data=yaml_path,
                imgsz=image_size,
                epochs=epoch_size,
                patience=early_stop,
                batch=batch_size,
                project=project_name,
                name=f'model_size_{image_size}_batch_{batch_size}',
                cache='ram',
                # 其他常用健全化
                device=0 if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else None,
            )

            # Optimizer 與 LR 相關（手動 or auto）
            train_kwargs['optimizer'] = optimizer  # 即使是 'auto' 也傳，Ultralytics 會處理

            if optimizer != 'auto':
                # 僅在非 auto 時才設置以下，避免被忽略/覆蓋
                train_kwargs.update(dict(
                    lr0=lr0,
                    weight_decay=weight_decay,
                    warmup_epochs=warmup_epochs,
                    cos_lr=cos_lr
                ))
                # 只有 SGD 需要 momentum
                if optimizer.upper() == 'SGD':
                    train_kwargs['momentum'] = momentum

            # 開始訓練
            model.train(**train_kwargs)

            self.status_var.set("✅ 訓練已結束，請查看終端機輸出。")
        except Exception as e:
            self.status_var.set("❌ 訓練失敗")
            messagebox.showerror("錯誤", str(e))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = TrainApp(root)
    root.mainloop()
