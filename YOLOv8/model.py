from ultralytics import YOLO
import multiprocessing
import torch



def model_train():
    # model storage path = my_project/test_run_01
    multiprocessing.freeze_support()
    model = YOLO('models/yolov8n.pt')
    results = model.train(
        data = 'data.yaml',
        imgsz = 256,
        epochs = 200,
        patience = 10,
        batch = 10,
        project = 'my_project',
        name = 'test_run_01'
    )
    return results

def load_model(model_path):
    """
    1. 默認情況下，模型會在 CPU 上運行(model = YOLO(model_path))
    2. 要在GPU上運行則要加上(model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    param
        model_path:模型路徑
    return
        使用的模型
    """
    model = YOLO(model_path)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model