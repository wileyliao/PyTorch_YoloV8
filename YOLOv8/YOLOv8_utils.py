import torch
import cv2

def detect_specific_objects(model, image):
    """
    使用YOLO模型檢測圖像中特定類別的物體。
    param:
        model:預訓練的YOLO檢測模型。
        image:需要檢測的圖像。
    return:
        所有 class_id = 1的物件。
        data structure: (x1, y1, x2, y2, conf, class_id)
    """
    specific_id = 1
    results = model(image)
    boxes = results[0].boxes.data
    specific_object = boxes[boxes[:, 5] == specific_id]

    return specific_object

def map_boxes(boxes, ori_tensor, rez_tensor):
    """
    將檢測到的框映射回原始影像的大小。
    param:
        boxes:檢測到的boxes
        ori_tensor(torch.Tensor): 原始圖像的張量
        rez_tensor(torch.Tensor): 調整大小後的圖像的張量
    return:
        torch.Tensor: 映射後的原始box
        data structure: (x1, y1, x2, y2, conf, class_id)
    """
    ori_height, ori_width = ori_tensor.shape[1:3]
    rez_height, rez_width = rez_tensor.shape[1:3]
    scale_x = ori_width / rez_width
    scale_y = ori_height / rez_height
    mapped_boxes = boxes.clone()
    mapped_boxes[:, [0, 2]] *= scale_x
    mapped_boxes[:, [1, 3]] *= scale_y

    return mapped_boxes

def boxes_with_matrix_position(boxes, image_tensor, rows, cols):
    """
    計算檢測的物件對應影像中的矩陣位置
    param
        boxes: 物件的檢測框
        image_tensor: 影像
        rows: 矩陣row數量
        cols: 矩陣column數量
    return:
        torch.Tensor: 含有矩陣位置訊息的boxes
        data structure: (x1, y1, x2, y2, conf, class_id, row_id, col_id)
    """
    #計算檢測框中心座標
    centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
    # 確定每個區域的寬度和高度
    height, width = image_tensor.shape[1:3]
    cell_width = width / cols
    cell_height = height / rows
    # 計算每個中心點對應的矩陣位置
    col_id = (centers[:, 0] // cell_width).long()  # 使用整數下取
    row_id = (centers[:, 1] // cell_height).long()
    # 將矩陣位置添加到 mapped_boxes 張量中
    positions = torch.stack((row_id, col_id), dim=1).float().to(boxes.device)  # 保持在 GPU 上
    boxes_with_positions = torch.cat((boxes, positions), dim=1)

    return boxes_with_positions

def cut_boxes_from_image(boxes, image_tensor):
    """
    從圖像中裁剪出物件
    param
        boxes: box位置。
        image_tensor (torch.Tensor): 原始圖像的張量表示。
    return:
        含有物件和對應矩陣位置的列表。
    """
    cropped_images = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_image = image_tensor[:, y1:y2, x1:x2]  # 保持在GPU上
        cropped_images.append(cropped_image)

    return cropped_images

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, window_name="Image"):
    """
    在影像上畫出 bounding boxes
    :param image: 原始影像
    :param boxes: bounding box 座標 [[x1, y1, x2, y2], ...]
    :param color: 框線顏色
    :param thickness: 框線粗細
    :param window_name: 顯示視窗名稱
    """
    image_with_boxes = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # 轉為整數
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow(window_name, image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
