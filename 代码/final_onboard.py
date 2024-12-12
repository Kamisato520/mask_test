import socket
import cv2
import pickle
import struct
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from rknnlite.api import RKNNLite
import time
# RKNN模型文件路径
rknn_model_path = './360.rknn'

# Anchor配置
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
image_size=100

# 生成Anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

# 推理函数
def inference(image, rknn, conf_thresh=0.1, iou_thresh=0.9, target_shape=(160, 160), draw_result=True):
    height, width, _ = image.shape
    num_blocks = 2
    block_height = height // num_blocks
    block_width = width // num_blocks

    output_info = []
    # 如果图像尺寸大于160x160，则直接返回原始图像
    if width > image_size or height > image_size:
        return output_info, image
    for i in range(num_blocks):
        for j in range(num_blocks):
            y_start = i * block_height
            x_start = j * block_width
            y_end = min((i + 1) * block_height, height)
            x_end = min((j + 1) * block_width, width)
            image_block = image[y_start:y_end, x_start:x_end]

            image_block_resized = cv2.resize(image_block, target_shape)
            image_block_np = image_block_resized / 255.0
            image_block_exp = np.expand_dims(image_block_np, axis=0)
            image_block_transposed = image_block_exp.transpose((0, 3, 1, 2))

            rknn_inputs = image_block_transposed.astype(np.uint8)
            outputs = rknn.inference(inputs=[rknn_inputs])

            if not outputs:
                continue

            y_bboxes_output, y_cls_output = outputs[0].astype(np.float32), outputs[1].astype(np.float32)
            y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
            y_cls = y_cls_output[0]

            bbox_max_scores = np.max(y_cls, axis=1)
            bbox_max_score_classes = np.argmax(y_cls, axis=1)

            keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh)

            for idx in keep_idxs:
                conf = float(bbox_max_scores[idx])
                class_id = bbox_max_score_classes[idx]
                bbox = y_bboxes[idx]
                xmin = max(0, int(bbox[0] * block_width) + x_start)
                ymin = max(0, int(bbox[1] * block_height) + y_start)
                xmax = min(int(bbox[2] * block_width) + x_start, width)
                ymax = min(int(bbox[3] * block_height) + y_start, height)

                if draw_result:
                    color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(image, f"{id2class[class_id]}: {conf:.2f}", (xmin + 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

                output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    return output_info, image

# 初始化RKNN
def init_rknn():
    rknn = RKNNLite(verbose=True)
    rknn.load_rknn(rknn_model_path)
    rknn.init_runtime()
    return rknn

# 主函数
def main():
    rknn = init_rknn()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '192.168.137.234'  # RKNN开发板的IP地址
    port = 9999
    server_socket.bind((host_ip, port))
    server_socket.listen(5)

    print("等待客户端连接...")

    while True:
        client_socket, addr = server_socket.accept()
        print('已获取连接：', addr)

        if client_socket:
            vid = cv2.VideoCapture(0)  # 打开摄像头

            while vid.isOpened():
                img, frame = vid.read()
                if img:
                    _, processed_frame = inference(frame, rknn)
                    # 将帧数据序列化
                    a = pickle.dumps(processed_frame)
                    message = struct.pack("Q", len(a)) + a
                    client_socket.sendall(message)

                    # 添加延迟以降低帧率
                    time.sleep(1)  # 1秒的延迟，可以根据需要进行调整

            vid.release()

if __name__ == "__main__":
    main()
