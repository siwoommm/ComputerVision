import cv2
import numpy as np
from sort import Sort  # https://github.com/abewley/sort 기반
from collections import defaultdict

# IOU 계산 함수
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = box1_area + box2_area - inter_area
    return inter_area / union if union != 0 else 0

# YOLO 모델 로딩
net = cv2.dnn.readNet("yolo-coco/yolov4.weights", "yolo-coco/yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# 클래스 이름 로드
with open("yolo-coco/coco.names", "r") as f:
    classes = f.read().strip().split('\n')

# 비디오 로드
cap = cv2.VideoCapture("input.mp4")

# SORT 추적기
tracker = Sort()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, w_box, h_box = (detection[0:4] * np.array([w, h, w, h])).astype("int")
                x = int(center_x - w_box / 2)
                y = int(center_y - h_box / 2)
                boxes.append([x, y, int(w_box), int(h_box)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    dets = []
    det_classes = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            x1, y1, x2, y2 = x, y, x + w_box, y + h_box
            dets.append([x1, y1, x2, y2, confidences[i]])
            det_classes.append(classes[class_ids[i]])

    # SORT 업데이트
    tracked_objects = tracker.update(np.array(dets))

    # 클래스 매칭
    id_to_label = defaultdict(lambda: "unknown")
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        best_iou = 0
        best_label = "unknown"
        for j, det in enumerate(dets):
            iou = compute_iou([x1, y1, x2, y2], det[:4])
            if iou > best_iou:
                best_iou = iou
                if j < len(det_classes):
                    best_label = det_classes[j]
        id_to_label[obj_id] = best_label

    # 시각화
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        label = id_to_label[obj_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id} | {label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Multi-Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
