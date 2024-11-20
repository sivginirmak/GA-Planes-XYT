import ultralytics
from ultralytics import YOLO
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import ipdb

# Cast color to ints
def get_color(color):
  return (int(color[0]), int(color[1]), int(color[2]))

# Get video dimensions
def get_video_dimensions(input_cap):
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  return height, width

# Get output video writer with same dimensions and fps as input video
def get_output_video_writer(input_cap, output_path):
  # Get the video's properties (width, height, FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  # Define the output video file
  output_codec = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
  output_video = cv2.VideoWriter(output_path, output_codec, fps, (width, height))

  return output_video

def check_video_writer(video_writer, output_path):
    if not video_writer.isOpened():
        raise Exception(f"Failed to open video writer for {output_path}")

# Visualize a video frame with bounding boxes, classes and confidence scores
def visualize_detections(frame, boxes, conf_thresholds, class_ids):
    frame_copy = np.copy(frame)
    for idx in range(len(boxes)):
        class_id = int(class_ids[idx])
        conf = float(conf_thresholds[idx])
        x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
        color = colors[class_id]
        label = f"{model.names[class_id]}: {conf:.2f}"
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), get_color(color), 2)
        cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, get_color(color), 2)
    return frame_copy


# Merge masks into a single, multi-colored mask
def merge_masks(masks, class_ids, chosen_class_ids):
  filtered_class_ids = []
  filtered_masks = []
  for idx, cid in enumerate(class_ids):
    if int(cid) in chosen_class_ids:
      filtered_class_ids.append(cid)
      filtered_masks.append(masks[idx])

  merged = (filtered_masks[0][0])
  if len(filtered_masks) == 1:
    return merged.astype(np.uint8)

  for i in range(1, len(filtered_masks)):
    curr_mask = filtered_masks[i][0]
    merged = np.bitwise_or(merged, curr_mask)

  return merged.astype(np.uint8)






model = YOLO("yolov8n.pt") ## path to your YOLO model
# each class id is assigned a different color
colors = np.random.randint(0, 256, size=(len(model.names), 3))
print(model.names)

# Specify which classes you care about. The rest of classes will be filtered out.
chosen_class_ids = [0]

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth" ## path to your SAM ckpt

sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)


VIDEO_PATH = "skateboarding.mp4" ## path to example video
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS of the video: {fps}")

# output_video_boxes = get_output_video_writer(cap, "skateboarding_boxes.mp4")
# check_video_writer(output_video_boxes, "skateboarding_boxes.mp4")
# output_video_masks = get_output_video_writer(cap, "skateboarding_masks.mp4")
# check_video_writer(output_video_masks, "skateboarding_masks.mp4")
maskList = []
timeList = []

# Loop through the frames of the video
frame_num = 0
used_frame_num = 0
while cap.isOpened():
  if frame_num % 30 == 0:
    print("Processing frames", frame_num, "-", frame_num+29)
  ret, frame = cap.read()
  if not ret:
      break
  
  frame_time = frame_num / fps

  # Run frame through YOLOv8 to get detections
  detections = model.predict(frame, conf=0.7) # frame is a numpy array
  
  # Run frame and detections through SAM to get masks
  transformed_boxes = mask_predictor.transform.apply_boxes_torch(detections[0].boxes.xyxy, list(get_video_dimensions(cap)))
  if len(transformed_boxes) == 0:
    print("No boxes found on frame", frame_num)
    frame_num += 1
    continue
  mask_predictor.set_image(frame)
  masks, scores, logits = mask_predictor.predict_torch(
    boxes = transformed_boxes,
    multimask_output=False,
    point_coords=None,
    point_labels=None
  )
  masks = np.array(masks.cpu())
  if masks is None or len(masks) == 0:
    print("No masks found on frame", frame_num)
    frame_num += 1
    continue

  merged_mask = merge_masks(masks, detections[0].boxes.cls, chosen_class_ids)
  maskList.append(merged_mask)
  timeList.append(frame_time)
  
  used_frame_num += 1 # frame num with detections
  frame_num += 1

  # For the purposes of this demo, only look at the first 90 frames
  if used_frame_num > 99:
    break

cap.release()

cv2.destroyAllWindows()

masks = np.array(maskList)
times = np.array(timeList)

np.save("person_times.npy",times)
np.save("person_masks.npy",masks)