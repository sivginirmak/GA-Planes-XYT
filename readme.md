
Project repo for video segmentation task with GA-Planes.

Time interpolation is achieved as the model is trained to predict the middle frame in a sliding window.

Segmentation is done with YOLO-v8 and SAM -- for an example loading a video and exporting segmented frames, see load_video.py. (based on https://labelbox.com/guides/using-metas-segment-anything-sam-model-on-video-with-labelbox-model-assisted-labeling/)

triplane_models.py includes all Tri-planes and GA-planes with addition, concatenation, or multiplication; with convex and semiconvex formulations where applicable. User-specified arguments are parsed to select the models and set training configs. See run.sh for an example.