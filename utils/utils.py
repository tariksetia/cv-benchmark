import torch
from utils.protocols import Detection, ModelDetection
from datetime import datetime
now = datetime.now()

def convert_model_detection(detections: ModelDetection) -> list[Detection]:
    boxes = detections['boxes'].detach().to('cpu').tolist()
    scores = detections['scores'].detach().to('cpu').tolist()
    labels = detections['labels']
    return [
        Detection(
            box=box,
            score=score,
            label=label if isinstance(label,str) else str(int(label.detach().to('cpu')))
        )
        for box, score, label in zip(boxes,scores,labels)
    ]

def get_file_name(base_dir, time, model, file):
    vid_file = file.split("/")[-1]
    fname=f"exp-{model}-{get_gpu_name()}-{vid_file}-{now.day}-{now.hour}-{now.minute}-{now.second}.csv"
    return f"{base_dir}/{fname}"

def get_gpu_name():
    try:
        gpu= torch.cuda.get_device_name()
        gpu = gpu.split()[:2]
        return "_".join(gpu)
        
    except Exception as e:
        return "cpu"