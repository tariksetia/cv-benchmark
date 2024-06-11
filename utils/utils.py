import torch
from utils.protocols import Detection, ModelDetection

def convert_model_detection(detections: ModelDetection) -> list[Detection]:
    boxes = detections['boxes'].detach().tolist()
    scores = detections['scores'].detach().tolist()
    labels = detections['labels']
    return [
        Detection(
            box=box,
            score=score,
            label=label if isinstance(label,str) else str(int(label.detach()))
        )
        for box, score, label in zip(boxes,scores,labels)
    ]

def get_file_name(base_dir, time, model, file):
    t = "".join(str(time).split("."))
    fname = f"{t}-{model}-{file.replace('/','-')}.json"
    return f"{base_dir}/{fname}"

def get_gpu_name():
    try:
        return torch.cuda.get_device_name()
    except Exception as e:
        return "cpu"