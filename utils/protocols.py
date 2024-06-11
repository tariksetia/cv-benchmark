import json
import time
from typing import TypeAlias, TypedDict
import os
from pydantic import BaseModel, Field
import torch
from datetime import datetime
from loguru import logger

BBox: TypeAlias = list[float]
FrameId: TypeAlias = int

class Detection(BaseModel):
    box: BBox
    score: float
    label: str

Detections = dict[FrameId, list[Detection]]

class FrameDetections(BaseModel):
    result: Detections


class ModelDetection(TypedDict):
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: list[str]


class Experiment(BaseModel):
    model: str
    gpu: str
    batch_size:int = Field(default=1)
    file: str
    frames: list[int] | None = Field(default=None)
    n_frames: int
    processing_time: float | int
    fps: float
    start_time: str
    end_time: str
    filename: str 
    
   
    @property
    def row(self):
        return (
            self.model,
            self.gpu,
            self.file,
            self.batch_size,
            self.n_frames,
            self.processing_time,
            self.fps,
            self.start_time,
            self.end_time,
            self.filename
        )
    
    @property
    def columns(self):
        return ("model", "gpu", "file", "batch_size", "n_frames", "processing_time", "fps", "start_time", "end_time", "result_file")
    
    def log(self):
        logger.info(f"{self.file} | frames={self.n_frames} | delta={self.processing_time} | fps={self.fps}")
        
    def save(self):
        with open(f"{self.filename}", 'w') as f:
            json.dump(self.model_dump(),f, indent=2)
    

        

class GDino(Experiment):
    data: Detections
    prompt: str
    
    @property
    def row(self):
        _row = super().row
        return _row + (self.prompt, self.data)
    
    @property
    def columns(self):
        _columns = super().columns
        return _columns + ("prompt", "data") 
    
    def log(self):
        logger.info(f"{self.file} | frames={self.n_frames} | delta={self.processing_time} | fps={self.fps}")
    

class Retina(Experiment):
    data: Detections
    
    @property
    def row(self):
        _row = super().row
        return _row + (self.data,)
    
    @property
    def columns(self):
        _columns = super().columns
        return _columns + ("data",) 

class OwlVit(GDino):
    pass