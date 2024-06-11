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
    video_file: str
    frames: list[int] | None = Field(default=None)
    n_frames: int
    
    pre_processing_time: float | int
    inference_time: float | int
    post_processing_time: float | int
    video_processing_time: float | int
    
    start_time: str
    end_time: str
    
    record_file: str 
    
    @property
    def pre_processing_fps(self):
        return self.n_frames/self.pre_processing_time

    @property
    def inference_fps(self):
        return self.n_frames/self.inference_time
    
    @property
    def post_processing_fps(self):
        return self.n_frames/self.post_processing_time
    
    @property
    def video_fps(self):
        return self.n_frames/self.video_processing_time
    
    @property
    def row(self):
        return (
            self.model,
            self.gpu,
            self.video_file,
            self.batch_size,
            self.n_frames,
            
            self.pre_processing_fps,
            self.inference_fps,
            self.post_processing_fps,
            self.video_fps,
            
            self.pre_processing_time,
            self.inference_time,
            self.post_processing_time,
            self.video_processing_time,
    
            self.start_time,
            self.end_time,
            self.record_file
        )
    
    @property
    def columns(self):
        return (
            "model", 
            "gpu", 
            "video_file", 
            "batch_size", 
            "n_frames", 
            
            "pre_processing_fps",
            "inference_fps",
            "post_processing_fps",
            "video_fps",
            
            "pre_processing_time",
            "inference_time",
            "post_processing_time",
            "video_processing_time",

            "start_time",
            "end_time",
            "record_file",
        )
    
    def log(self):
        logger.info(f"{self.video_file} | frames={self.n_frames} | model_fps={self.inference_fps} | inference_time={self.inference_time} | preprocess_time={self.pre_processing_time}")
        
    def save(self):
        with open(f"{self.record_file}", 'w') as f:
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

class Sample(BaseModel):
    data: Detections