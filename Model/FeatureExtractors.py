# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

from typing import List
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, CLIPModel, AutoModel
from PIL.Image import Image
import logging


logger = logging.getLogger(__name__)


class ClipFeatureExtractor(nn.Module):
    def __init__(self, device: torch.device, chkpt_dir: str="./pre_trained/OPENAI_CLIP/"):
        """
        params:
            device: torch.device, device to run the model on
            chkpt_dir: str, directory of the pre-trained CLIP model
        """
        super(ClipFeatureExtractor, self).__init__()
        self.clip_vision_model = CLIPModel.from_pretrained(chkpt_dir).vision_model
        self.clip_vision_model.requires_grad_(False)  # Freeze the CLIP model
        # self.clip_vision_model.config.output_hidden_states = True  # Enable output hidden states
        self.clip_vision_model.to(device)
        self.device = device

    def forward(self, imgs: List[Image] | Image):
        if isinstance(imgs, List):
            imgs = torch.stack([self.transform_img_clip()(img).float() for img in imgs]).to(self.device)
        elif isinstance(imgs, Image):
            imgs = self.transform_img_clip()(imgs).unsqueeze(0).float().to(self.device)
        else:
            raise ValueError("Input should be a list of PIL Images or a single PIL Image.")

        return self.clip_vision_model(imgs)

    def transform_img_clip(self, resize_size: int = 224):
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                            interpolation=transforms.InterpolationMode.BICUBIC)
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        return transforms.Compose([to_tensor, resize, normalize])


class Dinov2FeatureExtractor(nn.Module):
    def __init__(self, device: torch.device, chkpt_dir: str="./pre_trained/DINO_V2/"):
        """
        params:
            device: torch.device, device to run the model on
            chkpt_dir: str, directory of the pre-trained DINO V2 model
        """
        super(Dinov2FeatureExtractor, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(chkpt_dir, use_fast=True)
        self.dino_model = AutoModel.from_pretrained(chkpt_dir)
        self.dino_model.requires_grad_(False)  # Freeze the DINO model
        self.dino_model.to(device)
        self.device = device

    def forward(self, imgs: List[Image] | Image):
        if isinstance(imgs, List):
            imgs = torch.stack([self.transform_img_dino()(img).float() for img in imgs]).to(self.device)
        elif isinstance(imgs, Image):
            imgs = self.transform_img_dino()(imgs).unsqueeze(0).float().to(self.device)
        else:
            raise ValueError("Input should be a list of PIL Images or a single PIL Image.")
        
        inputs = self.processor(images=imgs, return_tensors="pt")
        
        return self.dino_model(**inputs)

    def transform_img_dino(self, resize_size: int = 224):
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                            interpolation=transforms.InterpolationMode.BICUBIC)
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        return transforms.Compose([to_tensor, resize, normalize])


class Dinov3FeatureExtractor(nn.Module):
    def __init__(self, device: torch.device, pre_ds: str = "LVD", chkpt_dir: str="./pre_trained/DINO_V3/"):
        """
        params:
            device: torch.device, device to run the model on
            pre_ds: str, pre-training dataset, defaults to "LVD-1689M", the other is "SAT-493M"
            chkpt_dir: str, directory of the pre-trained DINO V3 model
        """
        super(Dinov3FeatureExtractor, self).__init__()
        if pre_ds == "LVD":
            chkpt_dir = chkpt_dir + "LVD/"
        elif pre_ds == "SAT":
            chkpt_dir = chkpt_dir + "SAT/"
        else:
            raise ValueError("Unknown pre-training dataset.")
        
        self.processor = AutoImageProcessor.from_pretrained(chkpt_dir, use_fast=True)
        self.dino_model = AutoModel.from_pretrained(
            chkpt_dir,
            device_map="auto"
        )
        self.dino_model.requires_grad_(False)  # Freeze the DINO model
        self.dino_model.to(device)
        self.device = device
        self.pre_ds = pre_ds

    def forward(self, imgs: List[Image] | Image):
        if self.pre_ds == "LVD":
            transform_img = self.transform_img_lvd
        elif self.pre_ds == "SAT":
            transform_img = self.make_transform_sat

        if isinstance(imgs, List):
            imgs = torch.stack([transform_img()(img).float() for img in imgs]).to(self.device)
        elif isinstance(imgs, Image):
            imgs = transform_img()(imgs).unsqueeze(0).float().to(self.device)
        else:
            raise ValueError("Input should be a list of PIL Images or a single PIL Image.")

        inputs = self.processor(images=imgs, return_tensors="pt")

        return self.dino_model(**inputs)

    def transform_img_lvd(self, resize_size: int = 224):
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([to_tensor, resize, normalize])
    
    def make_transform_sat(self, resize_size: int = 224):
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        normalize = transforms.Normalize(
            mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143),
        )
        return transforms.Compose([to_tensor, resize, normalize])
