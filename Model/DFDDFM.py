# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, CLIPModel, AutoModel
from PIL.Image import Image
import math
import logging

logger = logging.getLogger(__name__)


class ClipFeatureExtractor(nn.Module):
    def __init__(self, device: str, chkpt_dir: str="./pre_trained/OPENAI_CLIP/"):
        """
        params:
            device: str, device to run the model on
            chkpt_dir: str, directory of the pre-trained CLIP model
        """
        super(ClipFeatureExtractor, self).__init__()
        self.clip_vision_model = CLIPModel.from_pretrained(chkpt_dir).vision_model
        self.clip_vision_model.requires_grad_(False)  # Freeze the CLIP model
        # self.clip_vision_model.config.output_hidden_states = True  # Enable output hidden states
        self.clip_vision_model = self.clip_vision_model.to(device)
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
    def __init__(self, device: str, chkpt_dir: str="./pre_trained/DINO_V2/"):
        """
        params:
            device: str, device to run the model on
            chkpt_dir: str, directory of the pre-trained DINO V2 model
        """
        super(Dinov2FeatureExtractor, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(chkpt_dir, use_fast=True)
        self.dino_model = AutoModel.from_pretrained(chkpt_dir)
        self.dino_model.requires_grad_(False)  # Freeze the DINO model
        self.dino_model = self.dino_model.to(device)
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
    def __init__(self, device: str, pre_ds: str = "LVD", chkpt_dir: str="./pre_trained/DINO_V3/"):
        """
        params:
            device: str, device to run the model on
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
        self.dino_model = self.dino_model.to(device)
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


class ClipSVD(nn.Module):
    def __init__(self, device: str, classifier: bool=False, chkpt_dir: str="./pre_trained/OPENAI_CLIP/"):
        """
           params:
               device: str, device to run the model on
               classifier: bool, whether to use the classifier head
               chkpt_dir: str, directory of the pre-trained CLIP model
        """
        super(ClipSVD, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.device = device
        self.__build_svd_clip()
        self.classifier = classifier
        self.head = nn.Linear(1024, 2)

    def __build_svd_clip(self):
        # Load the pre-trained CLIP model
        self.clip_vision_model = CLIPModel.from_pretrained(self.chkpt_dir).vision_model
        self.clip_vision_model = self.replace_svd_residual_to_attn_linear(self.clip_vision_model, 1023)
        self.clip_vision_model = self.clip_vision_model.to(self.device)

    def forward(self, x: List, y):
        pass
    
    # Method to replace nn.Linear modules within attention modules with SVDResidualLinear
    def replace_svd_residual_to_attn_linear(self, model, r):
        for name, module in model.named_children():
            if name == "encoder":
                for _, module1 in module.named_children():
                    for layer_num, clip_encoder_layer in enumerate(module1):
                        for name2, module2 in clip_encoder_layer.named_children():
                            if 'self_attn' == name2:
                                # Replace nn.Linear layers in this module
                                for sub_name, sub_module in module2.named_modules():
                                    if isinstance(sub_module, nn.Linear):
                                        # Get parent module within self_attn
                                        parent_module = module2
                                        # Replace the nn.Linear layer with SVDResidualLinear
                                        setattr(parent_module, sub_name,
                                                replace_svd_residual(layer_num, sub_module, r))

        # After replacing, set requires_grad for residual components
        for param_name, param in model.named_parameters():
            if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        return model


# Function to replace a module with SVDResidualLinear
def replace_svd_residual(layer_num, module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        logger.debug(f"current attention layer: {layer_num}, weight rank: {len(S)}, r value: {r}")

        # Keep top r singular components (main weight)
        U_r = U[:, :r]      # Shape: (out_features, r)
        S_r = S[:r]         # Shape: (r,)
        Vh_r = Vh[:r, :]    # Shape: (r, in_features)

        logger.debug(f"U_r shape: {U_r.size()}; S_r shape: {S_r.size()}; Vh_r shape: {Vh_r.size()}")

        # Reconstruct the main weight (fixed)
        weight_r = U_r @ torch.diag(S_r) @ Vh_r

        # Calculate the frobenius norm of main weight
        new_module.weight_r_fnorm = torch.norm(weight_r.data, p='fro')

        # Set the main weight
        new_module.weight_r.data.copy_(weight_r)

        # Residual components (trainable)
        U_residual = U[:, r:]    # Shape: (out_features, n - r)
        S_residual = S[r:]       # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        logger.debug(f"U_residual shape: {U_residual.size()}; S_residual shape: {S_residual.size()}; Vh_residual shape: {Vh_residual.size()}")

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
            
            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None
            
            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None

        return new_module
    else:
        return module


# SVDResidualLinear module for orthogonal fine-tuning
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude

        # Original weights (fixed)
        self.weight_r = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_r.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_r + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_r

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual

            logger.debug(f"residual_weight is not None: {residual_weight is not None}")
            logger.debug(f"residual_weight shape: {residual_weight.size()}")

            # Total weight is the fixed main weight plus the residual
            weight = self.weight_r + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_r

        return F.linear(x, weight, self.bias)
        
    # def compute_fn_loss(self):
    #     if (self.S_residual is not None):
    #         weight_current = self.weight_r + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
    #         weight_current_fnorm = torch.norm(weight_current, p='fro')
            
    #         loss = weight_current_fnorm ** 2
    #     else:
    #         loss = 0.0
        
    #     return loss