# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

from torch.utils.data import Dataset
import torch
import lightning as LTN
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL.Image import Image
from typing import List, Literal, Dict, Tuple
import os, logging, json

logger = logging.getLogger(__name__)


class DFDDFMDataset(Dataset):
    def __init__(self,
                 model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"],
                 img_root_path: str,
                 manifolds_paths: Tuple[str, str],
                 manifolds_labels: Tuple[int, int],
                 manifolds_indices_correspondence: Dict[str, str] = None):
        """
            params:
                model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"] The type of the model to be used. Options are "CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT".
                img_root_path: str The root path to the images.
                manifolds_paths: Tuple[str, str] The paths to the manifolds.
                manifolds_labels: Tuple[int, int] The labels for the image 'manifolds_paths'.
                manifolds_indices_correspondence: Dict[str, str] The correspondence between the indices of the manifolds.
        """
        assert os.path.exists(img_root_path), f"Image root path {img_root_path} does not exist."
        for path in manifolds_paths:
            manifold_path = os.path.join(img_root_path, path)
            assert os.path.exists(manifold_path), f"Manifold path {manifold_path} does not exist."
        assert isinstance(manifolds_paths, tuple) and all(isinstance(x, str) for x in manifolds_paths), f"manifolds_paths should be a tuple of strings, but got {type(manifolds_paths)}."
        assert isinstance(manifolds_labels, tuple) and all(isinstance(x, int) for x in manifolds_labels), f"manifolds_labels should be a tuple of integers, but got {type(manifolds_labels)}."
        assert len(manifolds_paths) == 2, f"manifolds_paths should contain exactly two paths, but got {len(manifolds_paths)}."
        assert manifolds_paths[0] != manifolds_paths[1], f"manifolds_paths should be different."
        assert len(manifolds_labels) == 2, f"manifolds_labels should contain exactly two labels, but got {len(manifolds_labels)}."
        assert model_type in ["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"], f"model_type should be one of ['CLIP', 'DINO_V2', 'DINO_V3_LVD', 'DINO_V3_SAT'], but got {model_type}."
        assert isinstance(manifolds_indices_correspondence, dict), \
            f"manifolds_indices_correspondence should be a dict, but got {type(manifolds_indices_correspondence)}."

        mani_0_imgs_paths = np.array(os.listdir(os.path.join(img_root_path, manifolds_paths[0])))
        mani_1_imgs_paths = np.array(os.listdir(os.path.join(img_root_path, manifolds_paths[1])))
        
        logger.debug(f"{manifolds_paths[0]} images paths sample: {mani_0_imgs_paths[:10]}")
        logger.debug(f"{manifolds_paths[1]} images paths sample: {mani_1_imgs_paths[:10]}")
        logger.debug(f"{manifolds_paths[0]} images paths shape: {mani_0_imgs_paths.shape}")
        logger.debug(f"{manifolds_paths[1]} images paths shape: {mani_1_imgs_paths.shape}")

        assert len(mani_0_imgs_paths) == len(mani_1_imgs_paths) and len(mani_0_imgs_paths) % 2 == 0, \
            f"{manifolds_paths[0]} and {manifolds_paths[1]} images must have the same number of images "\
          + f"(even number), but got {len(mani_0_imgs_paths)} and {len(mani_1_imgs_paths)}."

        pair_indices_length = len(mani_0_imgs_paths) // 2
        mani_0_idx_imgs_dict = {}
        for img_path in mani_0_imgs_paths:
            cls = img_path.split("_")[0]
            if cls not in mani_0_idx_imgs_dict:
                mani_0_idx_imgs_dict[cls] = [img_path]
            else:
                mani_0_idx_imgs_dict[cls].append(img_path)
        mani_1_idx_imgs_dict = {}
        for img_path in mani_1_imgs_paths:
            cls = img_path.split("_")[0]
            if cls not in mani_1_idx_imgs_dict:
                mani_1_idx_imgs_dict[cls] = [img_path]
            else:
                mani_1_idx_imgs_dict[cls].append(img_path)

        logger.debug(f"mani_0_idx_imgs_dict number of keys: {len(mani_0_idx_imgs_dict)}")
        logger.debug(f"mani_1_idx_imgs_dict number of keys: {len(mani_1_idx_imgs_dict)}")

        mani_0_1_pair = []
        mani_1_0_pair = []
        labels_0_1 = []
        labels_1_0 = []
        mani2_0_1 = []
        mani2_1_0 = []
        for key, imgs in mani_0_idx_imgs_dict.items():
            logger.debug(f"class: {key}, images: {imgs}")

            mani_1_imgs = mani_1_idx_imgs_dict[manifolds_indices_correspondence[key]]

            logger.debug(f"corresponding class: {manifolds_indices_correspondence[key]}, images: {mani_1_imgs}")

            mani_0_imgs = np.random.choice(imgs, size=len(imgs), replace=False)
            mani_1_imgs = np.random.choice(mani_1_imgs, size=len(mani_1_imgs), replace=False)

            logger.debug(f"shuffled {key} images: {mani_0_imgs}")
            logger.debug(f"shuffled {manifolds_indices_correspondence[key]} images: {mani_1_imgs}")

            mani_0_1_pair.extend(list(zip(mani_0_imgs[:len(mani_0_imgs)//2], mani_1_imgs[:len(mani_1_imgs)//2])))
            mani_1_0_pair.extend(list(zip(mani_1_imgs[len(mani_1_imgs)//2:], mani_0_imgs[len(mani_0_imgs)//2:])))
            labels_0_1.extend(np.array([manifolds_labels[0], manifolds_labels[1]]).reshape(1, 2).repeat(len(mani_0_imgs)//2, axis=0).tolist())
            labels_1_0.extend(np.array([manifolds_labels[1], manifolds_labels[0]]).reshape(1, 2).repeat(len(mani_1_imgs)//2, axis=0).tolist())
            mani2_0_1.append(np.array([1]).repeat(len(mani_0_imgs)//2, axis=0).tolist())
            mani2_1_0.append(np.array([0]).repeat(len(mani_1_imgs)//2, axis=0).tolist())

            logger.debug(f"mani_0_1_pair: {mani_0_1_pair}")
            logger.debug(f"mani_1_0_pair: {mani_1_0_pair}")
            logger.debug(f"labels_0_1: {labels_0_1}")
            logger.debug(f"labels_1_0: {labels_1_0}")
            logger.debug(f"mani2_0_1: {mani2_0_1}")
            logger.debug(f"mani2_1_0: {mani2_1_0}")

        self.data = np.concatenate([mani_0_1_pair, mani_1_0_pair], axis=0)

        logger.debug(f"Data sample: {self.data}")
        # logger.debug(f"Data sample (first 20 pairs): {self.data[:20]}")
        # logger.debug(f"Data sample (second 20 pairs): {self.data[pair_indices_length:pair_indices_length+20]}")
        logger.debug(f"Total number of pairs: {self.data.shape}")

        data_indices = np.arange(self.data.shape[0])

        logger.debug(f"data_indices before shuffling (first top 100): {data_indices[:100]}")
        logger.debug(f"data_indices before shuffling (second top 100): {data_indices[pair_indices_length:pair_indices_length+100]}")

        np.random.shuffle(data_indices)
        self.data = self.data[data_indices]

        logger.debug(f"data_indices after shuffling (first top 100): {data_indices[:100]}")
        logger.debug(f"data_indices after shuffling (second top 100): {data_indices[pair_indices_length:pair_indices_length+100]}")
        # logger.debug(f"Data sample (first 20 pairs): {self.data[:20]}")
        # logger.debug(f"Data sample (second 20 pairs): {self.data[pair_indices_length:pair_indices_length+20]}")
        logger.debug(f"Data sample after shuffling: {self.data}")
        logger.debug(f"Total number of pairs: {self.data.shape}")

        self.manifold2_indices = np.concatenate([mani2_0_1, mani2_1_0], axis=0).astype(np.int64)

        logger.debug(f"Manifold 2 indices: {self.manifold2_indices}")
        # logger.debug(f"Manifold 2 indices sample (first 20 indices): {self.manifold2_indices[:20]}")
        # logger.debug(f"Manifold 2 indices sample (second 20 indices): {self.manifold2_indices[pair_indices_length:pair_indices_length+20]}")
        logger.debug(f"Total number of manifold 2 indices: {self.manifold2_indices.shape}")

        self.manifold2_indices = self.manifold2_indices[data_indices]
        
        logger.debug(f"Manifold 2 indices after shuffling: {self.manifold2_indices}")
        # logger.debug(f"Manifold 2 indices sample after shuffling (first 20 indices): {self.manifold2_indices[:20]}")
        # logger.debug(f"Manifold 2 indices sample after shuffling (second 20 indices): {self.manifold2_indices[pair_indices_length:pair_indices_length+20]}")
        logger.debug(f"Total number of manifold 2 indices: {self.manifold2_indices.shape}")

        self.labels = np.concatenate([labels_0_1, labels_1_0], axis=0, dtype=np.int64)

        # logger.debug(f"Labels sample (first 10 labels): {self.labels[:10]}")
        # logger.debug(f"Labels sample (second 10 labels): {self.labels[pair_indices_length:pair_indices_length+10]}")
        logger.debug(f"Labels sample:{self.labels}")
        logger.debug(f"Total number of labels: {self.labels.shape}")

        self.labels = self.labels[data_indices]

        logger.debug(f"Labels sample after shuffling: {self.labels}")
        # logger.debug(f"Labels sample (first 10 labels): {self.labels[:10]}")
        # logger.debug(f"Labels sample (second 10 labels): {self.labels[pair_indices_length:pair_indices_length+10]}")
        logger.debug(f"Total number of labels: {self.labels.shape}")

        self.model_type = model_type

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_path_1, img_path_2 = self.data[idx]
        assert img_path_1 is not None and img_path_2 is not None, "Image paths cannot be None"
        assert img_path_1.endswith(".jpg") or img_path_1.endswith(".jpeg") or \
               img_path_1.endswith(".JPEG") or img_path_1.endswith(".png"),\
               "Image path must end with .jpg, .jpeg, .JPEG or .png"
        assert img_path_2.endswith(".jpg") or img_path_2.endswith(".jpeg") or \
               img_path_2.endswith(".JPEG") or img_path_2.endswith(".png"),\
               "Image path must end with .jpg, .jpeg, .JPEG or .png"

        label_1, label_2 = self.labels[idx]
        img_1 = Image.open(img_path_1).convert("RGB")
        img_2 = Image.open(img_path_2).convert("RGB")
        if self.model_type == "CLIP":
            transform = self.transform_img_clip()
        elif self.model_type == "DINO_V2":
            transform = self.transform_img_dinov2()
        elif self.model_type == "DINO_V3_LVD":
            transform = self.transform_img_lvd()
        elif self.model_type == "DINO_V3_SAT":
            transform = self.transform_img_sat()
        img_tensor_1 = transform(img_1)
        img_tensor_2 = transform(img_2)
        label_tensor_1 = torch.tensor(label_1).long()
        label_tensor_2 = torch.tensor(label_2).long()

        return (img_tensor_1, img_tensor_2), torch.tensor(self.manifold2_indices[idx]).long(),\
               (label_tensor_1, label_tensor_2)

    def transform_img_clip(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])

    def transform_img_dinov2(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])

    def transform_img_lvd(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])
    
    def transform_img_sat(self, resize_size: int = 256):
        resize = transforms.Resize((resize_size, resize_size), antialias=True,
                                   interpolation=transforms.InterpolationMode.BICUBIC)
        center_crop = transforms.CenterCrop(224)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143),
        )
        return transforms.Compose([resize, center_crop, to_tensor, normalize])


class DFDDFMTrainDataModule(LTN.LightningDataModule):
    def __init__(self, 
                 model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"],
                 train_dataset_path: str,
                 val_dataset_path: str,
                 test_dataset_path: str,
                 manifolds_paths: List[str],
                 manifolds_labels: List[int],
                 manifolds_indices_path: str = None,
                 batch_size: int = 120,
                 num_workers: int = 2):
        """
        Initialize the DFDDFMTrainDataModule with the given dataset paths.
        Params:
            model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"] The type of the model for the image transformations.
            train_dataset_path: str The path to the training dataset.
            val_dataset_path: str The path to the validation dataset.
            test_dataset_path: str The path to the test dataset.
            manifolds_paths: List[str] The paths to the manifolds images paths.
            manifolds_labels: List[int] The labels for the loaded manifolds images.
            manifolds_indices_path: str The path to the mapping of the corresponding manifolds' indices.
            batch_size: int The batch size for the dataloaders.
            num_workers: int The number of workers for the dataloaders.
        """
        super(DFDDFMTrainDataModule, self).__init__()

        assert not (train_dataset_path is None and val_dataset_path is None and test_dataset_path is None), "At least one of train_dataset_path, val_dataset_path, test_dataset_path must be provided."
        if not train_dataset_path:
            assert train_dataset_path.endswith("TRAIN") or train_dataset_path.endswith("train"), f"Invalid train dataset path {train_dataset_path}"
        if not val_dataset_path:
            assert val_dataset_path.endswith("VAL") or val_dataset_path.endswith("val"), f"Invalid val dataset path {val_dataset_path}"
        if not test_dataset_path:
            assert test_dataset_path.endswith("TEST") or test_dataset_path.endswith("test"), f"Invalid test dataset path {test_dataset_path}"
        assert manifolds_indices_path is not None and os.path.exists(manifolds_indices_path), f"manifolds_indices_path {manifolds_indices_path} does not exist."

        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.model_type = model_type
        self.manifolds_paths = manifolds_paths
        self.manifolds_labels = manifolds_labels
        with open(manifolds_indices_path, "r") as mani_idxes_file:
            self.manifolds_indices_correspondence = json.load(mani_idxes_file, encoding="utf-8")
        self.batch_size = batch_size
        self.num_workers = num_workers



    def setup(self, stage=None):
        if stage == "fit":
            assert self.train_dataset_path is not None, "train_dataset_path must be provided for fit"

            self.train_dataset = DFDDFMDataset(self.model_type, self.train_dataset_path,
                                               self.manifolds_paths, self.manifolds_labels,
                                               self.manifolds_indices_correspondence)
            if self.val_dataset_path is not None:
                self.val_dataset = DFDDFMDataset(self.model_type, self.val_dataset_path,
                                                 self.manifolds_paths, self.manifolds_labels,
                                                 self.manifolds_indices_correspondence)
        # if stage == "validate":
        #     self.val_dataset = DFDDFMDataset(self.model_type, self.val_dataset_path, self.manifolds_paths)
        if stage == "test":
            assert self.test_dataset_path is not None, "test_dataset_path must be provided for test"

            self.test_dataset = DFDDFMDataset(self.model_type, self.test_dataset_path,
                                               self.manifolds_paths, self.manifolds_labels,
                                               self.manifolds_indices_correspondence)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)