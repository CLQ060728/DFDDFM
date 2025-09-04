# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

from torch.utils.data import Dataset
import torch
import lightning as LTN
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL.Image import Image
from typing import List, Literal
import os, logging

logger = logging.getLogger(__name__)


class DFDDFMDataset(Dataset):
    def __init__(self,
                 model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"],
                 img_root_path: str,
                 manifolds_paths: List[str],
                 manifolds_labels: List[int]):
        """
            params:
                model_type: Literal["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"] The type of the model to be used. Options are "CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT".
                img_root_path: str The root path to the images.
                manifolds_paths: List[str] The paths to the manifolds.
                manifolds_labels: List[int] The labels for the image 'manifolds_paths'.
        """
        assert os.path.exists(img_root_path), f"Image root path {img_root_path} does not exist."
        for path in manifolds_paths:
            manifold_path = os.path.join(img_root_path, path)
            assert os.path.exists(manifold_path), f"Manifold path {manifold_path} does not exist."
        assert isinstance(manifolds_paths, List[str]), f"manifolds_paths should be a list of strings, but got {type(manifolds_paths)}."
        assert isinstance(manifolds_labels, List[int]), f"manifolds_labels should be a list of integers, but got {type(manifolds_labels)}."
        assert len(manifolds_paths) == 2, f"manifolds_paths should contain exactly two paths, but got {len(manifolds_paths)}."
        assert manifolds_paths[0] != manifolds_paths[1], f"manifolds_paths should be different."
        assert len(manifolds_labels) == 2, f"manifolds_labels should contain exactly two labels, but got {len(manifolds_labels)}."
        assert model_type in ["CLIP", "DINO_V2", "DINO_V3_LVD", "DINO_V3_SAT"], f"model_type should be one of ['CLIP', 'DINO_V2', 'DINO_V3_LVD', 'DINO_V3_SAT'], but got {model_type}."

        mani_0_imgs_paths = np.array(os.listdir(os.path.join(img_root_path, manifolds_paths[0])))
        mani_1_imgs_paths = np.array(os.listdir(os.path.join(img_root_path, manifolds_paths[1])))

        logger.debug(f"{manifolds_paths[0]} images paths sample: {mani_0_imgs_paths[:5]}")
        logger.debug(f"{manifolds_paths[1]} images paths sample: {mani_1_imgs_paths[:5]}")
        logger.debug(f"{manifolds_paths[0]} images paths shape: {mani_0_imgs_paths.shape}")
        logger.debug(f"{manifolds_paths[1]} images paths shape: {mani_1_imgs_paths.shape}")

        assert len(mani_0_imgs_paths) == len(mani_1_imgs_paths) and len(mani_0_imgs_paths) % 2 == 0, \
            f"{manifolds_paths[0]} and {manifolds_paths[1]} images must have the same number of images "\
          + f"(even number), but got {len(mani_0_imgs_paths)} and {len(mani_1_imgs_paths)}."

        pair_indices_length = len(mani_0_imgs_paths) // 2
        mani_0_random_indices = np.array(torch.multinomial(torch.arange(len(mani_0_imgs_paths)),
                                                len(mani_0_imgs_paths), replacement=False).tolist())
        mani_1_random_indices = np.array(torch.multinomial(torch.arange(len(mani_1_imgs_paths)),
                                                len(mani_1_imgs_paths), replacement=False).tolist())

        logger.debug(f"{manifolds_paths[0]} random indices sample: {mani_0_random_indices[:20]}")
        logger.debug(f"{manifolds_paths[1]} random indices sample: {mani_1_random_indices[:20]}")
        logger.debug(f"{manifolds_paths[0]} random indices shape: {mani_0_random_indices.shape}")
        logger.debug(f"{manifolds_paths[1]} random indices shape: {mani_1_random_indices.shape}")

        mani_0_random_imgs_paths = mani_0_imgs_paths[mani_0_random_indices]
        mani_1_random_imgs_paths = mani_1_imgs_paths[mani_1_random_indices]

        logger.debug(f"{manifolds_paths[0]} random images paths sample: {mani_0_random_imgs_paths[:5]}")
        logger.debug(f"{manifolds_paths[1]} random images paths sample: {mani_1_random_imgs_paths[:5]}")
        logger.debug(f"{manifolds_paths[0]} random images paths shape: {mani_0_random_imgs_paths.shape}")
        logger.debug(f"{manifolds_paths[1]} random images paths shape: {mani_1_random_imgs_paths.shape}")

        mani_0_1_pair = np.array(list(zip(mani_0_random_imgs_paths[:pair_indices_length],
                                    mani_1_random_imgs_paths[:pair_indices_length])))
        mani_1_0_pair = np.array(list(zip(mani_1_random_imgs_paths[pair_indices_length:],
                                    mani_0_random_imgs_paths[pair_indices_length:])))
        self.data = np.concatenate([mani_0_1_pair, mani_1_0_pair], axis=0)

        logger.debug(f"Data sample (first 5 pairs): {self.data[:5]}")
        logger.debug(f"Data sample (second 5 pairs): {self.data[pair_indices_length:pair_indices_length+5]}")
        logger.debug(f"Total number of pairs: {self.data.shape}")

        data_indices = np.arange(self.data.shape[0])
        np.random.shuffle(data_indices)
        self.data = self.data[data_indices]

        logger.debug(f"data_indices after shuffling (first top 10): {data_indices[:10]}")
        logger.debug(f"data_indices after shuffling (second top 10): {data_indices[pair_indices_length:pair_indices_length+10]}")
        logger.debug(f"Data sample (first 5 pairs): {self.data[:5]}")
        logger.debug(f"Data sample (second 5 pairs): {self.data[pair_indices_length:pair_indices_length+5]}")
        logger.debug(f"Total number of pairs: {self.data.shape}")

        manifold2_indices_first = np.ones(pair_indices_length, dtype=np.int64)
        manifold2_indices_second = np.zeros(pair_indices_length, dtype=np.int64)
        self.manifold2_indices = np.concatenate([manifold2_indices_first, manifold2_indices_second], axis=0)

        logger.debug(f"Manifold 2 indices sample (first 5 indices): {self.manifold2_indices[:5]}")
        logger.debug(f"Manifold 2 indices sample (second 5 indices): {self.manifold2_indices[pair_indices_length:pair_indices_length+5]}")
        logger.debug(f"Total number of manifold 2 indices: {self.manifold2_indices.shape}")

        self.manifold2_indices = self.manifold2_indices[data_indices]

        logger.debug(f"Manifold 2 indices sample (first 5 indices): {self.manifold2_indices[:5]}")
        logger.debug(f"Manifold 2 indices sample (second 5 indices): {self.manifold2_indices[pair_indices_length:pair_indices_length+5]}")
        logger.debug(f"Total number of manifold 2 indices: {self.manifold2_indices.shape}")

        labels_first = np.array(manifolds_labels).repeat(pair_indices_length, axis=0)
        labels_second = np.array(manifolds_labels[::-1]).repeat(pair_indices_length, axis=0)
        # labels_first = np.array(list(zip(np.zeros(pair_indices_length, dtype=np.int64), np.ones(pair_indices_length, dtype=np.int64))))
        # labels_second = np.array(list(zip(np.ones(pair_indices_length, dtype=np.int64), np.zeros(pair_indices_length, dtype=np.int64))))
        self.labels = np.concatenate([labels_first, labels_second], axis=0, dtype=np.int64)

        logger.debug(f"Labels sample (first 5 labels): {self.labels[:5]}")
        logger.debug(f"Labels sample (second 5 labels): {self.labels[pair_indices_length:pair_indices_length+5]}")
        logger.debug(f"Total number of labels: {self.labels.shape}")

        self.labels = self.labels[data_indices]

        logger.debug(f"Labels sample (first 5 labels): {self.labels[:5]}")
        logger.debug(f"Labels sample (second 5 labels): {self.labels[pair_indices_length:pair_indices_length+5]}")
        logger.debug(f"Total number of labels: {self.labels.shape}")

        self.model_type = model_type

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_path_1, img_path_2 = self.data[idx]
        assert img_path_1 is not None and img_path_2 is not None, "Image paths cannot be None"
        assert img_path_1.endswith(".jpg") or img_path_1.endswith(".jpeg") or img_path_1.endswith(".png"),\
               "Image path must end with .jpg, .jpeg or .png"
        assert img_path_2.endswith(".jpg") or img_path_2.endswith(".jpeg") or img_path_2.endswith(".png"),\
               "Image path must end with .jpg, .jpeg or .png"

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

        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.model_type = model_type
        self.manifolds_paths = manifolds_paths
        self.manifolds_labels = manifolds_labels
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit":
            assert self.train_dataset_path is not None, "train_dataset_path must be provided for fit"

            self.train_dataset = DFDDFMDataset(self.model_type, self.train_dataset_path,
                                               self.manifolds_paths, self.manifolds_labels)
            if self.val_dataset_path is not None:
                self.val_dataset = DFDDFMDataset(self.model_type, self.val_dataset_path,
                                                 self.manifolds_paths, self.manifolds_labels)
        # if stage == "validate":
        #     self.val_dataset = DFDDFMDataset(self.model_type, self.val_dataset_path, self.manifolds_paths)
        if stage == "test":
            assert self.test_dataset_path is not None, "test_dataset_path must be provided for test"

            self.test_dataset = DFDDFMDataset(self.model_type, self.test_dataset_path,
                                               self.manifolds_paths, self.manifolds_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)