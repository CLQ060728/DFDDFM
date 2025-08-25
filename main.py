import torch
from torch import nn
import lightning as LTN
import logging


if __name__ == "__main__":
    logging.basicConfig(filename='./logs/main.log', level=logging.INFO)
    # configure logging at the root level of lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler("./logs/ltn_core.log"))