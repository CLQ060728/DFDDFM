from abc import ABC, abstractmethod
# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import torch
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)


class BaseSVDDFM(ABC):
    def __init__(self, dfm: bool=False, dfm_num_layers: int=2, dfm_num_mani: int=4,
                 out_feat_type: str="hidden"):
        """
           params:
               dfm: bool, whether to add DFM (Disentangled Fake Manifolds) layers
               dfm_num_layers: int, number of DFM layers
               dfm_num_mani: int, number of orthogonal manifold layers
               out_feat_type: str, type of clip output features ("hidden" or "cls")
        """
        super(BaseSVDDFM, self).__init__()
        
        assert isinstance(dfm, bool), "dfm should be a boolean"
        assert isinstance(dfm_num_layers, int) and dfm_num_layers in [2, 3], "dfm_num_layers should be 2 or 3"
        assert isinstance(dfm_num_mani, int) and 0 < dfm_num_mani <= 8, "dfm_num_mani should be between 1 and 8"
        assert out_feat_type in ["hidden", "cls"], "out_feat_type should be either 'hidden' or 'cls'"

        self.dfm = dfm
        self.orthogonal_manifolds = nn.ModuleList([SingleManifoldLayer(dfm_num_layers) for _ in range(dfm_num_mani)]) if dfm else None
        self.dfm_decoder = SingleManifoldLayer(dfm_num_layers) if dfm else None
        self.out_feat_type = out_feat_type
        self.head = nn.Linear(1024, 2)

    def forward_common(self, x: torch.Tensor, feat_model):
        features = feat_model(x)
        if self.out_feat_type == "hidden":
            features = features.last_hidden_state[:, 1:, :].mean(dim=1) # Average pooling excluding CLS token
        elif self.out_feat_type == "cls":
            features = features.pooler_output 
        else:
            raise ValueError("Invalid output feature type")

        logger.debug(f"feature shape after feat_model: {features.size()}")

        if self.dfm:         
            for manifold in self.orthogonal_manifolds:
                features += manifold(features)
            features = self.dfm_decoder(features)
            return self.head(features)
        else:
            return self.head(features)
    
    # Method to replace nn.Linear modules within attention modules with SVDResidualLinear
    @abstractmethod
    def replace_svd_residual_to_attn_linear(self, model, r):
        pass


class SingleManifoldLayer(nn.Module):
    def __init__(self, dfm_num_layers: int=2):
        """
            params:
                dfm_num_layers: int, number of DFM layers
        """
        super(SingleManifoldLayer, self).__init__()
        if dfm_num_layers == 2:
            self.manifold = nn.Sequential(
                nn.Linear(1024, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 1024, bias=True)
            )
        elif dfm_num_layers == 3:
            self.manifold = nn.Sequential(
                nn.Linear(1024, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 1024, bias=True)
            )
        else:
            raise ValueError("Invalid number of DFM layers")

    def forward(self, x):
        x = self.manifold(x)

        return x


