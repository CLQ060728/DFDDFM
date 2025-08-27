from abc import ABC, abstractmethod
# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import torch
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)


class BaseSVDDFM(ABC):
    def __init__(self, device: torch.device, dfm: bool=False, dfm_num_layers: int=2, dfm_num_mani: int=4,
                 dfm_aggr: str ="sum", has_decoder: bool=True, out_feat_type: str="hidden"):
        """
           params:
               device: torch.device, the device to which the model will be moved
               dfm: bool, whether to add DFM (Disentangled Fake Manifolds) layers
               dfm_num_layers: int, number of DFM layers
               dfm_num_mani: int, number of orthogonal manifold layers
               dfm_aggr: str, aggregation method for DFM ("sum" or "concat")
               has_decoder: bool, whether to add a decoder layer
               out_feat_type: str, type of clip output features ("hidden" or "cls")
        """
        super(BaseSVDDFM, self).__init__()

        assert isinstance(device, torch.device), "device should be a torch.device"
        assert isinstance(dfm, bool), "dfm should be a boolean"
        assert isinstance(dfm_num_layers, int) and dfm_num_layers in [2, 3], "dfm_num_layers should be 2 or 3"
        assert isinstance(dfm_num_mani, int) and 0 < dfm_num_mani <= 8, "dfm_num_mani should be between 1 and 8"
        assert isinstance(dfm_aggr, str) and dfm_aggr in ["sum", "concat"], f"dfm_aggr should be either 'sum' or 'concat' but got {dfm_aggr}"
        assert isinstance(has_decoder, bool), "has_decoder should be a boolean"
        assert out_feat_type in ["hidden", "cls"], "out_feat_type should be either 'hidden' or 'cls'"

        self.device = device
        self.dfm = dfm
        self.orthogonal_manifolds = nn.ModuleList([SingleManifoldLayer(dfm_num_layers) for _ in range(dfm_num_mani)]) if dfm else None
        head_hidden_dim = 1024
        if dfm:
            self.dfm_aggr = dfm_aggr
            if dfm_aggr == "sum":
                hidden_dim = 1024
            else: # dfm_aggr == "concat":
                hidden_dim = 1024 * dfm_num_mani
            
            if has_decoder:
                logger.debug(f"hidden_dim and head_hidden_dim when having decoder: {hidden_dim}, {head_hidden_dim}")

                self.dfm_decoder = SingleManifoldLayer(dfm_num_layers, hidden_dim)
            else:
                head_hidden_dim = hidden_dim

                logger.debug(f"hidden_dim and head_hidden_dim when no decoder: {hidden_dim}, {head_hidden_dim}")

                self.dfm_decoder = None
        else:
            self.dfm_decoder = None
        
        self.out_feat_type = out_feat_type
        self.has_decoder = has_decoder
        self.feat_model = None  # To be defined in the subclass
        self.head = nn.Sequential(
            nn.Linear(head_hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 2)
        )

    def forward_common(self, x: torch.Tensor):
        features = self.feat_model(x)

        if self.out_feat_type == "hidden":
            features = features.last_hidden_state[:, 1:, :].mean(dim=1) # Average pooling excluding CLS token
        else:  # self.out_feat_type == "cls"
            features = features.pooler_output  # Use CLS token representation

        logger.debug(f"features shape after feat_model: {features.size()}")

        if self.dfm:
            results_manifolds_features = None
            if self.dfm_aggr == "sum":
                manifolds_features = torch.tensor([]).to(features.device)
                for manifold in self.orthogonal_manifolds:
                    manifold_feature = manifold(features)
                    manifolds_features = torch.cat((manifolds_features, manifold_feature.unsqueeze(0)))  # Concatenate along the new dimension
            
                logger.debug(f"manifold_features shape after orthogonal manifolds: {manifolds_features.size()}")

                features = manifolds_features.sum(dim=0)  # Summation pooling across manifold features
                results_manifolds_features = manifolds_features.clone()

                logger.debug(f"features shape after summation pooling: {features.size()}")
                logger.debug(f"results_manifolds_features shape: {results_manifolds_features.size()}")
            else:  # self.dfm_aggr == "concat":
                manifolds_features = []
                for manifold in self.orthogonal_manifolds:
                    manifold_feature = manifold(features)
                    manifolds_features.append(manifold_feature)

                logger.debug(f"manifold_features list length after orthogonal manifolds: {len(manifolds_features)}")

                features = torch.hstack(manifolds_features) # Concatenate all the manifolds features
                results_manifolds_features = features.clone()
                
                logger.debug(f"features shape after concatenation pooling: {features.size()}")

            if self.has_decoder:
                features = self.dfm_decoder(features)

            logger.debug(f"has decoder: {self.has_decoder}, features shape after decoder: {features.size()}")
            if self.has_decoder:
                logger.debug(f"features shape after decoder: {features.size()};"
                             f"results_manifolds_features shape: {results_manifolds_features.size()}")

            return self.head(features), features, results_manifolds_features
        else:
            return self.head(features), None, None

    def to(self):
        self.head.to(self.device)
        if self.dfm:
            if self.has_decoder:
                self.dfm_decoder.to(self.device)
            for manifold in self.orthogonal_manifolds:
                manifold.to(self.device)

    # Method to replace nn.Linear modules within attention modules with SVDResidualLinear
    @abstractmethod
    def replace_svd_residual_to_attn_linear(self, model, r):
        pass


class SingleManifoldLayer(nn.Module):
    def __init__(self, dfm_num_layers: int=2, hidden_dim: int=1024):
        """
            params:
                dfm_num_layers: int, number of DFM layers
                hidden_dim: int, hidden dimension size
        """
        super(SingleManifoldLayer, self).__init__()
        if dfm_num_layers == 2:
            self.manifold = nn.Sequential(
                nn.Linear(hidden_dim, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 1024, bias=True)
            )
        elif dfm_num_layers == 3:
            self.manifold = nn.Sequential(
                nn.Linear(hidden_dim, 4096, bias=True),
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


