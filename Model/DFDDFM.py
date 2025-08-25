# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)


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