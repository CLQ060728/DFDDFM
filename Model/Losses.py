# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import torch
from DFDDFM import SVDResidualLinear
import logging

logger = logging.getLogger(__name__)


def compute_orthogonal_loss(svd_residual_layer: SVDResidualLinear):
    if svd_residual_layer.S_residual is not None:
        # According to the properties of orthogonal matrices: A^TA = I
        UUT = torch.cat((svd_residual_layer.U_r, svd_residual_layer.U_residual), dim=1) @ torch.cat((svd_residual_layer.U_r, svd_residual_layer.U_residual), dim=1).t()
        VVT = torch.cat((svd_residual_layer.V_r, svd_residual_layer.V_residual), dim=0) @ torch.cat((svd_residual_layer.V_r, svd_residual_layer.V_residual), dim=0).t()
        # print(self.U_r.size(), self.U_residual.size())  # torch.Size([1024, 1023]) torch.Size([1024, 1])
        # print(self.V_r.size(), self.V_residual.size())  # torch.Size([1023, 1024]) torch.Size([1, 1024])
        # UUT = self.U_residual @ self.U_residual.t()
        # VVT = self.V_residual @ self.V_residual.t()
        
        # Construct an identity matrix
        UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
        VVT_identity = torch.eye(VVT.size(0), device=VVT.device)
        
        # Using frobenius norm to compute loss
        loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
    else:
        loss = 0.0
        
    return loss


def compute_keepsv_loss(svd_residual_layer: SVDResidualLinear):
    if (svd_residual_layer.S_residual is not None) and (svd_residual_layer.weight_original_fnorm is not None):
        # Total current weight is the fixed main weight plus the residual
        weight_current = svd_residual_layer.weight_r + svd_residual_layer.U_residual @ torch.diag(svd_residual_layer.S_residual) @ svd_residual_layer.V_residual
        # Frobenius norm of current weight
        weight_current_fnorm = torch.norm(weight_current, p='fro')

        loss = torch.abs(weight_current_fnorm ** 2 - svd_residual_layer.weight_original_fnorm ** 2)
    else:
        loss = 0.0
    
    return loss