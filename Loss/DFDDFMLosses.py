# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Literal
from Model.DFDDFM import SVDResidualLinear
from BaseLoss import BaseLoss
import logging

logger = logging.getLogger(__name__)


class ReconstructionLoss(BaseLoss):
    """
    A class representing a reconstruction loss function.

    Args:
        loss_type (Literal["l1", "l2", "smooth_l1"]): The type of loss function to use.
        coef (float): The coefficient to multiply the loss by.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        loss_type (Literal["l1", "l2", "smooth_l1"]): The type of loss function being used.
        loss_fn (function): The loss function being used.
        name (str): The name of the loss function.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "l2", "smooth_l1"] = "l2",
        coef: float = 1.0,
        name="reconstruction",
        reduction="mean",
        check_nan=False
    ):
        super(ReconstructionLoss, self).__init__(coef, check_nan, reduction)
        self.loss_type = loss_type
        if self.loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif self.loss_type == "l2":
            self.loss_fn = F.mse_loss
        elif self.loss_type == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError(f"Unknown loss type: {loss_type}")
        self.name = f"{name}_loss_{self.loss_type}"

    def __call__(
        self,
        predicted: Tensor,
        gt: Tensor,
        name: str = None
    ):
        """
        Compute the loss between the predicted and ground truth values.

        Args:
            predicted (Tensor): The predicted values.
            gt (Tensor): The ground truth values.
            name (str): The name of the loss function.

        Returns:
            dict: Dictionary containing the reconstruction loss value.
        """
        gt, predicted = gt.squeeze(), predicted.squeeze()
        loss = self.loss_fn(predicted, gt, reduction="none")
        name = self.name if name is None else name

        return self.return_loss(name, loss)


class SVDLoss(BaseLoss):
    """
    A class representing singular value decomposition (SVD)-related losses function.

    Args:
        coef (float): The coefficient to multiply the loss by.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="svd",
                 reduction="none",
                 check_nan=False
    ):
        super(SVDLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_losses_orth_keepsv"
    
    def __call__(self,
                 dfddfm_model,
                 name: str = None
    ):
        """
        Compute the orthogonal and keepsv losses for a given DFDDFM model.

        Args:
            dfddfm_model: The DFDDFM model to compute the losses for.
            name (str): The name of the loss function.

        Returns:
            dict: Dictionary containing the orthogonal and keepsv losses values.
        """
        reg_term = 0.0
        num_reg = 0
        for module in dfddfm_model.modules():

            logger.debug(f"module type: {type(module)}")
            logger.debug(f"current module: {module}")

            if isinstance(module, SVDResidualLinear):
                reg_term += module.__compute_orthogonal_loss__(module)

                logger.debug(f"orthogonal reg_term: {reg_term}")

                reg_term += module.__compute_keepsv_loss__(module)

                logger.debug(f"keepsv reg_term: {reg_term}")

                num_reg += 1
        loss = reg_term / num_reg

        logger.debug(f"number of svd loss layers: {num_reg}")

        name = self.name if name is None else name

        return self.return_loss(name, loss)

    def __compute_orthogonal_loss__(self, svd_residual_layer: SVDResidualLinear):
        if svd_residual_layer.S_residual is not None:
            # According to the properties of orthogonal matrices: A^TA = I
            UUT = torch.cat((svd_residual_layer.U_r, svd_residual_layer.U_residual), dim=1) @ torch.cat((svd_residual_layer.U_r, svd_residual_layer.U_residual), dim=1).t()
            VVT = torch.cat((svd_residual_layer.V_r, svd_residual_layer.V_residual), dim=0) @ torch.cat((svd_residual_layer.V_r, svd_residual_layer.V_residual), dim=0).t()
            # print(self.U_r.size(), self.U_residual.size())  # torch.Size([1024, 1023]) torch.Size([1024, 1])
            # print(self.V_r.size(), self.V_residual.size())  # torch.Size([1023, 1024]) torch.Size([1, 1024])
            # UUT = self.U_residual @ self.U_residual.t()
            # VVT = self.V_residual @ self.V_residual.t()
            
            # Construct an identity matrix
            UUT_identity = torch.eye(UUT.size(0)).to(UUT)
            VVT_identity = torch.eye(VVT.size(0)).to(VVT)

            # Using frobenius norm to compute loss
            loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
        else:
            loss = 0.0
            
        return loss

    def __compute_keepsv_loss__(self, svd_residual_layer: SVDResidualLinear):
        if (svd_residual_layer.S_residual is not None) and (svd_residual_layer.weight_original_fnorm is not None):
            # Total current weight is the fixed main weight plus the residual
            weight_current = svd_residual_layer.weight_r + svd_residual_layer.U_residual @ torch.diag(svd_residual_layer.S_residual) @ svd_residual_layer.V_residual
            # Frobenius norm of current weight
            weight_current_fnorm = torch.norm(weight_current, p='fro')

            loss = torch.abs(weight_current_fnorm ** 2 - svd_residual_layer.weight_original_fnorm ** 2)
        else:
            loss = 0.0
        
        return loss


class ConsistencyLoss(BaseLoss):
    """
    A class representing a consistency loss function.

    Args:
        coef (float): The coefficient for the loss.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="consistency",
                 reduction="mean",
                 check_nan=False):
        super(ConsistencyLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"
        self.loss_fn = F.mse_loss

    def __call__(self,
                 S_hat: Tensor,
                 S_original: Tensor,
                 name: str = None
    ):
        """
        Compute the consistency loss for a given SVDResidualLinear layer.

        Args:
            S_hat (Tensor): The predicted single disentangled manifold features.
            S_original (Tensor): The original single disentangled manifold features.
            name (str): The name of the loss function.

        Returns:
            dict: Dictionary containing the consistency loss value.
        """
        loss = self.loss_fn(S_hat, S_original, reduction='none')
        name = self.name if name is None else name

        return self.return_loss(name, loss)


class DistanceLoss(BaseLoss):
    """
    A class representing a distance loss function for the disentangled manifolds.

    Args:
        coef (float): The coefficient for the loss.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="distance",
                 reduction="mean",
                 check_nan=False):
        super(DistanceLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"

    def __call__(self,
                 manifolds_features1: Tensor,
                 manifolds_features2: Tensor,
                 manifolds_indices2: Tensor,
                 name: str = None
    ):
        """
        Compute the distance loss for disentangled manifolds.

        Args:
            manifolds_features1 (Tensor): The predicted single disentangled manifold features.
            manifolds_features2 (Tensor): The original single disentangled manifold features.
            manifolds_indices2 (Tensor): The indices of the disentangled manifolds to disentangle.
            name (str): The name of the loss function.

        Returns:
            dict: Dictionary containing the distance loss value.
        """
        dis_manifolds_features1 = F.normalize(manifolds_features1, p=2, dim=-1)
        dis_manifolds_features2 = F.normalize(manifolds_features2, p=2, dim=-1)
        sim_matrix = F.cosine_similarity(dis_manifolds_features1, dis_manifolds_features2, dim=-1).T
        sim_matrix = F.softmax(sim_matrix, dim=-1)

        logger.debug(f"sim_matrix shape: {sim_matrix.size()}; min value: {sim_matrix.min()}; max value: {sim_matrix.max()};")

        sim_mask = torch.zeros_like(sim_matrix)
        sim_mask[:, manifolds_indices2] = 1.0
        loss = sim_matrix * sim_mask

        logger.debug(f"loss shape: {loss.size()}; min value: {loss.min()}; max value: {loss.max()};")

        name = self.name if name is None else name

        return self.return_loss(name, loss)


class SparsityLoss(BaseLoss):
    """
    A class representing a sparsity loss function for the disentangled manifolds.

    Args:
        coef (float): The coefficient for the loss.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="sparsity",
                 reduction="mean",
                 check_nan=False):
        super(SparsityLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"

    def __call__(self,
                 manifolds_features1: Tensor,
                 manifolds_features2: Tensor,
                 name: str = None
    ):
        """
        Compute the sparsity loss for disentangled manifolds.

        Args:
            manifolds_features1 (Tensor): The predicted single disentangled manifold features.
            manifolds_features2 (Tensor): The original single disentangled manifold features.
            name (str): The name of the loss function.

        Returns:
            dict: Dictionary containing the sparsity loss value.
        """
        spar_manifolds_features1 = torch.zeros_like(manifolds_features1[0, :, :])
        spar_manifolds_features2 = torch.zeros_like(manifolds_features2[0, :, :])

        logger.debug(f"spar_manifolds_features1 shape: {spar_manifolds_features1.size()}")
        logger.debug(f"spar_manifolds_features2 shape: {spar_manifolds_features2.size()}")

        for manifold_idx in range(manifolds_features1.size(0)):
            remaining_indices = torch.tensor([idx for idx in range(manifolds_features1.size(0)) if idx != manifold_idx]).to(manifolds_features1)
            sum_manifolds_features1 = manifolds_features1[remaining_indices, :, :].sum(dim=0, keepdim=True)
            sum_manifolds_features2 = manifolds_features2[remaining_indices, :, :].sum(dim=0, keepdim=True)
            spar_manifolds_features1 += torch.abs(manifolds_features1[manifold_idx, :, :] * sum_manifolds_features1)
            spar_manifolds_features2 += torch.abs(manifolds_features2[manifold_idx, :, :] * sum_manifolds_features2)

        loss = spar_manifolds_features1 + spar_manifolds_features2

        logger.debug(f"loss shape: {loss.size()}; min value: {loss.min()}; max value: {loss.max()};")

        name = self.name if name is None else name

        return self.return_loss(name, loss)


class ReconRegLoss(BaseLoss):
    """
    A class representing a reconstruction regularization loss function for the disentangled manifolds.

    Args:
        coef (float): The coefficient for the loss.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="recon_reg",
                 reduction="mean",
                 check_nan=False):
        super(ReconRegLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"

    def __call__(self,
                 manifolds_features1: Tensor,
                 manifolds_features2: Tensor,
                 name: str = None
    ):
        """
        Compute the reconstruction regularization loss for disentangled manifolds.

        Args:
            manifolds_features1 (Tensor): The predicted single disentangled manifold features.
            manifolds_features2 (Tensor): The original single disentangled manifold features.
            name (str): The name of the loss function.

        Returns:
            dict: Dictionary containing the reconstruction regularization loss value.
        """
        dis_manifolds_features1 = F.normalize(manifolds_features1, p=2, dim=-1)
        dis_manifolds_features2 = F.normalize(manifolds_features2, p=2, dim=-1)
        dis_matrix = 1 - F.cosine_similarity(dis_manifolds_features1, dis_manifolds_features2, dim=-1).T
        dis_matrix = F.softmax(dis_matrix, dim=-1)

        logger.debug(f"dis_matrix shape: {dis_matrix.size()}; min value: {dis_matrix.min()}; max value: {dis_matrix.max()};")

        loss = (dis_matrix.mean(dim=0, keepdim=True) - 1 / dis_matrix.size(1))
        loss = torch.pow(loss, 2).sum(dim=1)

        logger.debug(f"loss shape: {loss.size()}; min value: {loss.min()}; max value: {loss.max()};")

        name = self.name if name is None else name

        return self.return_loss(name, loss)
