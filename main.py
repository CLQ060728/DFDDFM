# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

from typing import Literal, List, Dict, Any
from PIL.Image import Image
import torch
import torch.nn.functional as F
# from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as LTN
from torchmetrics.functional.classification import binary_accuracy, binary_auroc
from lightning.pytorch.cli import LightningCLI
from Model.DFDDFM import ClipSVDDFM, Dinov2SVDDFM, Dinov3SVDDFM
from Model.FeatureExtractors import ClipFeatureExtractor, Dinov2FeatureExtractor, Dinov3FeatureExtractor
from Loss.DFDDFMLosses import DFDLoss, ReconstructionLoss, SVDLoss, ConsistencyLoss, DistanceLoss
from Loss.DFDDFMLosses import SparsityLoss, ReconRegLoss
from Dataset.DatasetLoader import DFDDFMTrainDataModule
from Utils.UtilFunctions import ConfigDict
import logging, os

logger = logging.getLogger(__name__)


class DFDDFMTrainer(LTN.LightningModule):
    def __init__(self,
                 model_mode: Literal["SVDDFM", "SVD_DFM", "SVD", "FEAT", "FEAT_LINEAR"] = "SVDDFM",
                 svd_dfm: Dict[str, Any] = {},
                 model_type: Literal["CLIP", "DINO_V2", "DINO_V3"] | None = "DINO_V3",
                 feat_extractor_type: Literal["CLIP", "DINO_V2", "DINO_V3"] | None = None,
                 model_configs: Dict[str, Any] = {},
                 loss_configs: Dict[str, Any] = {},
                 optim_configs: Dict[str, Any] = {},
                 inference_configs: Dict[str, Any] = {}):
        """
            Initialize the DFDDFMTrainer with the given configurations.
            Params:
                model_mode: The mode of the model (SVDDFM, SVD_DFM, SVD, FEAT, FEAT_LINEAR).
                svd_dfm: Configuration dictionary for the SVD_DFM model.
                model_type: The type of the DFM model (CLIP, DINO_V2, DINO_V3).
                feat_extractor_type: The type of the feature extractor (CLIP, DINO_V2, DINO_V3).
                model_configs: Configuration dictionary for the model.
                loss_configs: Configuration dictionary for the loss functions.
                optim_configs: Configuration dictionary for the optimizer.
        """
        super(DFDDFMTrainer, self).__init__()
        self.save_hyperparameters()
        if model_mode == "SVD_DFM":
            assert svd_dfm is not None and len(svd_dfm) > 0, "svd_dfm config must be provided for SVD_DFM model_mode"
            assert svd_dfm.get("svd_chkpt_path", None) is not None and svd_dfm.get("svd_chkpt_path", None) != "", "svd_chkpt_path must be provided for SVD_DFM model_mode"
        assert model_mode in ["SVDDFM", "SVD_DFM", "SVD", "FEAT", "FEAT_LINEAR"], f"Invalid model_mode: {model_mode}"
        assert model_type in ["CLIP", "DINO_V2", "DINO_V3"], f"Invalid model_type: {model_type}"

        self.model_mode = model_mode
        self.svd_dfm = ConfigDict(svd_dfm)
        self.svd_dfm_with_dfd = self.svd_dfm.svd_dfm_with_dfd
        self.learning_rate = optim_configs.get("learning_rate", 2e-4)
        self.do_reconstruction = optim_configs.get("do_reconstruction", False)
        self.use_recon_reg_loss = optim_configs.get("use_recon_reg_loss", False)
        self.use_recon_reg_loss = True if self.do_reconstruction else self.use_recon_reg_loss
        self.model_type = model_type
        self.feat_extractor_type = feat_extractor_type
        self.model_configs = ConfigDict(model_configs)
        self.loss_configs = ConfigDict(loss_configs)
        self.optim_configs = ConfigDict(optim_configs)
        self.inference_configs = ConfigDict(inference_configs)
        self.beta_1 = self.loss_configs.DistanceSparsity.beta_1_start
        self.__get_all_training_objects__()  # Initialize model and losses
    
    def __check_model_grad__(self, model):
        total_frozen_params = 0
        total_trainable_params = 0
        for _, param in model.named_parameters():
            if param.requires_grad:
                total_trainable_params += 1
            else:
                total_frozen_params += 1
        
        logger.info(f"Total trainable params: {total_trainable_params}")
        logger.info(f"Total frozen params: {total_frozen_params}")

    def __get_all_training_objects__(self):
        if self.model_mode == "SVD":
            self.model_configs.ClipSVDDFM.dfm = False
            self.model_configs.Dinov2SVDDFM.dfm = False
            self.model_configs.Dinov3SVDDFM.dfm = False
        elif self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
            self.model_configs.ClipSVDDFM.dfm = True
            self.model_configs.Dinov2SVDDFM.dfm = True
            self.model_configs.Dinov3SVDDFM.dfm = True

        if self.model_mode == "SVDDFM" or self.model_mode == "SVD":
            if self.model_type == "CLIP":
                self.model = ClipSVDDFM(self.device, self.model_configs.ClipSVDDFM.dfm,
                                        self.model_configs.ClipSVDDFM.dfm_num_layers,
                                        self.model_configs.ClipSVDDFM.dfm_aggr,
                                        self.model_configs.ClipSVDDFM.out_feat_type,
                                        self.model_configs.ClipSVDDFM.chkpt_dir)
            elif self.model_type == "DINO_V2":
                self.model = Dinov2SVDDFM(self.device, self.model_configs.Dinov2SVDDFM.dfm,
                                        self.model_configs.Dinov2SVDDFM.dfm_num_layers,
                                        self.model_configs.Dinov2SVDDFM.dfm_aggr,
                                        self.model_configs.Dinov2SVDDFM.out_feat_type,
                                        self.model_configs.Dinov2SVDDFM.chkpt_dir)
            elif self.model_type == "DINO_V3":
                self.model = Dinov3SVDDFM(self.device, self.model_configs.Dinov3SVDDFM.dfm,
                                        self.model_configs.Dinov3SVDDFM.dfm_num_layers,
                                        self.model_configs.Dinov3SVDDFM.dfm_aggr,
                                        self.model_configs.Dinov3SVDDFM.out_feat_type,
                                        self.model_configs.Dinov3SVDDFM.model_type,
                                        self.model_configs.Dinov3SVDDFM.chkpt_dir)
        
        if self.model_mode == "SVD_DFM":
            assert os.path.exists(self.svd_dfm.svd_chkpt_path), f"SVD checkpoint path does not exist: {self.svd_dfm.svd_chkpt_path}"
            logger.debug(f"Loading pretrained SVD model from {self.svd_dfm.svd_chkpt_path} for SVD_DFM training...")

            pre_trained_svd_feat_model = self.load_from_checkpoint(self.svd_dfm.svd_chkpt_path).model.feat_model
            pre_trained_svd_feat_model.requires_grad_(False)  # Freeze the feature extractor
            self.__check_model_grad__(pre_trained_svd_feat_model)

            if self.model_type == "CLIP":
                self.model = ClipSVDDFM(self.device, self.model_configs.ClipSVDDFM.dfm,
                                        self.model_configs.ClipSVDDFM.dfm_num_layers,
                                        self.model_configs.ClipSVDDFM.dfm_aggr,
                                        self.model_configs.ClipSVDDFM.out_feat_type,
                                        self.model_configs.ClipSVDDFM.chkpt_dir)
            elif self.model_type == "DINO_V2":
                self.model = Dinov2SVDDFM(self.device, self.model_configs.Dinov2SVDDFM.dfm,
                                        self.model_configs.Dinov2SVDDFM.dfm_num_layers,
                                        self.model_configs.Dinov2SVDDFM.dfm_aggr,
                                        self.model_configs.Dinov2SVDDFM.out_feat_type,
                                        self.model_configs.Dinov2SVDDFM.chkpt_dir)
            elif self.model_type == "DINO_V3":
                self.model = Dinov3SVDDFM(self.device, self.model_configs.Dinov3SVDDFM.dfm,
                                        self.model_configs.Dinov3SVDDFM.dfm_num_layers,
                                        self.model_configs.Dinov3SVDDFM.dfm_aggr,
                                        self.model_configs.Dinov3SVDDFM.out_feat_type,
                                        self.model_configs.Dinov3SVDDFM.model_type,
                                        self.model_configs.Dinov3SVDDFM.chkpt_dir)
            self.__check_model_grad__(self.model.feat_model)

            self.model.feat_model = pre_trained_svd_feat_model

            self.__check_model_grad__(self.model.feat_model)
            self.__check_model_grad__(self.model)

        if self.model_mode == "FEAT" or self.model_mode == "FEAT_LINEAR":
            self.model_configs.ClipFeatureExtractor.as_linear_classifier =\
                True if self.model_mode == "FEAT_LINEAR" else False
            self.model_configs.Dinov2FeatureExtractor.as_linear_classifier =\
                True if self.model_mode == "FEAT_LINEAR" else False
            self.model_configs.Dinov3FeatureExtractor.as_linear_classifier =\
                True if self.model_mode == "FEAT_LINEAR" else False
            
            if self.feat_extractor_type == "CLIP":
                self.feat_extractor = ClipFeatureExtractor(self.device,
                                                self.model_configs.ClipFeatureExtractor.as_linear_classifier,
                                                self.model_configs.ClipFeatureExtractor.chkpt_dir)
            elif self.feat_extractor_type == "DINO_V2":
                self.feat_extractor = Dinov2FeatureExtractor(self.device,
                                                self.model_configs.Dinov2FeatureExtractor.as_linear_classifier,
                                                self.model_configs.Dinov2FeatureExtractor.chkpt_dir)
            elif self.feat_extractor_type == "DINO_V3":
                self.feat_extractor = Dinov3FeatureExtractor(self.device,
                                                self.model_configs.Dinov3FeatureExtractor.pre_ds,
                                                self.model_configs.Dinov3FeatureExtractor.as_linear_classifier,
                                                self.model_configs.Dinov3FeatureExtractor.chkpt_dir)
        
        if self.model_mode != "FEAT":
            self.dfd_loss = DFDLoss(coef=self.loss_configs.DFDLoss.coef)
            if not self.svd_dfm_with_dfd and (self.model_mode == "SVD_DFM"):
                self.dfd_loss = None
                del self.dfd_loss
            if self.model_mode == "SVDDFM" or self.model_mode == "SVD":
                self.svd_loss = SVDLoss(coef=self.loss_configs.SVDLoss.coef)
            if self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
                self.consistency_loss = ConsistencyLoss()
                self.distance_loss = DistanceLoss()
                self.sparsity_loss = SparsityLoss()
                if self.do_reconstruction:
                    self.recon_loss = ReconstructionLoss(coef=self.loss_configs.ReconstructionLoss.coef)
                    self.recon_reg_loss = ReconRegLoss(coef=self.loss_configs.ReconRegLoss.beta_3_max)
                else:
                    if self.use_recon_reg_loss:
                        self.recon_reg_loss = ReconRegLoss(coef=self.loss_configs.ReconRegLoss.beta_3_max)

    def forward(self, batch):
        if self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
            x_pair, x2_manifold_indices, y_pair = batch
            x_1, x_2 = x_pair
            y_hat_1, encoder_features_1, decoder_features_1, manifolds_features_1 = self.model(x_1)
            y_hat_2, encoder_features_2, decoder_features_2, manifolds_features_2 = self.model(x_2)
            return y_hat_1, encoder_features_1, decoder_features_1, manifolds_features_1,\
                   y_hat_2, encoder_features_2, decoder_features_2, manifolds_features_2,\
                   x2_manifold_indices, y_pair
        elif self.model_mode == "SVD":
            x, y = batch
            y_hat, _, _, _ = self.model(x)
            return y_hat, y
        elif self.model_mode == "FEAT_LINEAR":
            x, y = batch
            y_hat = self.feat_extractor(x)
            return y_hat, y
        else:
            return self.feat_extractor(batch)

    def __svd_linear_training_step__(self, batch, total_loss):
        x_pair, _, y_pair = batch  # x2_manifold_indices
        x_1, x_2 = x_pair
        y_1, y_2 = y_pair
        batch_1 = (x_1, y_1)
        batch_2 = (x_2, y_2)
        y_hat_1, y_1 = self(batch_1)
        y_hat_2, y_2 = self(batch_2)
        dfd_loss_dict_1 = self.dfd_loss(y_hat_1, y_1)
        dfd_loss_dict_2 = self.dfd_loss(y_hat_2, y_2)

        logger.debug(f"dfd_loss_dict_1: {dfd_loss_dict_1}")
        logger.debug(f"dfd_loss_dict_2: {dfd_loss_dict_2}")

        dfd_loss_value_1 = dfd_loss_dict_1["dfd_loss"]
        dfd_loss_value_2 = dfd_loss_dict_2["dfd_loss"]
        total_loss.update({"dfd_loss": (dfd_loss_value_1 + dfd_loss_value_2) / 2})
        if self.model_mode != "FEAT_LINEAR" and self.model_mode != "FEAT":
            svd_losses_dict = self.svd_loss(self.model)
            svd_losses_value = svd_losses_dict["svd_losses_orth_keepsv"]
            total_loss.update({"svd_losses": svd_losses_value})

    def __check_network_grad__(self):
        total_frozen_params = 0
        total_trainable_params = 0
        if self.model_mode == "SVDDFM" or self.model_mode == "SVD":
            for _, param in self.model.named_parameters():
                if param.requires_grad:
                    total_trainable_params += 1
                else:
                    total_frozen_params += 1
        elif self.model_mode == "FEAT" or self.model_mode == "FEAT_LINEAR":
            for _, param in self.feat_extractor.named_parameters():
                if param.requires_grad:
                    total_trainable_params += 1
                else:
                    total_frozen_params += 1
        
        logger.debug(f"{self.model_mode}, Total trainable params: {total_trainable_params}")
        logger.debug(f"{self.model_mode}, Total frozen params: {total_frozen_params}")

        # if self.model_mode == "SVDDFM" or self.model_mode == "SVD":
        #     logger.debug(f"Checking feat extractor parameters...")
        #     for param in network.feat_model.parameters():
        #         if param.requires_grad:
        #             total_trainable_params += 1
        #         else:
        #             total_frozen_params += 1
                
        # logger.debug(f"Total trainable params with feat extractor: {total_trainable_params}")
        # logger.debug(f"Total frozen params with feat extractor: {total_frozen_params}")

    def training_step(self, batch, batch_idx):
        total_loss = {}
        if self.model_mode == "SVDDFM":

            self.__check_network_grad__()

            if self.current_epoch < self.optim_configs.dfm_start_epoch:
                logger.debug(f"Switching model mode from SVDDFM to SVD")

                self.model_mode = "SVD"
                self.model.dfm = False

                logger.debug(f"Model type set to: {self.model_type}; Model DFM set to: {self.model.dfm}")
                logger.debug(f"total_loss before __svd_linear_training_step__: {total_loss}")

                self.__svd_linear_training_step__(batch, total_loss)

                logger.debug(f"total_loss after __svd_linear_training_step__: {total_loss}")

                self.model_mode = "SVDDFM"
                self.model.dfm = True

                self.log_dict(total_loss, prog_bar=True)
                logger.debug(f"Model type set to: {self.model_type}; Model DFM set to: {self.model.dfm}")
            else:
                logger.debug(f"Epoch {self.current_epoch}: Training with full SVDDFM model")

                y_hat_1, encoder_features_1, decoder_features_1, manifolds_features_1,\
                y_hat_2, encoder_features_2, decoder_features_2, manifolds_features_2,\
                x2_manifold_indices, y_pair = self(batch)
                y_1, y_2 = y_pair
                
                # compute dfd and svd losses
                dfd_loss_dict_1 = self.dfd_loss(y_hat_1, y_1)
                dfd_loss_dict_2 = self.dfd_loss(y_hat_2, y_2)
                dfd_loss_value_1 = dfd_loss_dict_1["dfd_loss"]
                dfd_loss_value_2 = dfd_loss_dict_2["dfd_loss"]
                total_loss.update({"dfd_loss": (dfd_loss_value_1 + dfd_loss_value_2) / 2})
                svd_losses_dict = self.svd_loss(self.model)
                svd_losses_value = svd_losses_dict["svd_losses_orth_keepsv"]
                total_loss.update({"svd_losses": svd_losses_value})
                
                # compute dfm reconstruction and regularization losses
                if self.do_reconstruction:
                    recon_loss_dict_1 = self.recon_loss(decoder_features_1, encoder_features_1)
                    recon_loss_dict_2 = self.recon_loss(decoder_features_2, encoder_features_2)
                    recon_loss_value_1 = recon_loss_dict_1[f"reconstruction_loss_{self.recon_loss.loss_type}"]
                    recon_loss_value_2 = recon_loss_dict_2[f"reconstruction_loss_{self.recon_loss.loss_type}"]
                    total_loss.update({"recon_loss": (recon_loss_value_1 + recon_loss_value_2) / 2})
                    
                    if self.current_epoch < self.optim_configs.dissparcons_start_epoch:
                        recon_reg_loss_dict = self.recon_reg_loss(manifolds_features_1, manifolds_features_2)
                        recon_reg_loss_value = recon_reg_loss_dict["recon_reg_loss"]
                        total_loss.update({"recon_reg_loss": recon_reg_loss_value})
                
                # compute other dfm losses after 'dissparcons_start_epoch's, i.e., distance, sparsity, consistency
                if self.current_epoch >= self.optim_configs.dissparcons_start_epoch:
                    self.beta_1 = self.loss_configs.DistanceSparsity.beta_1_start * (1 + (1 - 1 / (batch_idx + 1)))\
                                if self.beta_1 < self.loss_configs.DistanceSparsity.beta_1_end\
                                else self.loss_configs.DistanceSparsity.beta_1_end
                    beta_2 = (self.beta_1 - 1 / (self.current_epoch + 1))\
                                if (self.beta_1 - 1 / (self.current_epoch + 1)) > 0 else 0
                    beta_2 = self.loss_configs.ConsistencyLoss.beta_2_max\
                                if beta_2 > self.loss_configs.ConsistencyLoss.beta_2_max\
                                else beta_2
                    beta_3 = self.loss_configs.ReconRegLoss.beta_3_max - beta_2
                    self.distance_loss.set_coef(self.beta_1)
                    self.sparsity_loss.set_coef(self.beta_1)
                    self.consistency_loss.set_coef(beta_2)
                    self.recon_reg_loss.set_coef(beta_3)

                    logger.debug(f"Distance loss coefficient set to: {self.distance_loss.coef}")
                    logger.debug(f"Sparsity loss coefficient set to: {self.sparsity_loss.coef}")
                    logger.debug(f"Consistency loss coefficient set to: {self.consistency_loss.coef}")
                    logger.debug(f"Reconstruction regularization loss coefficient set to: {self.recon_reg_loss.coef}")

                    distance_loss_dict = self.distance_loss(manifolds_features_1, manifolds_features_2,
                                                            x2_manifold_indices)
                    distance_loss_value = distance_loss_dict["distance_loss"]
                    total_loss.update({"distance_loss": distance_loss_value})

                    sparsity_loss_dict = self.sparsity_loss(manifolds_features_1, manifolds_features_2)
                    sparsity_loss_value = sparsity_loss_dict["sparsity_loss"]
                    total_loss.update({"sparsity_loss": sparsity_loss_value})

                    # prepare S_hat for all the manifolds, consistency loss
                    S_hat = torch.tensor([]).to(manifolds_features_1)
                    for manifold_idx in range(manifolds_features_1.size(0)):
                        remaining_indices = torch.tensor([idx for idx in range(manifolds_features_1.size(0))\
                                                          if idx != manifold_idx]).to(manifolds_features_1.device)

                        logger.debug(f"manifold_idx: {manifold_idx}; remaining_indices: {remaining_indices}")

                        if self.model.dfm_aggr == "SUM":
                            aggr_12 = torch.cat((manifolds_features_1[[manifold_idx], :, :],
                                                 manifolds_features_2[remaining_indices, :, :]),
                                                 dim=0).sum(dim=0)
                            
                            logger.debug(f"aggr_12 shape: {aggr_12.size()}")
                        else: # self.model.dfm_aggr == "CONCAT"
                            aggr_12 = []
                            aggr_12.append(manifolds_features_1[manifold_idx, :, :])
                            for remaining_idx in remaining_indices:
                                aggr_12.append(manifolds_features_2[remaining_idx, :, :])
                            aggr_12 = torch.hstack(aggr_12)
                            
                            logger.debug(f"aggr_12 shape: {aggr_12.size()}")
                        
                        X_s_hat = self.model.dfm_decoder(aggr_12)
                        f_12 = self.model.dfm_encoder(X_s_hat)
                        S_manifold_idx = self.model.orthogonal_manifolds[manifold_idx](f_12)
                        S_hat = torch.cat((S_hat, S_manifold_idx.unsqueeze(0)), dim=0)

                    logger.debug(f"S_hat shape: {S_hat.size()}")

                    consistency_loss_dict = self.consistency_loss(S_hat, manifolds_features_1)
                    consistency_loss_value = consistency_loss_dict["consistency_loss"]
                    total_loss.update({"consistency_loss": consistency_loss_value})

                    if self.use_recon_reg_loss:
                        recon_reg_loss_dict = self.recon_reg_loss(manifolds_features_1, manifolds_features_2)
                        recon_reg_loss_value = recon_reg_loss_dict["recon_reg_loss"]
                        total_loss.update({"recon_reg_loss": recon_reg_loss_value})

                self.log_dict(total_loss, prog_bar=True)
        elif self.model_mode == "SVD_DFM":
            logger.debug(f"Epoch {self.current_epoch}: Training with SVD_DFM model")

            y_hat_1, encoder_features_1, decoder_features_1, manifolds_features_1,\
            y_hat_2, encoder_features_2, decoder_features_2, manifolds_features_2,\
            x2_manifold_indices, y_pair = self(batch)
            y_1, y_2 = y_pair
            
            # compute dfd losses if applicable
            if self.svd_dfm_with_dfd:
                dfd_loss_dict_1 = self.dfd_loss(y_hat_1, y_1)
                dfd_loss_dict_2 = self.dfd_loss(y_hat_2, y_2)
                dfd_loss_value_1 = dfd_loss_dict_1["dfd_loss"]
                dfd_loss_value_2 = dfd_loss_dict_2["dfd_loss"]
                total_loss.update({"dfd_loss": (dfd_loss_value_1 + dfd_loss_value_2) / 2})
            
            # compute dfm reconstruction and regularization losses
            if self.do_reconstruction:
                recon_loss_dict_1 = self.recon_loss(decoder_features_1, encoder_features_1)
                recon_loss_dict_2 = self.recon_loss(decoder_features_2, encoder_features_2)
                recon_loss_value_1 = recon_loss_dict_1[f"reconstruction_loss_{self.recon_loss.loss_type}"]
                recon_loss_value_2 = recon_loss_dict_2[f"reconstruction_loss_{self.recon_loss.loss_type}"]
                total_loss.update({"recon_loss": (recon_loss_value_1 + recon_loss_value_2) / 2})
                
                if self.current_epoch < self.optim_configs.dissparcons_start_epoch:
                    recon_reg_loss_dict = self.recon_reg_loss(manifolds_features_1, manifolds_features_2)
                    recon_reg_loss_value = recon_reg_loss_dict["recon_reg_loss"]
                    total_loss.update({"recon_reg_loss": recon_reg_loss_value})
            
            # compute other dfm losses after 'dissparcons_start_epoch's, i.e., distance, sparsity, consistency
            if self.current_epoch >= self.optim_configs.dissparcons_start_epoch:
                self.beta_1 = self.loss_configs.DistanceSparsity.beta_1_start * (1 + (1 - 1 / (batch_idx + 1)))\
                            if self.beta_1 < self.loss_configs.DistanceSparsity.beta_1_end\
                            else self.loss_configs.DistanceSparsity.beta_1_end
                beta_2 = (self.beta_1 - 1 / (self.current_epoch + 1))\
                            if (self.beta_1 - 1 / (self.current_epoch + 1)) > 0 else 0
                beta_2 = self.loss_configs.ConsistencyLoss.beta_2_max\
                            if beta_2 > self.loss_configs.ConsistencyLoss.beta_2_max\
                            else beta_2
                beta_3 = self.loss_configs.ReconRegLoss.beta_3_max - beta_2
                self.distance_loss.set_coef(self.beta_1)
                self.sparsity_loss.set_coef(self.beta_1)
                self.consistency_loss.set_coef(beta_2)
                self.recon_reg_loss.set_coef(beta_3)

                logger.debug(f"Distance loss coefficient set to: {self.distance_loss.coef}")
                logger.debug(f"Sparsity loss coefficient set to: {self.sparsity_loss.coef}")
                logger.debug(f"Consistency loss coefficient set to: {self.consistency_loss.coef}")
                logger.debug(f"Reconstruction regularization loss coefficient set to: {self.recon_reg_loss.coef}")

                distance_loss_dict = self.distance_loss(manifolds_features_1, manifolds_features_2,
                                                        x2_manifold_indices)
                distance_loss_value = distance_loss_dict["distance_loss"]
                total_loss.update({"distance_loss": distance_loss_value})

                sparsity_loss_dict = self.sparsity_loss(manifolds_features_1, manifolds_features_2)
                sparsity_loss_value = sparsity_loss_dict["sparsity_loss"]
                total_loss.update({"sparsity_loss": sparsity_loss_value})

                # prepare S_hat for all the manifolds, consistency loss
                S_hat = torch.tensor([]).to(manifolds_features_1)
                for manifold_idx in range(manifolds_features_1.size(0)):
                    remaining_indices = torch.tensor([idx for idx in range(manifolds_features_1.size(0))\
                                                        if idx != manifold_idx]).to(manifolds_features_1.device)

                    logger.debug(f"manifold_idx: {manifold_idx}; remaining_indices: {remaining_indices}")

                    if self.model.dfm_aggr == "SUM":
                        aggr_12 = torch.cat((manifolds_features_1[[manifold_idx], :, :],
                                                manifolds_features_2[remaining_indices, :, :]),
                                                dim=0).sum(dim=0)
                        
                        logger.debug(f"aggr_12 shape: {aggr_12.size()}")
                    else: # self.model.dfm_aggr == "CONCAT"
                        aggr_12 = []
                        aggr_12.append(manifolds_features_1[manifold_idx, :, :])
                        for remaining_idx in remaining_indices:
                            aggr_12.append(manifolds_features_2[remaining_idx, :, :])
                        aggr_12 = torch.hstack(aggr_12)
                        
                        logger.debug(f"aggr_12 shape: {aggr_12.size()}")
                    
                    X_s_hat = self.model.dfm_decoder(aggr_12)
                    f_12 = self.model.dfm_encoder(X_s_hat)
                    S_manifold_idx = self.model.orthogonal_manifolds[manifold_idx](f_12)
                    S_hat = torch.cat((S_hat, S_manifold_idx.unsqueeze(0)), dim=0)

                logger.debug(f"S_hat shape: {S_hat.size()}")

                consistency_loss_dict = self.consistency_loss(S_hat, manifolds_features_1)
                consistency_loss_value = consistency_loss_dict["consistency_loss"]
                total_loss.update({"consistency_loss": consistency_loss_value})

                if self.use_recon_reg_loss:
                    recon_reg_loss_dict = self.recon_reg_loss(manifolds_features_1, manifolds_features_2)
                    recon_reg_loss_value = recon_reg_loss_dict["recon_reg_loss"]
                    total_loss.update({"recon_reg_loss": recon_reg_loss_value})

            self.log_dict(total_loss, prog_bar=True)
        elif self.model_mode == "SVD":
            self.__check_network_grad__()

            logger.debug(f"total_loss before __svd_linear_training_step__: {total_loss}")

            self.__svd_linear_training_step__(batch, total_loss)

            logger.debug(f"total_loss after __svd_linear_training_step__: {total_loss}")
            self.log_dict(total_loss, prog_bar=True)
        elif self.model_mode == "FEAT_LINEAR":
            self.__check_network_grad__()

            logger.debug(f"{self.model_mode}, total_loss before __svd_linear_training_step__: {total_loss}")

            self.__svd_linear_training_step__(batch, total_loss)

            logger.debug(f"total_loss after __svd_linear_training_step__: {total_loss}")
            self.log_dict(total_loss, prog_bar=True)

        total_loss_value = sum(total_loss.values()) if len(total_loss) > 0 else None

        logger.debug(f"total_loss_value: {total_loss_value}")

        return total_loss_value

    def __val_test_common_step__(self, batch: torch.Tensor,
                                 step_mode: Literal["val", "test"] = "val"):
        total_loss = {}
        total_performance = {}
        if self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
            self.__check_network_grad__()

            y_hat_1, encoder_features_1, decoder_features_1, manifolds_features_1,\
                y_hat_2, encoder_features_2, decoder_features_2, manifolds_features_2,\
                x2_manifold_indices, y_pair = self(batch)
            y_1, y_2 = y_pair

            # compute accuracy and ROC-AUC
            accuracy_1 = binary_accuracy(F.sigmoid(y_hat_1), y_1.long())
            accuracy_2 = binary_accuracy(F.sigmoid(y_hat_2), y_2.long())
            roc_auc_1 = binary_auroc(F.sigmoid(y_hat_1), y_1.long())
            roc_auc_2 = binary_auroc(F.sigmoid(y_hat_2), y_2.long())
            accuracy = (accuracy_1 + accuracy_2) / 2
            roc_auc = (roc_auc_1 + roc_auc_2) / 2
            total_performance.update({"accuracy": accuracy, "roc_auc": roc_auc})
            
            if step_mode == "val":
                # COMPUTE ALL LOSSES
                if self.svd_dfm_with_dfd or (self.model_mode == "SVDDFM"):
                    # compute dfd and svd losses
                    dfd_loss_dict_1 = self.dfd_loss(y_hat_1, y_1)
                    dfd_loss_dict_2 = self.dfd_loss(y_hat_2, y_2)
                    dfd_loss_value_1 = dfd_loss_dict_1["dfd_loss"]
                    dfd_loss_value_2 = dfd_loss_dict_2["dfd_loss"]
                    total_loss.update({"dfd_loss": (dfd_loss_value_1 + dfd_loss_value_2) / 2})
                if self.model_mode == "SVDDFM":
                    # compute svd losses
                    svd_losses_dict = self.svd_loss(self.model)
                    svd_losses_value = svd_losses_dict["svd_losses_orth_keepsv"]
                    total_loss.update({"svd_losses": svd_losses_value})

                # compute dfm reconstruction and regularization losses
                if self.current_epoch >= self.optim_configs.dfm_start_epoch:
                    if self.do_reconstruction:
                        recon_loss_dict_1 = self.recon_loss(decoder_features_1, encoder_features_1)
                        recon_loss_dict_2 = self.recon_loss(decoder_features_2, encoder_features_2)
                        recon_loss_value_1 = recon_loss_dict_1[f"reconstruction_loss_{self.recon_loss.loss_type}"]
                        recon_loss_value_2 = recon_loss_dict_2[f"reconstruction_loss_{self.recon_loss.loss_type}"]
                        total_loss.update({"recon_loss": (recon_loss_value_1 + recon_loss_value_2)})
                        if self.use_recon_reg_loss:
                            recon_reg_loss_dict = self.recon_reg_loss(manifolds_features_1, manifolds_features_2)
                            recon_reg_loss_value = recon_reg_loss_dict["recon_reg_loss"]
                            total_loss.update({"recon_reg_loss": recon_reg_loss_value})
                    
                    # compute other dfm losses after 'dissparcons_start_epoch's, i.e., distance, sparsity, consistency
                    if self.current_epoch >= self.optim_configs.dissparcons_start_epoch:
                        distance_loss_dict = self.distance_loss(manifolds_features_1, manifolds_features_2,
                                                                x2_manifold_indices)
                        distance_loss_value = distance_loss_dict["distance_loss"]
                        total_loss.update({"distance_loss": distance_loss_value})

                        sparsity_loss_dict = self.sparsity_loss(manifolds_features_1, manifolds_features_2)
                        sparsity_loss_value = sparsity_loss_dict["sparsity_loss"]
                        total_loss.update({"sparsity_loss": sparsity_loss_value})

                        # prepare S_hat for all the manifolds, consistency loss
                        S_hat = torch.tensor([]).to(manifolds_features_1)
                        for manifold_idx in range(manifolds_features_1.size(0)):
                            remaining_indices = torch.tensor([idx for idx in range(manifolds_features_1.size(0))\
                                                            if idx != manifold_idx]).to(manifolds_features_1.device)
                            if self.model.dfm_aggr == "SUM":
                                aggr_12 = torch.cat((manifolds_features_1[[manifold_idx], :, :],
                                                        manifolds_features_2[remaining_indices, :, :]),
                                                        dim=0).sum(dim=0)
                            else: # self.model.dfm_aggr == "CONCAT"
                                aggr_12 = []
                                aggr_12.append(manifolds_features_1[manifold_idx, :, :])
                                for remaining_idx in remaining_indices:
                                    aggr_12.append(manifolds_features_2[remaining_idx, :, :])
                                aggr_12 = torch.hstack(aggr_12)
                            X_s_hat = self.model.dfm_decoder(aggr_12)
                            f_12 = self.model.dfm_encoder(X_s_hat)
                            S_manifold_idx = self.model.orthogonal_manifolds[manifold_idx](f_12)
                            S_hat = torch.cat((S_hat, S_manifold_idx.unsqueeze(0)), dim=0)
                        consistency_loss_dict = self.consistency_loss(S_hat, manifolds_features_1)
                        consistency_loss_value = consistency_loss_dict["consistency_loss"]
                        total_loss.update({"consistency_loss": consistency_loss_value})

                        if self.use_recon_reg_loss:
                            recon_reg_loss_dict = self.recon_reg_loss(manifolds_features_1, manifolds_features_2)
                            recon_reg_loss_value = recon_reg_loss_dict["recon_reg_loss"]
                            total_loss.update({"recon_reg_loss": recon_reg_loss_value})
        elif self.model_mode == "SVD":
            self.__check_network_grad__()

            self.__svd_linear_validation_step__(batch, total_performance, total_loss, step_mode)
        elif self.model_mode == "FEAT_LINEAR":
            self.__check_network_grad__()

            self.__svd_linear_validation_step__(batch, total_performance, total_loss, step_mode)

        val_test_results = total_performance | total_loss

        self.log_dict(val_test_results, prog_bar=True)

    def __svd_linear_validation_step__(self, batch,
                                       total_performance, total_loss,
                                       step_mode: Literal["val", "test"] = "val"):
        x_pair, _, y_pair = batch  # x2_manifold_indices
        x_1, x_2 = x_pair
        y_1, y_2 = y_pair
        batch_1 = (x_1, y_1)
        batch_2 = (x_2, y_2)
        y_hat_1, y_1 = self(batch_1)
        y_hat_2, y_2 = self(batch_2)

        # compute accuracy and ROC-AUC
        accuracy_1 = binary_accuracy(F.sigmoid(y_hat_1), y_1.long())
        accuracy_2 = binary_accuracy(F.sigmoid(y_hat_2), y_2.long())
        roc_auc_1 = binary_auroc(F.sigmoid(y_hat_1), y_1.long())
        roc_auc_2 = binary_auroc(F.sigmoid(y_hat_2), y_2.long())
        accuracy = (accuracy_1 + accuracy_2) / 2
        roc_auc = (roc_auc_1 + roc_auc_2) / 2
        total_performance.update({"accuracy": accuracy, "roc_auc": roc_auc})
        
        if step_mode == "val":
            # COMPUTE ALL LOSSES
            # compute dfd loss
            dfd_loss_dict_1 = self.dfd_loss(y_hat_1, y_1)
            dfd_loss_dict_2 = self.dfd_loss(y_hat_2, y_2)
            dfd_loss_value_1 = dfd_loss_dict_1["dfd_loss"]
            dfd_loss_value_2 = dfd_loss_dict_2["dfd_loss"]
            total_loss.update({"dfd_loss": (dfd_loss_value_1 + dfd_loss_value_2) / 2})
            # compute svd loss
            if self.model_mode != "FEAT_LINEAR" and self.model_mode != "FEAT":
                svd_losses_dict = self.svd_loss(self.model)
                svd_losses_value = svd_losses_dict["svd_losses_orth_keepsv"]
                total_loss.update({"svd_losses": svd_losses_value})

    def validation_step(self, batch: torch.Tensor):
        self.__val_test_common_step__(batch)

    def test_step(self, batch: torch.Tensor):
        self.__val_test_common_step__(batch, step_mode="test")

    def predict_step(self, batch: List[Image]):
        pass

    def configure_optimizers(self):
        if self.model_mode == "FEAT_LINEAR":
            optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                                self.feat_extractor.parameters()),
                                         lr=self.learning_rate)
        elif self.model_mode == "SVD" or self.model_mode == "SVDDFM":
            optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                                self.model.parameters()),
                                         lr=self.learning_rate)

        return optimizer
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.optim_configs.scheduler_T_max)

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch"  # or "step"
        #     }
        # }


def cli_main():
    cli = LightningCLI(DFDDFMTrainer, DFDDFMTrainDataModule, seed_everything_default=88)


if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)
    logging.basicConfig(filename='./logs/main.log', level=logging.INFO)
    # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # configure logging on module level, redirect to file
    # logger = logging.getLogger("pytorch_lightning.core")
    # logger.addHandler(logging.FileHandler("./logs/ltn_core.log"))

    cli_main()