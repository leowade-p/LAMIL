from pathlib import Path
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, Accuracy, AUROC


class VeryBasicModel(pl.LightningModule):
    def __init__(self, save_hyperparameters=True):
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters()
        self._step_train = -1
        self._step_val = -1
        self._step_test = -1


    def forward(self, x, cond=None):
        raise NotImplementedError

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        raise NotImplementedError
    
    def _epoch_end(self, state:str):
        return 

    def training_step(self, batch: dict, batch_idx: int ):
        self._step_train += 1 
        return self._step(batch, batch_idx, "train", self._step_train)

    def validation_step(self, batch: dict, batch_idx: int):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val )

    def test_step(self, batch: dict, batch_idx: int):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test)

    def on_train_epoch_end(self) -> None: 
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")


    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_epoch': Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, **kwargs):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'r') as f:
            path_rel_best_checkpoint = Path(json.load(f)['best_model_epoch'])
        return Path(path_checkpoint_dir)/path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, **kwargs):
        path_best_checkpoint = cls._get_best_checkpoint_path(path_checkpoint_dir)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)

    def load_pretrained(self, checkpoint_path, map_location=None, **kwargs):
        if checkpoint_path.is_dir():
            checkpoint_path = self._get_best_checkpoint_path(checkpoint_path, **kwargs)  

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
     
        return self.load_weights(checkpoint["state_dict"], **kwargs)
    
    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        filter = kwargs.get('filter', lambda key:key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self 




class BasicModel(VeryBasicModel):
    def __init__(
        self, 
        optimizer=torch.optim.Adam, 
        optimizer_kwargs={'lr':1e-3, 'weight_decay':1e-2},
        lr_scheduler= None, 
        lr_scheduler_kwargs={},
        save_hyperparameters=True
    ):
        super().__init__(save_hyperparameters=save_hyperparameters)
        if save_hyperparameters:
            self.save_hyperparameters()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler 
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            lr_scheduler_config  = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [lr_scheduler_config ]
        else:
            return [optimizer]





# class BasicClassifier(BasicModel):
#     def __init__(
#         self, 
#         in_ch,
#         out_ch,
#         spatial_dims,
#         loss = torch.nn.CrossEntropyLoss,
#         loss_kwargs = {},
#         optimizer=torch.optim.AdamW, 
#         optimizer_kwargs={'lr':1e-4, 'weight_decay':1e-2},
#         lr_scheduler= None, 
#         lr_scheduler_kwargs={},
#         # aucroc_kwargs={"task":"binary"},
#         # acc_kwargs={"task":"binary"}
#         aucroc_kwargs={"task":"multiclass"},
#         acc_kwargs={"task":"multiclass"},
#         save_hyperparameters=True,
#     ):
#         super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, save_hyperparameters)
#         self.in_ch = in_ch 
#         self.out_ch = out_ch 
#         self.spatial_dims = spatial_dims
#         self.loss_func = loss(**loss_kwargs)
#         self.loss_kwargs = loss_kwargs 

#         aucroc_kwargs.update({'num_classes':out_ch}) 
#         acc_kwargs.update({'num_classes':out_ch}) 

#         self.auc_roc = nn.ModuleDict({state:AUROC(**aucroc_kwargs) for state in ["train_", "val_", "test_"]}) # 'train' not allowed as key
#         self.acc = nn.ModuleDict({state:Accuracy(**acc_kwargs) for state in ["train_", "val_", "test_"]})

    
#     def _step(self, batch: dict, batch_idx: int, state: str, step: int):
#         target = batch['target']
#         # target = target[:,None].float()
#         batch_size = target.shape[0]
#         self.batch_size = batch_size 

#         # Run Model 
#         pred = self(**batch)

#         # ------------------------- Compute Loss ---------------------------
#         logging_dict = {}
#         logging_dict['loss'] = self.compute_loss(pred, target)

#         # --------------------- Compute Metrics  -------------------------------
#         with torch.no_grad():
#             # Aggregate here to compute for entire set later 
#             self.acc[state+"_"].update(pred, target)
#             self.auc_roc[state+"_"].update(pred, target) 
            
#             # ----------------- Log Scalars ----------------------
#             for metric_name, metric_val in logging_dict.items():
#                 self.log(f"{state}/{metric_name}", metric_val, batch_size=batch_size, on_step=True, on_epoch=True, 
#                          sync_dist=False) 

#         return logging_dict['loss'] 

#     def _epoch_end(self, state):
#         for name, value in [("ACC", self.acc[state+"_"]), ("AUC_ROC", self.auc_roc[state+"_"])]:
#             self.log(f"{state}/{name}", value.compute(), batch_size=self.batch_size, on_step=False, on_epoch=True, 
#                      sync_dist=True)
#             value.reset()

#     def compute_loss(self, pred, target):
#         return self.loss_func(pred, target)


import torch
import torch.nn as nn
# 引入所有需要的指标
from torchmetrics import Accuracy, AUROC, F1Score, Recall, Precision, Specificity

# class BasicClassifier(BasicModel):
#     def __init__(
#         self, 
#         in_ch,
#         out_ch,
#         spatial_dims,
#         loss = torch.nn.CrossEntropyLoss,
#         loss_kwargs = {},
#         optimizer=torch.optim.AdamW, 
#         optimizer_kwargs={'lr':1e-4, 'weight_decay':1e-2},
#         lr_scheduler= None, 
#         lr_scheduler_kwargs={},
#         aucroc_kwargs={"task":"multiclass"},
#         acc_kwargs={"task":"multiclass"},
#         save_hyperparameters=True,
#     ):
#         super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, save_hyperparameters)
#         self.in_ch = in_ch 
#         self.out_ch = out_ch 
#         self.spatial_dims = spatial_dims
#         self.loss_func = loss(**loss_kwargs)
#         self.loss_kwargs = loss_kwargs 

#         # 确保参数正确
#         aucroc_kwargs.update({'num_classes': out_ch}) 
#         acc_kwargs.update({'num_classes': out_ch}) 
        
#         # 定义通用参数：多分类模式，计算 Macro Average
#         # 对于二分类问题 (Tumor vs Normal)，Macro Average 会计算两类的平均值
#         metric_args = {'task': 'multiclass', 'num_classes': out_ch, 'average': 'macro'}

#         # === 定义所有指标 ===
#         # 使用 ModuleDict 方便管理 train/val/test 三个阶段
#         self.metrics = nn.ModuleDict({
#             "ACC":  nn.ModuleDict({s: Accuracy(**acc_kwargs) for s in ["train_", "val_", "test_"]}),
#             "AUC_ROC":  nn.ModuleDict({s: AUROC(**aucroc_kwargs) for s in ["train_", "val_", "test_"]}),
#             "Sens": nn.ModuleDict({s: Recall(**metric_args) for s in ["train_", "val_", "test_"]}),      # Sensitivity = Recall
#             "Spec": nn.ModuleDict({s: Specificity(**metric_args) for s in ["train_", "val_", "test_"]}), # Specificity
#             "Prec": nn.ModuleDict({s: Precision(**metric_args) for s in ["train_", "val_", "test_"]}),   # Precision
#             "F1":   nn.ModuleDict({s: F1Score(**metric_args) for s in ["train_", "val_", "test_"]}),     # F1-Score
#         })

#     def _step(self, batch: dict, batch_idx: int, state: str, step: int):
#         target = batch['target']
#         batch_size = target.shape[0]
#         self.batch_size = batch_size 

#         # Run Model 
#         pred = self(**batch)

#         # Compute Loss 
#         logging_dict = {}
#         logging_dict['loss'] = self.compute_loss(pred, target)

#         # === Update Metrics (每个 Batch 更新) ===
#         with torch.no_grad():
#             # 遍历所有指标类型 (ACC, AUC, Sens...)
#             for metric_name, metric_dict in self.metrics.items():
#                 # 获取当前阶段(train/val/test)的指标对象并更新
#                 metric_dict[state+"_"].update(pred, target)
            
#             # Log Loss
#             self.log(f"{state}/loss", logging_dict['loss'], batch_size=batch_size, on_step=True, on_epoch=True, sync_dist=False) 

#         return logging_dict['loss'] 

#     def _epoch_end(self, state):
#         # === Compute & Log Metrics (Epoch 结束时计算) ===
#         for metric_name, metric_dict in self.metrics.items():
#             metric_obj = metric_dict[state+"_"]
            
#             # 计算最终结果
#             try:
#                 score = metric_obj.compute()
#                 # Log 到控制台和 WandB
#                 # prog_bar=True 会让它显示在训练进度条上
#                 self.log(f"{state}/{metric_name}", score, batch_size=self.batch_size, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
#             except Exception as e:
#                 print(f"Warning: Metric {metric_name} failed to compute: {e}")
            
#             # 重置状态
#             metric_obj.reset()

#     def compute_loss(self, pred, target):
#         return self.loss_func(pred, target)
    
class BasicClassifier(BasicModel):
    def __init__(
        self, 
        in_ch,
        out_ch,
        spatial_dims,
        loss=torch.nn.CrossEntropyLoss,
        loss_kwargs={},
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr': 1e-4, 'weight_decay': 1e-2},
        lr_scheduler=None, 
        lr_scheduler_kwargs={},
        save_hyperparameters=True,
    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            save_hyperparameters=save_hyperparameters
        )

        self.in_ch = in_ch 
        self.out_ch = out_ch 
        self.spatial_dims = spatial_dims

        self.loss_func = loss(**loss_kwargs)
        self.loss_kwargs = loss_kwargs 

        if out_ch != 2:
            raise ValueError(
                "This BasicClassifier version is written for binary classification only. "
                "Expected out_ch=2."
            )

        # Binary metrics:
        # target: 0 = Normal, 1 = Tumor
        # pred_for_metric: probability of class 1, shape [N]
        binary_metric_args = {
            "task": "binary",
            "threshold": 0.5,
        }

        self.metrics = nn.ModuleDict({
            "ACC": nn.ModuleDict({
                s: Accuracy(**binary_metric_args)
                for s in ["train_", "val_", "test_"]
            }),

            "AUC_ROC": nn.ModuleDict({
                s: AUROC(task="binary")
                for s in ["train_", "val_", "test_"]
            }),

            # Sensitivity = Recall of positive class, here Tumor = 1
            "Sens": nn.ModuleDict({
                s: Recall(**binary_metric_args)
                for s in ["train_", "val_", "test_"]
            }),

            # Specificity = True Negative Rate, here correctly identifying Normal = 0
            "Spec": nn.ModuleDict({
                s: Specificity(**binary_metric_args)
                for s in ["train_", "val_", "test_"]
            }),

            "Prec": nn.ModuleDict({
                s: Precision(**binary_metric_args)
                for s in ["train_", "val_", "test_"]
            }),

            "F1": nn.ModuleDict({
                s: F1Score(**binary_metric_args)
                for s in ["train_", "val_", "test_"]
            }),
        })

        self.batch_size = None

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        target = batch["target"]

        # 保证 target 是 [N] 的 long tensor
        # CrossEntropyLoss 需要 target 为 long 类型
        if target.ndim > 1:
            target = target.view(-1)

        target = target.long()

        batch_size = target.shape[0]
        self.batch_size = batch_size

        # pred 是 raw logits, shape [N, 2]
        pred = self(**batch)

        # loss 直接用 raw logits
        loss = self.compute_loss(pred, target)

        # metric 用 class 1 的概率
        # 假设 class 1 = Tumor / Positive
        with torch.no_grad():
            pred_prob_pos = torch.softmax(pred, dim=1)[:, 1]

            for metric_name, metric_dict in self.metrics.items():
                metric_dict[state + "_"].update(pred_prob_pos, target)

            self.log(
                f"{state}/loss",
                loss,
                batch_size=batch_size,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                prog_bar=True,
            )

        return loss

    def _epoch_end(self, state):
        for metric_name, metric_dict in self.metrics.items():
            metric_obj = metric_dict[state + "_"]

            try:
                score = metric_obj.compute()

                self.log(
                    f"{state}/{metric_name}",
                    score,
                    batch_size=self.batch_size,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    prog_bar=True,
                )

            except Exception as e:
                print(f"Warning: Metric {metric_name} failed to compute: {e}")

            metric_obj.reset()

    def compute_loss(self, pred, target):
        return self.loss_func(pred, target)
