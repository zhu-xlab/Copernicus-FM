"""Base Class."""

from lightning import LightningModule
import torch
import pdb

class LightningTask(LightningModule):
    def __init__(self, args, model_config, data_config):
        super().__init__()
        self.model_config = model_config  # model_config
        self.args = args  # args for optimization params
        self.data_config = data_config  # dataset_config
        self.save_hyperparameters()

    def loss(self, outputs, labels):
        raise NotImplementedError(
            "This method should be implemented in task-specific classes"
        )

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True
    
    def freeze_non_lora_params(self, module):
        """
        Freeze the encoder parameters except for LoRA-specific ones.
        """
        for name, param in module.named_parameters():
            if "lora" not in name:  # Skip LoRA parameters
                param.requires_grad = False

    def log_metrics(self, outputs, targets, prefix="train"):
        """Abstract method for logging task-specific metrics."""
        raise NotImplementedError(
            "This method should be implemented in task-specific classes"
        )

    def forward(self, samples):
        raise NotImplementedError(
            "This method should be implemented in task-specific classes"
        )

    def training_step(self, batch, batch_idx):
        # current_lr = self.optimizers().param_groups[0]['lr']
        if len(batch)==3:
            images, targets, metas = batch
        else:
            images, targets = batch
            metas = torch.full((images.shape[0], 4), float('nan'))
        #pdb.set_trace()
        if self.model_config.task != "regression":
            targets = targets.long()
        if self.model_config.model_type == 'copernicusfm':
            outputs = self(images, metas)
        else:
            outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        #print("In validation_step, model.training =", self.training)
        #self.training = False
        #pdb.set_trace()
        if len(batch)==3:
            images, targets, metas = batch
        else:
            images, targets = batch
            metas = torch.full((images.shape[0], 4), float('nan'))

        if self.model_config.task != "regression":
            targets = targets.long()
        if self.model_config.model_type == 'copernicusfm':
            outputs = self(images,metas)
        else:
            outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="val")
        return loss

    def test_step(self, batch, batch_idx):
        #print("In test_step, model.training =", self.training)
        #self.training = False
        if len(batch)==3:
            images, targets, metas = batch
        else:
            images, targets = batch
            metas = torch.full((images.shape[0], 4), float('nan'))
        if self.model_config.task != "regression":
            targets = targets.long()
        if self.model_config.model_type == 'copernicusfm':
            outputs = self(images,metas)
        else:
            outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="test")
        return loss

    def configure_optimizers(self):
        if self.model_config.task == "classification":
            optimizer = torch.optim.SGD(
                self.params_to_optimize(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(self.params_to_optimize(), lr=self.args.lr)

        num_warmup_steps = (
            len(self.trainer.datamodule.train_dataloader())
            * self.args.warmup_epochs
            // self.args.num_gpus
        )
        total_steps = (
            len(self.trainer.datamodule.train_dataloader())
            * self.args.epochs
            // self.args.num_gpus
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.lr,
            total_steps=total_steps,
            anneal_strategy="cos",  # Cosine annealing
            pct_start=float(num_warmup_steps)
            / float(total_steps),  # Percentage of warmup
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
