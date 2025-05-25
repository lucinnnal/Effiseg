
import torch
import numpy as np
from torch import nn

from transformers.trainer import Trainer
from src.utils.loss import CrossEntropyLoss2d
from src.utils.poly import PolynomialLRDecay
from typing import Dict, List, Tuple, Optional, Any, Union


class BaseTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.loss_fn = CrossEntropyLoss2d()
        
    def compute_loss(self, model, inputs, num_items_in_batch = None, return_outputs=False):
        image = inputs['piexel_values']
        label = inputs['labels']
        # Forward pass
        output = model(image)
        # Compute loss
        loss = self.loss_fn(output, label)

        if return_outputs:
            return loss, output, label
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = ["patch_embeddings", "feature_map"],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: # Optional은 Tensor 또는 None이 올 수 있다는 뜻 
        
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model,inputs,return_outputs = True)
        
        return (eval_loss,pred,label)
    
    def create_scheduler(self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None):
        if optimizer is None:
            optimizer = self.optimizer

        self.lr_scheduler = PolynomialLRDecay(
            optimizer,
            max_decay_steps=num_training_steps,
            end_learning_rate=self.args.learning_rate,
            power=1.0
        )
        
        return self.lr_scheduler