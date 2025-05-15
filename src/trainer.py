
import torch
import numpy as np
from torch import nn

from transformers.trainer import Trainer
from src.utils.loss import CrossEntropyLoss2d

from typing import Dict, List, Tuple, Optional, Any, Union

class BaseTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.loss_fn = CrossEntropyLoss2d()
        
    def compute_loss(self, model, inputs, num_items_in_batch = None, return_outputs=False):
        image = inputs['image']
        label = inputs['label']
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
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model,inputs,return_outputs = True)
        
        return (eval_loss,pred,label)