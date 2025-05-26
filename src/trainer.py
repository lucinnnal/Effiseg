
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
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    def compute_loss(self, model, inputs, num_items_in_batch = None, return_outputs=False):
        image = inputs['pixel_values'].to(self.device)
        label = inputs['labels'].to(self.device)
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