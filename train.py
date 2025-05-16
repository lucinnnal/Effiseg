import os
import sys
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
import transformers
from transformers import TrainingArguments
import wandb

from arguments import get_arguments

from src.trainer import Trainer
from src.dataset.get_dataset import get_dataset
from src.model.get_model import get_model
from src.trainer import BaseTrainer
from src.utils.compute_metrics import compute_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(device)
    set_seed(42)

    ## =================== Data =================== ##
    train_dataset, val_dataset, data_collator = get_dataset(args)

    ## =================== Model =================== ##
    model = get_model(args)

    ## =================== Trainer =================== ##

    wandb.init(project='Effiseg', name=f'{args.save_dir}')

    training_args = TrainingArguments(
        output_dir=f"./ckpt/{args.save_dir}",
        eval_strategy='epoch',
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        metric_for_best_model="mean_iou",
        save_strategy="epoch",
        save_total_limit=None,
        save_steps=args.save_steps,
        remove_unused_columns=False,
        report_to="wandb",
        dataloader_num_workers=0,
    )

    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    
    trainer.train()


if __name__=="__main__":

    args = get_arguments()
    main(args)