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

from configs.train_arguements import get_arguments

from src.data.get_dataset import get_dataset
from src.models.get_model import get_model
from src.trainer import BaseTrainer
from src.utils.compute_metrics import compute_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"device : {device}")
    set_seed(42)

    ## =================== Data =================== ##
    train_dataset, val_dataset = get_dataset(args)

    ## =================== Model =================== ##
    model = get_model(args)
    model.to(device)

    ## =================== Trainer =================== ##
    wandb.init(project='Effiseg', name=f'{args.save_dir}')

    training_args = TrainingArguments(
        output_dir=f"./ckpt/{args.save_dir}",
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size, 
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=6e-5, # default optimizer : AdamW with betas=(0.9, 0.999), eps=1e-8
        logging_steps=10,
        metric_for_best_model="mean_iou",
        save_strategy="steps",
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
        data_collator=None,
    )

    trainer.train()


if __name__=="__main__":

    args = get_arguments()
    main(args)