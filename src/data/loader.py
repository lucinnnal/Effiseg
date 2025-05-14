import sys
import os
import argparse

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from torch.utils.data import DataLoader
from dataset import Traindataset, Testdataset, Cotransform, get_test_transform

import cv2

from PIL import Image, ImageOps
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage, InterpolationMode
from transform import Relabel, ToLabel, Colorize

# args.data_dir = "../data/cityscapes"

def get_train_dataloader(args):
    cotransformer = Cotransform(augment=args.augmentation, height=args.size, model=args.model)
    dataset = Traindataset(args.data_dir, co_transform=cotransformer, subset='train')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    return train_loader

def get_val_dataloader(args):
    cotransformer = Cotransform(augment=False, height=args.size, model=args.model)
    dataset = Traindataset(args.data_dir, co_transform=cotransformer, subset='val')
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return val_loader

def get_test_dataloader(args):
    input_transformer, output_transformer = get_test_transform()
    dataset = Testdataset(args.data_dir, input_transform=input_transformer, target_transform=output_transformer, subset='test')
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return test_loader

# 예시를 위해 넣어둠(main에서 사용)
def get_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description="Cityscapes Dataloader Arguments")
    
    # Dataset
    parser.add_argument("--data-dir", type=str, default="../data/cityscapes",
                        help="Path to the directory containing the Cityscapes dataset")
    parser.add_argument("--size", type=int, default=512,
                        help="Comma-separated string with height and width of images")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of images sent to the network in one step")
    parser.add_argument("--augmentation", type=bool, default=True,
                        help="Apply data augmentation")
    parser.add_argument("--model", type=str, default="SegformerB0",
                        help="Model type to use for training")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    train_loader = get_train_dataloader(args)
    val_loader = get_val_dataloader(args)
    test_loader = get_test_dataloader(args)

    imgs, labels = next(iter(test_loader))
    breakpoint()