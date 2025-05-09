import sys
import logging
import warnings
warnings.filterwarnings("ignore")
import argparse
from torch.utils import data
from dataset import CityScapes_DataSet
import numpy as np

def get_dataloader(args, split='train'):

    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    
    if split == 'train':
        h, w = map(int, args.input_size.split(','))
        dataset = CityScapes_DataSet(
            root=args.data_dir,
            list_path='./train.lst',
            max_iters=args.num_steps * args.batch_size,
            crop_size=(h, w),
            scale=args.random_scale,
            mirror=args.random_mirror,
            mean=IMG_MEAN
        )
        
        dataloader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    elif split == 'val':
        h, w = map(int, args.input_size.split(','))
        dataset = CityScapes_DataSet(
            root=args.data_dir,
            list_path='./val.lst',
            crop_size=(h, w),
            mean=IMG_MEAN,
            scale=False,
            mirror=False
        )
        
        dataloader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )
    
    else:
        raise ValueError(f"Invalid split: {split}. Choose either 'train' or 'val'")
        
    return dataloader

# 예시를 위해 넣어둠(main에서 사용)
def get_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description="Cityscapes Dataloader Arguments")
    
    # Dataset
    parser.add_argument("--data-dir", type=str, default="",
                        help="Path to the directory containing the Cityscapes dataset")
    parser.add_argument("--input-size", type=str, default="512,512",
                        help="Comma-separated string with height and width of images")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of images sent to the network in one step")
    parser.add_argument("--num-steps", type=int, default=50000,
                        help="Number of training steps")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during training")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during training")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    train_loader = get_dataloader(args, 'train')
    val_loader = get_dataloader(args, 'val')

    imgs, labels, _, _ = next(iter(train_loader))
    breakpoint()