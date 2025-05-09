import sys
import logging
import warnings
warnings.filterwarnings("ignore")

from torch.utils import data
from dataset import CityScapes_DataSet
import numpy as np

def get_dataloader(args, split='train'):
    """
    CityScapes 데이터셋의 DataLoader를 생성하는 함수
    
    Args:
        args: 학습 관련 인자들을 담은 객체
        split (str): 'train' 또는 'val' (기본값: 'train')
    
    Returns:
        torch.utils.data.DataLoader: 설정된 DataLoader 객체
    """
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    
    if split == 'train':
        h, w = map(int, args.input_size.split(','))
        dataset = CityScapes_DataSet(
            root=args.data_dir,
            list_path='./dataset/list/cityscapes/train.lst',
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
        dataset = CityScapes_DataSet(
            root=args.data_dir,
            list_path='./dataset/list/cityscapes/val.lst',
            crop_size=(1024, 2048),
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

if __name__ == '__main__':
    args = TrainOptions().initialize()
    train_loader = get_dataloader(args, 'train')
    val_loader = get_dataloader(args, 'val')