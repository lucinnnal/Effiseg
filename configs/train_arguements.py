import argparse

def get_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description="Training Arguments")
    
    # Dataset
    parser.add_argument("--datadir", type=str, default="../data/cityscapes", help="Path to the directory containing the dataset")
    parser.add_argument("model", type=str, default="SegformerB0", help="Model type to use for training")
    parser.add_argument("--augmentation", type=bool, default=True, help="Apply data augmentation during training")
    parser.add_argument("--input_size", type=int, default=512, help="Length of the shorter side of the image")
    
    return parser.parse_args()