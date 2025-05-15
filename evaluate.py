import os
import importlib
import time
from PIL import Image
from arguements import get_arguments

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor, ToPILImage

from src.data.loader import get_test_dataloader
from src.data.transform import Relabel, ToLabel, Colorize
from src.utils.iouEval import iouEval, getColorEntry, getColorEntry

from src.models.segformer.model import mit_b0, mit_b2
from src.models.get_model import get_model, load_segformer_weights

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(42) # Set random seed for reproducibility
    device = "cuda" if not args.cpu else "cpu"
    if(not os.path.exists(args.datadir)): 
        print ("Error: datadir could not be loaded")
        
    # Dataloder
    loader = get_test_dataloader(args) 

    # Load model to evaluate, if multiple GPUs are available, use DataParallel
    model = get_model(args)
    model = load_segformer_weights(model, args.weightspath, device=device)
    if (not args.cpu): 
        model = torch.nn.DataParallel(model).cuda()
        print("model loaded to multiple GPUs")
    print ("Model and weights LOADED successfully")
    model.to(device)
    model.eval()

    iouEvalVal = iouEval(args.num_classes) # Metric class load

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            _,outputs,_ = model(images)

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)
        filenameSave = filename[0].split("leftImg8bit/")[1] 

        # print (step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    args = get_arguments()
    main(args)