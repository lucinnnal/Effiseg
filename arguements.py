import argparse
from argparse import ArgumentParser

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True) 
    parser.add_argument('--model', default="SegformerB0")

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--dataset',default="cityscapes", choices=['ACDC','cityscapes','NYUv2'])
    parser.add_argument('--datadir', default="/path/to/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)   
    parser.add_argument('--savedir', default = 'ckpt')
    parser.add_argument('--savedate', default=True)
    parser.add_argument('--visualize', action='store_true',default=False)
    parser.add_argument('--distillation-type', default='ckpt', type=str, help="")
    parser.add_argument('--iouTrain', action='store_true', default=False) 
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument("--device", default='cuda', help="Device on which the network will be trained. Default: cuda")
    parser.add_argument('--student-pretrained',default= True)
    
    return parser