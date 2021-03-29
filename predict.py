# import things
import sys
import argparse
import torch
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch import tensor
import torch.nn.functional as F

from predict_utils import *

parser = argparse.ArgumentParser()

parser.add_argument('image', action="store", type = str, default='./flowers/test/101/image_07949.jpg')
parser.add_argument('checkpoint',type = str, action="store", default='./checkpoint.pth')
parser.add_argument('--topk', dest="topk", action="store", default="5")
parser.add_argument('--cpu', dest="cpu", action="store", default="cuda")
parser.add_argument('--category_names', action="store", default="cat_to_name.json")
#parser.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str)

def main():
    args = parser.parse_args()
    print(args)
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model = checkpoint_load(args.checkpoint)    

    top_probs, top_classes = predict(args.image, model, args.topk, args.cpu)
    
    flower_names = [cat_to_name[str(num)] for num in top_classes]
    
    print('Flower: ', flower_names[0], ' with probability: ', top_probs[0])
    return flower_names[0], top_probs[0]
    
if __name__== "__main__":
    main()