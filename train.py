# import things
import sys
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch import tensor
import torch.nn.functional as F

from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str)
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=[5120, 1024, 102])
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=10)
parser.add_argument('--learning_rate', dest="learning_rate", action="store", type=float, default=0.001)
parser.add_argument('--cpu', dest="cpu", action="store", default="cuda")
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)

def main():
    args = parser.parse_args()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    print('Parsed arguments: ', args)
        
    # Pass transforms in, then create trainloader
    train_data = train_transformer(train_dir)
    valid_data = test_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    
    trainloader = data_loader(train_data, train=True)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    model_to_train, criterion, optimizer = model(args.arch, args.learning_rate, args.cpu)

    train(model_to_train, criterion, optimizer, args.epochs, args.cpu, trainloader, validloader)
    
    model_test(model_to_train, criterion, optimizer, args.cpu, testloader)
    
    model_to_train.class_to_idx = train_data.class_to_idx
    make_checkpoint(model_to_train, args.arch, args.hidden_units, optimizer, args.epochs, args.learning_rate, args.dropout)

if __name__== "__main__":
    main()