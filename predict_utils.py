import time
import PIL
from PIL import Image
import glob, os
import sys
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch import tensor
import torch.nn.functional as F
import numpy as np

#def checkpoint_load(filepath, architecture):
def checkpoint_load(filepath):
    checkpoint = torch.load(filepath)
    
#    model_call = getattr(models, architecture)
    model = getattr(models, checkpoint['model'])
    classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_units'][0]),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(checkpoint['hidden_units'][0], checkpoint['hidden_units'][1]),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(checkpoint['hidden_units'][1],checkpoint['hidden_units'][2]),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier

    for param in model.parameters():
        param.requires_grad = False
    #load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    image = Image.open(image)
    
    transformed_image = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    processed_image = transformed_image(image)
    return processed_image

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if device == "cuda" and torch.cuda.is_available(): model.to("cuda")
    else: model.to("cpu")
    
    model.eval()
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        if device == "cuda" and torch.cuda.is_available(): img.to("cuda")
        else: img.to("cpu")
        
        output = model.forward(img)
    
    prob = torch.exp(output)
        
    top_probs, top_indices = prob.topk(int(topk), dim=1)
    
    idx_to_class = {val: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices[0].tolist()]
    
    if device == 'cuda':
        top_prob, top_class = top_probs.cpu().numpy()[0], top_classes.cpu().numpy()[0]
    else:
        top_prob, top_class = top_probs.numpy()[0], top_classes.numpy()[0]
        
    return top_prob, top_class