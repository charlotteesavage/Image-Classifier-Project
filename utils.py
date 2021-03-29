import sys
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch import tensor
import torch.nn.functional as F

def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

def test_transformer(test_dir):
    tester_transforms= transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229,0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=tester_transforms)
    return test_data

def data_loader(data, train):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader
   
    
def model(architecture, learning_rate, device, hidden_layers):
    if not hasattr(models, architecture):
        print("Please only use VGG Models. Attempt to use non-standard architecture")
        return None
    
    model_call = getattr(models, architecture)
    model = model_call(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False 
        
    classifier = nn.Sequential(nn.Linear(25088, int(hidden_layers[0])),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(int(hidden_layers[0], int(hidden_layers[1]),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(int(hidden_layers[1],int(hidden_layers[2]),
                               nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    if device == "cuda" and torch.cuda.is_available(): model.to("cuda")
    else: model.to("cpu")

    return model, criterion, optimizer

def train(model,criterion, optimizer, epochs, device, trainloader, validloader, print_every = 10):
    steps = 0
                                         
    for e in range(epochs):
        running_loss = 0
        
        for images, labels in trainloader:

            steps +=1
            images, labels = next(iter(trainloader))
            if device == "cuda" and torch.cuda.is_available(): images.to("cuda") and labels.to("cuda")
            else: images.to("cpu") and labels.to("cpu")                           
             
#            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
  
            running_loss += loss.item()
    
            if steps % print_every ==0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for vimages, vlabels in validloader:
                                         if device == "cuda" and torch.cuda.is_available(): vimages.to("cuda") and vlabels.to("cuda")
                                         else: vimages.to("cpu") and vlabels.to("cpu")
#                        vimages, vlabels = vimages.to(device), vlabels.to(device)
                        
                                         output = model.forward(vimages)
                                         batch_loss = criterion(output, vlabels)
                                         valid_loss += batch_loss.item()
                 
                                         ps = torch.exp(output)
                                         top_p, top_class = ps.topk(1, dim=1)
#                        print('Class shape: ', top_class.shape, ' and labels class: ', vlabels.shape)
                                         equals = top_class == vlabels.view(*top_class.shape)
                        
                                         accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
               
                    print(f"Epoch {e+1}/{epochs} ",
                          "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                          "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                          "Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                    running_loss = 0
                    model.train()

def model_test(model,criterion, optimizer, device, testloader):
    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
                
            output = model.forward(images)
            test_batch_loss = criterion(output, labels)
                
            test_loss += test_batch_loss.item()
                 
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
#            print('Class shape: ', top_class.shape, ' and labels class: ', labels.shape)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

def make_checkpoint(model, arch, hidden_units, optimizer, epochs, learning_rate, dropout):
    checkpoint = {'hidden_units': hidden_units,
                  'model' : arch,
                  'epochs': epochs, 
                  'learning_rate': learning_rate,
                  'dropout': dropout,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict()}
    
    torch.save(checkpoint, 'checkpoint.pth')
