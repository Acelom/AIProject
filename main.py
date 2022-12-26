import tkinter
from tkinter import filedialog
import os
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.io import read_image
from PIL import Image
import torch.nn as nn
import cv2

if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()
    currdir = os.getcwd()
    file = filedialog.askopenfile(parent = root, initialdir = currdir, title = "Please Select an Image of a Rabbit")
    torch.manual_seed(19)
    dir(models)
    image = Image.open(file.name)
    imagecv = cv2.imread(file.name)

    transform = transforms.Compose([transforms.Resize(224),  # resize to 224x?
                                    transforms.CenterCrop(224),  # take a square (224x224) crop from the centre
                                    transforms.ToTensor(),  # convert data to torch.FloatTensor
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    model = models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    model.cuda()
    model.load_state_dict(torch.load("resnet18_model_fine_tune_aug.pt"))
    model.eval()
    image = transform(image)
    image = image.cuda()
    image = image.unsqueeze(0)
    output = model(image)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    if preds == 0:
        breed = "American Fuzzy Lop"
        wild = "Domestic."
    elif preds == 1:
        breed = "Blanc De Hotot"
        wild = "Domestic."
    elif preds == 2:
        breed = "Checkered Giant"
        wild = "Domestic."
    elif preds == 3:
        breed = "Cottontail"
        wild = "Wild."
    elif preds == 4:
        breed = "Dutch"
        wild = "Domestic."
    elif preds == 5:
        breed = "English Lop"
        wild = "Domestic."
    elif preds == 6:
        breed = "Jackrabbit"
        wild = "Wild."
    elif preds == 7:
        breed = "Netherland Dwarf"
        wild = "Domestic."
    elif preds == 8:
        breed = "Silver Marten"
        wild = "Domestic."

    intro = "This rabbit probably is a "
    second = "and it is probably "
    firstText = intro + breed
    secondText = second + wild
    height, width, channels = imagecv.shape
    cv2.rectangle(imagecv, (0, height - 80), (300 + len(breed) * 6, height), (255, 255, 255), -1)
    cv2.putText(imagecv, firstText, (10, int(height - 50)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
    cv2.putText(imagecv, secondText, (10, int(height - 25)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
    cv2.imshow("show", imagecv)
    cv2.waitKey()
