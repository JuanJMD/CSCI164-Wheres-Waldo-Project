from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import os
import re
import json

def main():
    # Obtains the paths to the training and validation data
    train_path = os.path.join('Where Is Wally.v1-wally.coco', 'train')
    val_path = os.path.join('Where Is Wally.v1-wally.coco', 'valid')
    # Obtain paths for the annotations file
    t_annotations = os.path.join('Where Is Wally.v1-wally.coco', 'train', '_annotations.coco.json')
    v_annotations = os.path.join('Where Is Wally.v1-wally.coco', 'valid', '_annotations.coco.json')

    # Checks if the files exist
    for path in [train_path, val_path, t_annotations, v_annotations]:
        assert os.path.exists(path), f"Path {path} does not exist"

    # Transform pipeline is established
    #   - Images are resized to 800x800
    transform = tf.Compose([
        tf.Resize((800, 800)),
        tf.ToTensor()
    ])

    # Training Dataset is created using CocoDetection with the path of training data and annotations
    #   - Applied to the transform pipeline
    train_dataset = CocoDetection(root = train_path, annFile = t_annotations, transform = transform)
    
    # Validation Dataset is created using CocoDetection with the path of validation data and annotations
    #   - Applied to the transform pipeline
    val_dataset = CocoDetection(root = val_path, annFile = v_annotations, transform = transform)

    # Creates dataloader for each dataset
    train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size = 2, shuffle = True, num_workers = 4)

    # A R-CNN model is loaded
    baseModel = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # Since we have no GPU, we default to CPU
    device = torch.device('cpu')
    optimizer = torch.optim.SGD(baseModel.parameters(), lr=0.005, momentum=0.9)


    for epoch in range(10):
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            print(images.shape())
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            for target in targets:
                loss_dict = baseModel(images, target)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()


    # TESTING PHASE
    imagePath = os.path.join('Hey-Waldo-master', '256', 'waldo', '5_0_1.jpg')

    loadedImage = Image.open(imagePath)
    transform = tf.ToTensor()
    img = transform(loadedImage)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        baseModel.eval()
        prediction = baseModel([img])



    boundaryBoxes = prediction[0]['boxes']
    accurScores = prediction[0]['scores']
    accurScore_labels = prediction[0]['labels']
    coco_names = ["waldo", "person" , "bicycle" , "car" , "motorcycle" , "airplane" , 
                "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , 
                "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , 
                "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , 
                "eye glasses" , "handbag" , "tie" , "suitcase" , 
                "frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
                "baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
                "plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
                "banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
                "pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
                "mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
                "laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
                "oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
                "clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]

    imgGraph = cv2.imread(imagePath)
    imgGraph = cv2.cvtColor(imgGraph, cv2.COLOR_BGR2RGB)
    for i in range(10):
        x1, x2, y1, y2 = boundaryBoxes[i].numpy().astype(int)
        print(x1, x2, y1, y2)
        
        classification = coco_names[accurScore_labels[i].item() - 1]
        imgGraph = cv2.rectangle(imgGraph, (x1, y1), (x2, y2), (0, 255, 0), 1)
        imgGraph = cv2.putText(imgGraph, classification, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    plt.imshow(imgGraph)
    plt.show()

if __name__ == '__main__':
    main()
