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
    train_path = os.path.join('Where Is Wally.v1-wally.coco', 'train')
    val_path = os.path.join('Where Is Wally.v1-wally.coco', 'valid')
    t_annotations = os.path.join('Where Is Wally.v1-wally.coco', 'train', '_annotations.coco.json')
    v_annotations = os.path.join('Where Is Wally.v1-wally.coco', 'valid', '_annotations.coco.json')

    assert os.path.exists(train_path), f"Training path {train_path} does not exist"
    assert os.path.exists(val_path), f"Validation path {val_path} does not exist"
    assert os.path.exists(t_annotations), f"Training annotations file {t_annotations} does not exist"
    assert os.path.exists(v_annotations), f"Validation annotations file {v_annotations} does not exist"

    transform = tf.Compose([
        tf.Resize((800, 800)),
        tf.ToTensor()
    ])

    train_dataset = CocoDetection(root = train_path, annFile = t_annotations, transform = transform)
    for i in range(10):
        image, target = train_dataset[i]
        print(f"Image shape: {image.shape}, Target: {target}")

    val_dataset = CocoDetection(root = val_path, annFile = v_annotations, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size = 2, shuffle = True, num_workers = 4)

    baseModel = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cpu')
    optimizer = torch.optim.SGD(baseModel.parameters(), lr=0.005, momentum=0.9)

    print(len(train_loader))
    for images, targets in train_loader:
        print(len(images), len(targets))
        #if all(len(target) == 0 for target in targets):
        #    continue
        print(f"Number of images: {len(images)}, Number of targets: {len(targets)}")

        images = list(image.to(device) for image in images)
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        for i in range(len(targets)):
            bbox = targets[i]["bbox"]
            x, y, width, height = bbox
            targets[i]["boxes"] = torch.tensor([int(x), int(y), int(x + width), int(y + height)]).to(device)
            del targets[i]["bbox"]
        optimizer.zero_grad()

        lost_dictation = baseModel(images, targets)
        losses= sum(loss for loss in lost_dictation.values())
        losses.backward()
        optimizer.step()

    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_images = list(image.to(device) for image in val_images)
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

            val_loss_dict = baseModel(val_images, val_targets)
            val_losses = sum(loss for loss in val_loss_dict.values())

    #baseModel = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

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
