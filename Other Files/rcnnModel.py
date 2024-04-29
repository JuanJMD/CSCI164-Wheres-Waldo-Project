from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import transforms as tf
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import os
import re
import json

def getCoordinates(dataset, hasFile, coord_file, w, dir):
    tempDataset = dataset
    # Opens the coordinates file
    if(hasFile == True):
        with open(coord_file, 'r') as f:
            coords = json.load(f)
    imgData_pairs = []
    # Retrieves the image name and the coordinates
    # Appends them together and preps them for coord appointing
    for img in tempDataset:
        if(img != '.DS_Store'):
            pattern = r'\d+'
            digits = re.findall(pattern, img)
            if len(digits[0]) == 1:
                digits[0] = '0' + digits[0]
            imgData_pairs.append((img, digits)) 


    imgData_pairs = sorted(imgData_pairs, key=lambda x: x[1])

    files = []
    coordinates = []

    for img in imgData_pairs:
        getIMG = img[0]
        getCOORDS = img[1]
        x = int(getCOORDS[1])
        y = int(getCOORDS[2])        
        x1, x2 = (x - 1), (x + 1)
        y1, y2 = (y - 1), (y + 1)

        numericalCoords = list((x1, x2, y1, y2))
        #numericalCoords = list((x, y))
        print(numericalCoords)
        #dirStr = os.path.join(dir, getIMG)
        dirStr = getIMG
        files.append(dirStr)
        coordinates.append(numericalCoords)

    dataset = files
    return files, coordinates

   
class datasetFormation(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        waldo_dir = os.path.join('Hey-Waldo-master', '64', 'waldo')
        self.root = waldo_dir
        self.transforms = transforms
        self.imgs = sorted(os.listdir(waldo_dir))
        self.imgs, coords = getCoordinates(self.imgs, True, os.path.join('Hey-Waldo-master', 'data.json'), 64, waldo_dir)
        self.labels = coords

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        #target_path = os.path.join(self.root, self.labels[idx])
        img = Image.open(img_path).convert("RGB")
        #target = Image.open(target_path).convert("RGB")
        target = self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)
            target = torch.tensor(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
# Importing the images and coordinates
# Directories
if __name__ == '__main__':
    transform_waldo = tf.Compose([
        tf.ToTensor()
    ])

    dataset = datasetFormation(transform_waldo)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    rnnCustomModel = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 1 class (waldo) + background
    in_features = rnnCustomModel.roi_heads.box_predictor.cls_score.in_features
    rnnCustomModel.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(rnnCustomModel.parameters(), lr=0.001, momentum=0.9)

    device = torch.device('cpu')
    num_epochs = 10

    for epoch in range(num_epochs):
        rnnCustomModel.train()
        for imgs, targets in dataloader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{ 'boxes': t.to(device), 'labels': torch.ones((t.shape[0],), dtype=torch.int64).to(device) } for t in targets]
            loss_dict = rnnCustomModel(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

    imagePath = os.path.join('Hey-Waldo-master', '64', 'waldo', '6_15_1.jpg')

    loadedImage = Image.open(imagePath)
    transform = tf.ToTensor()
    img = transform(loadedImage)


    with torch.no_grad():
        rnnCustomModel.eval()
        prediction = rnnCustomModel([img])

    #baseModel = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    #imagePath = os.path.join('Hey-Waldo-master', '256', 'waldo', '5_0_1.jpg')

    #loadedImage = Image.open(imagePath)
    #transform = tf.ToTensor()
    #img = transform(loadedImage)

    #with torch.no_grad():
    #    baseModel.eval()
    #    prediction = baseModel([img])



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

