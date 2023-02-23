import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Model():
    model = torchvision.models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Linear(256, 3),
    )
    return model.to(device)

class DEPTH:
	def __init__(self,path):
		self.train_transform = transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		self.checkpoint= torch.load(path,map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		self.model=Model()
		self.model.load_state_dict(self.checkpoint['state_dict'])
		self.model.eval()
		self.thr = 0.3

	def image_loader(self,image_name):
		print(image_name)
		image=cv2.cvtColor(image_name, cv2.COLOR_BGR2RGB)
		image=cv2.resize(image,(224,224))
		image= Image.fromarray(image)
		image=self.train_transform(image).float()
		image=image.unsqueeze(0)
		return image
 

	def infer(self,img,id_=None):
		print(type(img))
		imgs=self.image_loader(img)
		#print(imgs.shape)
		outputs=self.model(imgs)
		#print(outputs[0])
		#print(outputs.shape)
		#print(outputs)
		npy = outputs[0].detach().numpy()
		max_pos = npy.argmax(axis =0)
		#print(max_pos)
#         probabilities = nn.functional.softmax(outputs, dim=1)
# 		predicted = torch.max(outputs[0], 1)
		#print(torch.max(outputs[0], 1))
		#print(predicted)
#         class_names = ['Bed', 'Chair', 'Sofa']
# 		print('Predicted class:', class_names[predicted.item()])
# 		print(predicted.item())
		if max_pos == 1:
			return "Chair"
		elif max_pos ==2:
			return "Sofa"
		else: 
			return "Bed"

if __name__ == "__main__":
	a = DEPTH(r'D:\work\fulhas\checkpoints\checkpoint_37.ckpt')
	image = cv2.imread(r'C:\Users\ra_saval\Desktop\sofa-furniture-isolated-on-white-600w-238184509.jpg')
	b = a.infer(image,"temp1")
	print(b)
