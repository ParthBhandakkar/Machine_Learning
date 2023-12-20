#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load a pre-trained depth estimation model (e.g., MiDaS)
model = torch.hub.load("intel-isl/MiDaS", "MiDaS")

# Set the model to evaluation mode
model.eval()

# Load an input image (replace with your image)
image_path = "car-49278_1280.jpg"
input_image = Image.open(image_path)

# Resize the input image to match the expected size (384x384 pixels)
resize_transform = transforms.Resize((384, 384))
input_image = resize_transform(input_image)

# Apply other necessary preprocessing
preprocess = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])

input_tensor = preprocess(input_image).unsqueeze(0)

# Perform depth estimation
with torch.no_grad():
    prediction = model(input_tensor)

# Post-process the depth map for visualization
depth_map = prediction.squeeze().cpu().numpy()

# Visualize the depth map
plt.imshow(depth_map, cmap='plasma')
plt.colorbar()
plt.title("Depth Map")
plt.show()


# In[5]:


import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load a pre-trained depth estimation model (e.g., MiDaS)
model = torch.hub.load("intel-isl/MiDaS", "MiDaS")

# Set the model to evaluation mode
model.eval()

# Load an input image (replace with your image)
image_path = "truck-2181037_1280.png"
input_image = Image.open(image_path)

# Resize the input image to match the expected size (384x384 pixels)
resize_transform = transforms.Resize((384, 384))
input_image = resize_transform(input_image)

# Apply other necessary preprocessing
preprocess = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])

input_tensor = preprocess(input_image).unsqueeze(0)

# Perform depth estimation
with torch.no_grad():
    prediction = model(input_tensor)

# Post-process the depth map for visualization
depth_map = prediction.squeeze().cpu().numpy()

# Visualize the depth map
plt.imshow(depth_map, cmap='plasma')
plt.colorbar()
plt.title("Depth Map")
plt.show()


# In[ ]:




