import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import streamlit as st

class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.Conv1 = nn.Conv2d(1, 10, kernel_size = 5 )
    self.Conv2 = nn.Conv2d(10, 20, kernel_size = 3)
    self.Conv2_drop = nn.Dropout2d()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(500, 100)
    self.fc2 = nn.Linear(100, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.Conv1(x), 2))
    x = F.relu(F.max_pool2d(self.Conv2_drop(self.Conv2(x)), 2))
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = F.log_softmax(self.fc2(x), dim=1)
    return x


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Ensure images are in RGB format

        label = int(self.labels_df.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label
    
def center(image, caption) : 
  col1, col2, col3 = st.columns([1.5,2,1.5])

  with col1:
      st.write("")

  with col2:
      st.image(image, caption=caption, width= 150 )

  with col3:
      st.write("")

def save_image_and_prediction(image, prediction, destination_folder, index):
    
    images_folder = os.path.join(destination_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)
    image_path = os.path.join(images_folder, f"image_{index}.png")
    pil_image = Image.open(image)
    pil_image.save(image_path)

    prediction_path = os.path.join(destination_folder, "predictions.csv")
    with open(prediction_path, "a") as f:
        f.write(f"Image_{index}.png, {prediction}\n") 