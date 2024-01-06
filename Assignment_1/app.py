import os
import streamlit as st
from PIL import Image

import torch
from model_architecture import SimpleCNN
from  torchvision import  transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleCNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()


def predict(raw_image):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(raw_image)
    image = transform(image)
    grey = transforms.Grayscale()
    image = grey(image)
    image=image.unsqueeze(0)
    a=model(image)
    return int(torch.argmax(a))


# centering the image
def center(image, caption) : 
    col1, col2, col3 = st.columns([1.5,2,1.5])

    with col1:
        st.write("")

    with col2:
        st.image(image, caption=caption, width= 150 )

    with col3:
        st.write("")

def save_image_and_prediction(image, prediction, destination_folder, index):

    image_path = os.path.join(destination_folder, f"image_{index}.png")
    pil_image = Image.open(image)
    pil_image.save(image_path)

    # Save the prediction
    prediction_path = os.path.join(destination_folder, "predictions.txt")
    with open(prediction_path, "a") as f:
        f.write(f"Image {index}: Predicted Label - {prediction}\n")   

st.title("AI/ML Lab 1 - Digit Recognition")
image = Image.open("image.jpg")

# st.image(image, caption="Prediction : 2")
center(image, "Prediction : 2" )
cnt = 0 
raw_image = st.file_uploader("Choose an image...", type="jpg")

if raw_image is not None:
    # st.image(raw_image, caption="Uploaded Image.", use_column_width=True)
    prediction = predict(raw_image)
    center(raw_image, caption = f"Prediction : {prediction}")

    flag = st.button("Is Wrong Prediction ?")

    if flag :
        target = st.number_input("What is the expected target ?")
        save_image_and_prediction(raw_image, target, "./wrong_result", cnt )
        cnt += 1 
        if str(target):
            st.write("Thank you")

