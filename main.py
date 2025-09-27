# SneakyPixels Streamlit App: MNIST Adversarial Attack (Trained Model)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import streamlit as st
from PIL import Image

# ------------------------
# 0. Setup
# ------------------------
os.makedirs("screenshots", exist_ok=True)
st.title("SneakyPixels: MNIST Adversarial Attack Demo")
st.write("Small pixel changes can fool a neural network.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# 1. Load MNIST
# ------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# ------------------------
# 2. Define CNN
# ------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.fc1 = nn.Linear(64*12*12,128)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,64*12*12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# ------------------------
# 3. Train the CNN quickly for demo
# ------------------------
if not os.path.exists("mnist_cnn.pth"):
    st.write("Training CNN for demo (1 epoch)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(1):  # 1 epoch demo
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "mnist_cnn.pth")
    st.write("Training finished and model saved!")
else:
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    st.write("Loaded pretrained CNN.")
model.eval()

# ------------------------
# 4. FGSM attack
# ------------------------
def fgsm_attack(image, epsilon, data_grad):
    return torch.clamp(image + epsilon*data_grad.sign(), 0, 1)

# ------------------------
# 5. Pixel-perfect upscale
# ------------------------
def upscale(img_tensor, scale=15):
    img = (img_tensor.squeeze().detach().cpu().numpy()*255).astype(np.uint8)
    img_up = np.kron(img, np.ones((scale,scale),dtype=np.uint8))
    img_rgb = np.stack([img_up]*3, axis=-1)
    return img_rgb

# ------------------------
# 6. Select image & epsilon
# ------------------------
index = st.slider("Select image index", 0, len(test_dataset)-1, 0)
epsilon = st.slider("Epsilon (perturbation size)", 0.0, 0.5, 0.25, 0.01)

image, label = test_dataset[index]
image = image.unsqueeze(0).to(device)
image.requires_grad = True

# ------------------------
# 7. Predict & attack
# ------------------------
outputs = model(image)
init_pred = outputs.max(1, keepdim=True)[1].item()

loss = F.nll_loss(F.log_softmax(outputs, dim=1), torch.tensor([label]).to(device))
model.zero_grad()
loss.backward()
perturbed = fgsm_attack(image, epsilon, image.grad.data)
final_pred = model(perturbed).max(1, keepdim=True)[1].item()

# ------------------------
# 8. Display images side-by-side
# ------------------------
col1, col2 = st.columns(2)
with col1:
    st.image(upscale(image), caption=f"Original - Predicted: {init_pred}")
with col2:
    st.image(upscale(perturbed), caption=f"Adversarial - Predicted: {final_pred}")

# ------------------------
# 9. Save screenshots
# ------------------------
if st.button("Save screenshots"):
    Image.fromarray(upscale(image)).save(f"screenshots/original_{index}.png")
    Image.fromarray(upscale(perturbed)).save(f"screenshots/adversarial_{index}.png")
    st.success(f"Screenshots saved in 'screenshots/' folder!")
