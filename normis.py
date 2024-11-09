# Import required libraries
import os
import shutil
from tqdm import tqdm  
import numpy as np
import cv2  
from deepface import DeepFace  
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor  # For CPU-bound tasks
import random
import torch  
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Function to remove small folders containing fewer than min_files files
def remove_small_folders(dir_path, min_files):
    if not os.path.exists(dir_path):
        print(f"Directory '{dir_path}' does not exist.")
        return

    for entry in os.scandir(dir_path):
        if entry.is_dir():
            folder_path = entry.path
            file_count = sum(1 for f in os.scandir(folder_path) if f.is_file())
            if file_count < min_files:
                print(f'Removing folder: {folder_path}')
                try:
                    shutil.rmtree(folder_path)
                except OSError as e:
                    print(f"Error removing {folder_path}: {e}")

# Function to visualize face extraction for a given image path
def visualize_face_extraction(image_path):
    try:
        # Ensure 'yolov8' backend is properly installed
        face_obj = DeepFace.extract_faces(img_path=image_path, detector_backend='yolov8')
        if face_obj:
            face_img = face_obj[0]['face']
            face_img = cv2.resize(face_img, (224, 224))

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            original_img = plt.imread(image_path)

            axs[0].imshow(original_img)
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            axs[1].imshow(face_img)
            axs[1].set_title('Detected Face')
            axs[1].axis('off')

            plt.show()
            print(f"Detected face shape: {face_img.shape}")
        else:
            print("No face detected in the image.")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Function to process an individual image for face extraction
def process_image(src_image_path, dest_image_path, min_faces, max_faces):
    try:
        face_objs = DeepFace.extract_faces(img_path=src_image_path, detector_backend='yolov8')
        if min_faces <= len(face_objs) <= max_faces:
            face_image = face_objs[0]['face']

            # Ensure face_image is in uint8 format with values [0,255]
            if face_image.dtype != np.uint8:
                face_image = (face_image * 255).astype(np.uint8)

            # Save face image in BGR format
            bgr_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dest_image_path, bgr_face_image)
    except Exception as e:
        print(f"Error processing image {src_image_path}: {e}")

# Function to crop faces from images in source directory and save to destination directory
def crop_and_save_faces(src_directory, dest_directory, min_faces=1, max_faces=1):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_image = {}
        for foldername in tqdm(os.listdir(src_directory), desc="Processing folders"):
            src_folder_path = os.path.join(src_directory, foldername)
            dest_folder_path = os.path.join(dest_directory, foldername)

            if os.path.isdir(src_folder_path):
                if not os.path.exists(dest_folder_path):
                    os.makedirs(dest_folder_path)

                image_files = [f for f in os.listdir(src_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
                for filename in image_files:
                    src_image_path = os.path.join(src_folder_path, filename)
                    dest_image_path = os.path.join(dest_folder_path, filename)
                    future = executor.submit(process_image, src_image_path, dest_image_path, min_faces, max_faces)
                    future_to_image[future] = src_image_path

        # Collect results and handle exceptions
        for future in tqdm(future_to_image.keys(), desc="Processing images"):
            try:
                future.result()
            except Exception as e:
                src_image_path = future_to_image[future]
                print(f"Error processing image {src_image_path}: {e}")

# Dataset class for faces
class FaceDataset(Dataset):
    def __init__(self, root_dir, selected_folders, label_mapping, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapping = label_mapping
        self.image_paths, self.labels = self._load_data(selected_folders)
        
    def _load_data(self, selected_folders):
        image_paths = []
        labels = []
        
        for folder in selected_folders:
            folder_path = os.path.join(self.root_dir, folder)
            label = self.label_mapping[folder]
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg','.png', '.bmp', '.gif')):
                    image_paths.append(os.path.join(folder_path, filename))
                    labels.append(label)
        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])                    

# Triplet Dataset class
class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_indices = {}
        self.valid_indices = []

        # Organize indices by class label
        for idx, label in enumerate(dataset.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        # Keep only classes with at least 2 samples
        self.class_indices = {k: v for k, v in self.class_indices.items() if len(v) >= 2}
        
        # Generate a list of valid indices
        self.valid_indices = [idx for indices in self.class_indices.values() for idx in indices]
        
        # Check if we have any valid indices
        if len(self.valid_indices) == 0:
            raise ValueError("No valid classes with at least 2 samples each were found in the dataset.")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Select a valid index from filtered valid indices
        anchor_idx = self.valid_indices[idx]
        anchor_label = self.dataset.labels[anchor_idx]

        # Select a positive index that is not the same as the anchor index
        positive_idx = anchor_idx
        while positive_idx == anchor_idx:
            positive_idx = random.choice(self.class_indices[anchor_label])

        anchor_img = self.dataset[anchor_idx][0]
        positive_img = self.dataset[positive_idx][0]

        # Select a negative image from a different class
        negative_label = random.choice([label for label in self.class_indices.keys() if label != anchor_label])
        negative_idx = random.choice(self.class_indices[negative_label])
        negative_img = self.dataset[negative_idx][0]

        return anchor_img, positive_img, negative_img

# Loss function:
class TripletLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.alpha)
        
        return losses.mean()
    
    
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (anchors, positives, negatives) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        
        optimizer.zero_grad()
        
        anchor_embeddings = model(anchors)
        positive_embeddings = model(positives)
        negative_embeddings = model(negatives)
        
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    print(f'Train Epoch: \tLoss: {avg_loss:.4f}')
    return avg_loss


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (anchors, positives, negatives) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Validation'):
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)
            
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            running_loss += loss.item()
            
    avg_loss = running_loss / len(val_loader)
    print(f'Validation Epoch: \tLoss: {avg_loss:.4f}')
    return avg_loss

def unnormalize(img_tensor):
    img = img_tensor.clone().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

if __name__ == "__main__":
    dir_path = r'C:\Users\Meiras\Desktop\DL\face\lfw-deepfunneled'
    min_files_count = 4
    image_path = os.path.join(dir_path, 'Adam_Sandler', 'Adam_Sandler_0002.jpg')
    dest_directory_path = r'C:\Users\Meiras\Desktop\DL\face\lfw-deepfunneled_cropped_upd'

    # Remove small folders
    remove_small_folders(dir_path, min_files_count)

    # Visualize face extraction for a single image
    visualize_face_extraction(image_path)

    # Crop faces and save them to destination directory
    crop_and_save_faces(dir_path, dest_directory_path)
    
    # Create label mapping
    all_folders = sorted(os.listdir(dest_directory_path))
    label_mapping = {folder: idx for idx, folder in enumerate(all_folders)}

    # Split into train and validation sets
    random.seed(42)
    random.shuffle(all_folders)
    split_index = int(0.75 * len(all_folders))
    train_folders = all_folders[:split_index]
    val_folders = all_folders[split_index:]
    
    # Create datasets
    train_dataset = FaceDataset(dest_directory_path, train_folders, label_mapping, transform=train_transform)
    val_dataset = FaceDataset(dest_directory_path, val_folders, label_mapping, transform=val_transform)
    
    # Initialize the Triplet Datasets and DataLoaders
    train_triplet_dataset = TripletDataset(train_dataset)
    val_triplet_dataset = TripletDataset(val_dataset)

    train_loader = DataLoader(dataset=train_triplet_dataset, 
                            batch_size=16, 
                            shuffle=True)
    val_loader = DataLoader(dataset=val_triplet_dataset, 
                            batch_size=16, 
                            shuffle=False)

    # Visualize a batch of triplets
    anchors, positives, negatives = next(iter(val_loader))

    anchors_np = unnormalize(anchors[0])
    positives_np = unnormalize(positives[0])
    negatives_np = unnormalize(negatives[0])

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(anchors_np)
    axs[0].set_title('Anchor')
    axs[1].imshow(positives_np)
    axs[1].set_title('Positive')
    axs[2].imshow(negatives_np)
    axs[2].set_title('Negative')

    for ax in axs:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained MobileNetV2 model with weights
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model = model.features  # Use the feature extractor part of the model

    # Add adaptive pooling and flatten
    model = nn.Sequential(
        model,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()  # Flatten to (batch_size, 1280)
    )
    model = model.to(device)

    # Define loss function and optimizer
    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Validate before training
    print('Validation epoch before training:')
    validate_epoch(model, val_loader, criterion, device)

    # Training loop
    num_epochs = 10
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print('------------------------')
        print(f'Epoch {epoch+1}/{num_epochs} training:')
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
    # Plot the losses after training
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    # Save the model state dict
    torch.save(model.state_dict(), 'model_weights.pth')

    # Save the entire model
    torch.save(model, 'model.pth')

    # Try to script the model
    try:
        scripted_model = torch.jit.script(model)
        scripted_model.save('model_scripted.pt')
    except Exception as e:
        print(f"Failed to script the model: {e}")
