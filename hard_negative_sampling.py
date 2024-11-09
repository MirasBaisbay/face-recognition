# Import 
import os
import shutil
import logging
from tqdm import tqdm
import numpy as np
import cv2
from deepface import DeepFace
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from collections import Counter
import json
import imagehash

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to remove small folders containing fewer than min_files files
def remove_small_folders(dir_path, min_files):
    if not os.path.exists(dir_path):
        logging.warning(f"Directory '{dir_path}' does not exist.")
        return

    for entry in os.scandir(dir_path):
        if entry.is_dir():
            folder_path = entry.path
            file_count = sum(1 for f in os.scandir(folder_path) if f.is_file())
            if file_count < min_files:
                logging.info(f'Removing folder: {folder_path}')
                try:
                    shutil.rmtree(folder_path)
                except OSError as e:
                    logging.error(f"Error removing {folder_path}: {e}")

# Function to visualize face extraction for a given image path
def visualize_face_extraction(image_path):
    try:
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
            logging.info(f"Detected face shape: {face_img.shape}")
        else:
            logging.info("No face detected in the image.")
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")

# Function to process an individual image for face extraction
def process_image(src_image_path, dest_image_path, min_faces, max_faces):
    try:
        face_objs = DeepFace.extract_faces(img_path=src_image_path, detector_backend='yolov8')
        if min_faces <= len(face_objs) <= max_faces:
            face_image = face_objs[0]['face']
            if face_image.dtype != np.uint8:
                face_image = (face_image * 255).astype(np.uint8)
            bgr_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dest_image_path, bgr_face_image)
    except Exception as e:
        logging.error(f"Error processing image {src_image_path}: {e}")

# Function to crop faces from images in source directory and save to destination directory
def crop_and_save_faces(src_directory, dest_directory, min_faces=1, max_faces=1):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
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

        for future in tqdm(future_to_image.keys(), desc="Processing images"):
            try:
                future.result()
            except Exception as e:
                src_image_path = future_to_image[future]
                logging.error(f"Error processing image {src_image_path}: {e}")

# Function to detect duplicates based on perceptual image hash
def find_duplicates(directory):
    logging.info("Checking for duplicates...")
    image_hashes = {}
    duplicates = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    img_hash = imagehash.phash(img)
                    if img_hash in image_hashes:
                        duplicates.append((img_path, image_hashes[img_hash]))
                    else:
                        image_hashes[img_hash] = img_path
                except Exception as e:
                    logging.error(f"Error processing {img_path}: {e}")

    if duplicates:
        logging.info(f"Found {len(duplicates)} duplicate images:")
        for dup in duplicates:
            logging.info(f"Duplicate pair: {dup[0]} and {dup[1]}")
    else:
        logging.info("No duplicates found.")

# Function to check class balance
def check_class_balance(directory):
    logging.info("Analyzing class distribution...")
    class_counts = Counter()

    for root, _, files in os.walk(directory):
        class_name = os.path.basename(root)
        image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] += len(image_files)

    logging.info("Class distribution:")
    for class_name, count in class_counts.items():
        logging.info(f"{class_name}: {count} images")

    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution")
    plt.grid(True)
    plt.show()

def visualize_triplet_samples(dataset, num_samples=5):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    for i, idx in enumerate(indices):
        anchor_img, positive_img, negative_img = dataset[idx]
    
        imgs = [anchor_img, positive_img, negative_img]
        for j, img in enumerate(imgs):
            img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Denormalize
            img = torch.clamp(img, 0, 1)  # Clamp the values to be in the range [0, 1]
            
            axs[i, j].imshow(img)
            axs[i, j].axis('off')

        axs[i, 0].set_title('Anchor')
        axs[i, 1].set_title('Positive')
        axs[i, 2].set_title('Negative')

    plt.tight_layout()
    plt.show()


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
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])                    

def precompute_embeddings(dataset, model, device='cuda'):
    model.eval()  
    embeddings = {}
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            img = dataset[idx][0].to(device)
            
            embedding = model(img.unsqueeze(0)).detach().cpu()  #  (1, embedding_dim)
            embeddings[idx] = embedding.squeeze(0)  # (embedding_dim)
    
    return embeddings

# Triplet Dataset class
class TripletDataset(Dataset):
    def __init__(self, dataset, model, device, precomputed_embeddings, indices_file):
        self.dataset = dataset
        self.class_indices = {}
        self.valid_indices = []
        self.model = model
        self.device = device
        self.precomputed_embeddings = precomputed_embeddings
        self.indices_file = indices_file
        
        for idx, label in enumerate(dataset.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        self.class_indices = {k: v for k, v in self.class_indices.items() if len(v) >= 2}

        self.valid_indices = [idx for indices in self.class_indices.values() for idx in indices]

        if len(self.valid_indices) == 0:
            raise ValueError("No valid classes with at least 2 samples each were found in the dataset.")

        # Load or create triplet indices
        if os.path.exists(self.indices_file):
            with open(self.indices_file, 'r') as file:
                self.triplet_indices = json.load(file)
            logging.info(f"Loaded triplet indices from {self.indices_file}")
        else:
            logging.info(f"No saved indices found. Mining hard negatives...")
            self.triplet_indices = self._mine_triplets(precomputed_embeddings)
            with open(self.indices_file, 'w') as file:
                json.dump(self.triplet_indices, file)
            logging.info(f"Saved triplet indices to {self.indices_file}")

    def _mine_triplets(self, precomputed_embeddings):
        triplet_indices = []
        distances_info = []  

        with torch.no_grad():
            for anchor_idx in tqdm(self.valid_indices, desc="Hard Negative Mining"):
                anchor_label = self.dataset.labels[anchor_idx]

                positive_idx = random.choice([i for i in self.class_indices[anchor_label] if i != anchor_idx])

                anchor_embedding = precomputed_embeddings[anchor_idx]
                positive_embedding = precomputed_embeddings[positive_idx]

                negative_distances = []
                for label, indices in self.class_indices.items():
                    if label != anchor_label:
                        for neg_idx in indices:
                            negative_embedding = precomputed_embeddings[neg_idx]
                            distance = nn.functional.pairwise_distance(anchor_embedding.unsqueeze(0), negative_embedding.unsqueeze(0))
                            negative_distances.append((distance.item(), neg_idx))

                # Find the hardest negative
                hardest_distance, hardest_negative_idx = min(negative_distances, key=lambda x: x[0])

                # Save the triplet (anchor, positive, hardest negative)
                triplet_indices.append({
                    'anchor_idx': anchor_idx,
                    'positive_idx': positive_idx,
                    'negative_idx': hardest_negative_idx
                })

                # Save the distance for the hardest negative
                distances_info.append({
                    'anchor_idx': anchor_idx,
                    'negative_idx': hardest_negative_idx,
                    'distance': hardest_distance
                })

        # Save the distances to a JSON file
        with open('hard_negative_distances.json', 'w') as distance_file:
            json.dump(distances_info, distance_file, indent=4)

        return triplet_indices
    
    def __len__(self):
        return len(self.triplet_indices)

    def __getitem__(self, idx):
        triplet = self.triplet_indices[idx]
        anchor_img = self.dataset[triplet['anchor_idx']][0]
        positive_img = self.dataset[triplet['positive_idx']][0]
        negative_img = self.dataset[triplet['negative_idx']][0]

        return anchor_img, positive_img, negative_img

# Loss function:
class TripletLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(TripletLoss, self).__init__()
        self.alpha = alpha

    def calc_distance(self, x1, x2):
        return nn.functional.pairwise_distance(x1, x2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_distance(anchor, positive)
        distance_negative = self.calc_distance(anchor, negative)
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    logging.info(f'Train Epoch: 	Loss: {avg_loss:.4f}')
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
    logging.info(f'Validation Epoch: 	Loss: {avg_loss:.4f}')
    return avg_loss

if __name__ == "__main__":
    dir_path = r'C:\Users\Miras\Desktop\DL\face\lfw-deepfunneled'
    min_files_count = 4
    dest_directory_path = r'C:\Users\Miras\Desktop\DL\face\lfw-deepfunneled_cropped_upd'

    # remove_small_folders(dir_path, min_files_count)

    # crop_and_save_faces(dir_path, dest_directory_path)

    # find_duplicates(dest_directory_path)
    # check_class_balance(dest_directory_path)

    # Create label mapping
    all_folders = sorted(os.listdir(dest_directory_path))
    label_mapping = {folder: idx for idx, folder in enumerate(all_folders)}

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
    criterion = nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Split into train and validation sets
    random.seed(42)
    random.shuffle(all_folders)
    split_index = int(0.75 * len(all_folders))
    train_folders = all_folders[:split_index]
    val_folders = all_folders[split_index:]
  
    # Create datasets
    train_dataset = FaceDataset(dest_directory_path, train_folders, label_mapping, transform=train_transform)
    val_dataset = FaceDataset(dest_directory_path, val_folders, label_mapping, transform=val_transform)
    precomputed_embeddings_train = precompute_embeddings(train_dataset, model, device=device)
    precomputed_embeddings_val = precompute_embeddings(val_dataset, model, device=device)

    # Initialize the Triplet Datasets and DataLoaders
    train_indices_file = 'triplet_indices_train.json'   # load the json formatted triplet indices    
    val_indices_file = 'triplet_indices_val.json'
    train_triplet_dataset = TripletDataset(train_dataset, model, device, precomputed_embeddings_train, train_indices_file)
    val_triplet_dataset = TripletDataset(val_dataset, model, device, precomputed_embeddings_val, val_indices_file)

    train_loader = DataLoader(dataset=train_triplet_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=os.cpu_count())
    val_loader = DataLoader(dataset=val_triplet_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=os.cpu_count())

    visualize_triplet_samples(train_triplet_dataset, num_samples=5)
    visualize_triplet_samples(val_triplet_dataset, num_samples=5)
    
    # Training loop
    num_epochs = 3
    train_losses, val_losses = [], []
    with open('model_weights.txt', 'w') as f:
        f.write(f'Before training: \n\n')
        for name, param in model.named_parameters():
            f.write(f'Weights of layer {name}:\n{param.data}\n\n')
    for epoch in range(num_epochs):
        logging.info('------------------------')
        logging.info(f'Epoch {epoch+1}/{num_epochs} training:')
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Train Loss (Epoch {epoch+1}): {train_loss}")
        print(f"Val Loss (Epoch {epoch+1}): {val_loss}")
        with open('model_weights.txt', 'w') as f:
            for name, param in model.named_parameters():
                f.write(f'Weights of layer {name}:\n{param.data}\n\n')
        
        precomputed_embeddings_train = precompute_embeddings(train_dataset, model, device)
        train_triplet_dataset.triplet_indices = train_triplet_dataset._mine_triplets(precomputed_embeddings_train)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

     # Plot the losses after training
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='b', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='r', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Save the model state dict
    torch.save(model.state_dict(), 'model_weights_3.pth')

    # Save the entire model
    torch.save(model, 'model_3.pth')

    # Try to script the model
    try:
        scripted_model = torch.jit.script(model)
        scripted_model.save('model_scripted_3.pt')
    except Exception as e:
        logging.error(f"Failed to script the model: {e}")