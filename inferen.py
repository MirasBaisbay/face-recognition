import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load('model_scripted_3.pt')
model = model.to(device)
model.eval()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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

class FaceVerificationDataset(Dataset):
    def __init__(self, dataset, num_pairs=1000):
        self.dataset = dataset
        self.num_pairs = num_pairs
        self.pairs = []
        self.labels = []
        self._generate_pairs()
        
    def _generate_pairs(self):
        class_indices = {}
        for idx, label in enumerate(self.dataset.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Generate positive pairs
        positive_pairs = 0
        while positive_pairs < self.num_pairs // 2:
            label = random.choice(list(class_indices.keys()))
            if len(class_indices[label]) >= 2:
                idx1, idx2 = random.sample(class_indices[label], 2)
                self.pairs.append((idx1, idx2))
                self.labels.append(1)  # Same class
                positive_pairs += 1
        
        # Generate negative pairs
        labels_list = list(class_indices.keys())
        negative_pairs = 0
        while negative_pairs < self.num_pairs // 2:
            label1, label2 = random.sample(labels_list, 2)
            idx1 = random.choice(class_indices[label1])
            idx2 = random.choice(class_indices[label2])
            self.pairs.append((idx1, idx2))
            self.labels.append(0)  # Different classes
            negative_pairs += 1
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        img1, _ = self.dataset[idx1]
        img2, _ = self.dataset[idx2]
        label = self.labels[idx]
        return img1, img2, label

def evaluate_model(model, dataset, device):
    verification_dataset = FaceVerificationDataset(dataset)
    verification_loader = DataLoader(verification_dataset, batch_size=32, shuffle=False)
    
    embeddings1 = []
    embeddings2 = []
    labels = []
    model.eval()
    with torch.no_grad():
        for img1, img2, label in tqdm(verification_loader, desc='Evaluating'):
            img1 = img1.to(device)
            img2 = img2.to(device)
            emb1 = model(img1)
            emb2 = model(img2)
            embeddings1.append(emb1.cpu().numpy())
            embeddings2.append(emb2.cpu().numpy())
            labels.extend(label.numpy())
    
    embeddings1 = np.vstack(embeddings1)
    embeddings2 = np.vstack(embeddings2)
    labels = np.array(labels)
    
    distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)
    
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    roc_auc = auc(fpr, tpr)
    
    accuracies = []
    for threshold in thresholds:
        preds = distances < -threshold
        acc = accuracy_score(labels, preds)
        accuracies.append(acc)
    best_threshold = thresholds[np.argmax(accuracies)]
    best_accuracy = np.max(accuracies)
    
    print(f'Best Threshold: {-best_threshold:.4f}')
    print(f'Best Accuracy: {best_accuracy*100:.2f}%')
    print(f'AUC: {roc_auc:.4f}')
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Face Verification')
    plt.legend(loc='lower right')
    plt.show()
    
    return best_threshold, best_accuracy, roc_auc

def verify_faces(model, img_path1, img_path2, threshold, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')
    
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        emb1 = model(img1)
        emb2 = model(img2)
    
    distance = torch.norm(emb1 - emb2, p=2).item()
    
    if distance < -threshold:
        print(f'The faces match (distance: {distance:.4f} < threshold: {-threshold:.4f})')
    else:
        print(f'The faces do not match (distance: {distance:.4f} >= threshold: {-threshold:.4f})')


if __name__ == "__main__":
    dest_directory_path = r'C:\Users\Meiras\Desktop\DL\face\lfw-deepfunneled_cropped_upd'  # Update this path
    random.seed(42)
    
    all_folders = sorted(os.listdir(dest_directory_path))
    label_mapping = {folder: idx for idx, folder in enumerate(all_folders)}
    
    random.shuffle(all_folders)
    split_index = int(0.75 * len(all_folders))
    train_folders = all_folders[:split_index]
    val_folders = all_folders[split_index:]
    
    val_dataset = FaceDataset(dest_directory_path, val_folders, label_mapping, transform=val_transform)
    
    best_threshold, best_accuracy, roc_auc = evaluate_model(model, val_dataset, device)
    
    img_path1 = os.path.join(dest_directory_path, val_folders[0], os.listdir(os.path.join(dest_directory_path, val_folders[0]))[0])
    img_path2 = os.path.join(dest_directory_path, val_folders[1], os.listdir(os.path.join(dest_directory_path, val_folders[1]))[0])
    
    verify_faces(model, img_path1, img_path2, best_threshold, device)