
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from tqdm import tqdm

# Config
DATASET_DIR = "./Pascal-VOC-2012-13/train"
MODEL_SAVE_DIR = "./saved_models"
EPOCHS = 5
BATCH_SIZE = 2
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Label map
CLASS_MAP = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
    "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
    "fire hydrant": 11, "stop sign": 12, "parking meter": 13, "bench": 14,
    "bird": 15, "cat": 16, "dog": 17, "horse": 18, "sheep": 19, "cow": 20,
}

class VOCDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.files = [f for f in os.listdir(root) if f.endswith(".xml")]
        if not self.files:
            raise RuntimeError(f"No annotation files found in: {root}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        xml_file = self.files[idx]
        img_file = xml_file.replace(".xml", ".jpg")
        img_path = os.path.join(self.root, img_file)
        xml_path = os.path.join(self.root, xml_file)

        image = Image.open(img_path).convert("RGB")
        boxes, labels = [], []

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin, ymin = float(bbox.find("xmin").text), float(bbox.find("ymin").text)
            xmax, ymax = float(bbox.find("xmax").text), float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_MAP.get(label, 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

transform = T.Compose([T.ToTensor()])
full_dataset = VOCDataset(DATASET_DIR, transforms=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

def train(model, loader, device, epochs):
    model.train()
    for epoch in range(epochs):
        print(f"\n🟢 Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        for images, targets in tqdm(loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if any(t["boxes"].numel() == 0 for t in targets):
                continue

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"🔻 Avg Loss: {avg_loss:.4f}")

        model_path = os.path.join(MODEL_SAVE_DIR, f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved at {model_path}")

if __name__ == "__main__":
    train(model, train_loader, device, EPOCHS)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "model_final.pth"))
    print("✅ Final model saved.")
