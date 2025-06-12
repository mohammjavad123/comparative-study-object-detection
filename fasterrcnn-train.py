import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# CLASS MAP (VOC)
# ======================
CLASS_MAP = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
    "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
    "fire hydrant": 11, "stop sign": 12, "parking meter": 13, "bench": 14,
    "bird": 15, "cat": 16, "dog": 17, "horse": 18, "sheep": 19, "cow": 20,
}

# ======================
# Custom VOC Dataset
# ======================
class VOCDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.files = [f for f in os.listdir(root) if f.endswith(".xml")]
        if len(self.files) == 0:
            raise RuntimeError(f"No annotation files found in: {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        xml_file = self.files[idx]
        img_file = xml_file.replace(".xml", ".jpg")
        img_path = os.path.join(self.root, img_file)
        xml_path = os.path.join(self.root, xml_file)

        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in CLASS_MAP:
                continue
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_MAP[label])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

# ======================
# Transform & Dataset
# ======================
transform = T.Compose([T.ToTensor()])
train_path = "/home/mrajabi/vision/fasterrcnn/Pascal-VOC-2012-13/train"
output_dir = "/home/mrajabi/vision/fasterrcnn/output"
os.makedirs(output_dir, exist_ok=True)

dataset = VOCDataset(train_path, transforms=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# ======================
# Model & Optimizer
# ======================
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# ======================
# Training Function
# ======================
def train(model, dataloader, device, epochs=25):
    model.train()
    for epoch in range(epochs):
        print(f"\n🟢 Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        for images, targets in tqdm(dataloader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip samples with no boxes
            if any(t["boxes"].numel() == 0 for t in targets):
                continue

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        print(f"🔻 Epoch {epoch+1} - Total Loss: {running_loss:.4f}")

    # Save model after training
    save_path = os.path.join(output_dir, "fasterrcnn_pascalvoc13.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to: {save_path}")

# ======================
# Run Training
# ======================
if __name__ == "__main__":
    train(model, dataloader, device)
            