import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import random
import wandb
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
# set_seed()
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
wandb.init(
            project = "test-fl",
            entity= "whitewall_9-jinan-university",
            name= 'pll_central_0.9',
)
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def generate_partial_labels(true_labels, num_classes=10, num_candidates=10):
    partial_labels = []
    for y in true_labels:
        candidates = {y}
        while len(candidates) < num_candidates:
            r = random.randint(0, num_classes - 1)
            candidates.add(r)
        partial_labels.append(sorted(list(candidates)))
    print(partial_labels)
    return partial_labels
def generate_partial_labels_prob(true_labels, num_classes=10, rho=0.3):
    partial_labels = []
    print(f'partital label rho: {rho}')
    for true_label in true_labels:
        other_labels = list(range(num_classes))
        other_labels.remove(true_label)

        candidate_labels = [true_label]
        for label in other_labels:
            if random.random() < rho:
                candidate_labels.append(label)

        if len(candidate_labels) == 1:
            random_irrelevant = random.choice(other_labels)
            candidate_labels.append(random_irrelevant)

        candidate_labels.sort()
        partial_labels.append(candidate_labels)
    # print(partial_labels)
    return partial_labels
class MNIST_PartialLabel(torch.utils.data.Dataset):
    def __init__(self, dataset, rho=0.0, num_classes=10, num_candidates=10):
        self.dataset = dataset
        self.num_classes = num_classes
        # self.partial_labels = generate_partial_labels([label for _, label in dataset], num_candidates=num_candidates)
        self.partial_labels = generate_partial_labels_prob([label for _, label in dataset], rho=rho)

    def __getitem__(self, index):
        img, true_label = self.dataset[index]
        partial_label = self.partial_labels[index]
        mask = torch.zeros(self.num_classes)
        mask[torch.tensor(partial_label, dtype=torch.long)] = 1.0

        return img, mask, true_label

    def __len__(self):
        return len(self.dataset)

partial_train_dataset = MNIST_PartialLabel(train_dataset, rho=0.9, num_candidates=5)
train_loader = torch.utils.data.DataLoader(partial_train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

import matplotlib.pyplot as plt

label_counts = torch.zeros([10])
for i in range(len(partial_train_dataset)):
        _, mask, true_label = partial_train_dataset[i]
        for j in range(10):
                    if mask[j] == 1:
                        label_counts[j] += 1
            
counts_np = label_counts.numpy()
classes = np.arange(10)

plt.figure(figsize=(10, 6))
bars = plt.bar(classes, counts_np, color='skyblue', edgecolor='black')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
        f'{int(height)}', ha='center', va='bottom')

plt.title('Class Distribution', fontsize=14)
plt.xlabel('Class Label', fontsize=12)
plt.ylabel('Sample Count', fontsize=12)
plt.xticks(classes)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
wandb.run.log({
                "dataset": wandb.Image(plt.gcf())
             })

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 13 * 13, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
def pll_loss(logits, candidate_labels):
    batch_size = logits.size(0)
    losses = []
    for i in range(batch_size):
        logits_i = logits[i]
        candidates = candidate_labels[i]
        log_probs = torch.log_softmax(logits_i, dim=0)
        loss_i = -log_probs[candidates].mean()
        losses.append(loss_i)
    return torch.stack(losses).mean()
def partial_label_loss(logits, mask):
        """
        logits: [B, C]
        mask: [B, C] (0/1 vector indicating candidate labels)
        """
        # Normalize mask to sum=1 over candidates (avoid bias)
        normalized_mask = mask / mask.sum(dim=1, keepdim=True)

        # Cross-entropy with soft labels
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(normalized_mask * log_probs).sum(dim=1).mean()
        return loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, true_labels in loader:
            images = images.to(device)
            true_labels = true_labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == true_labels).sum().item()
            total += true_labels.size(0)
    print(f"Test Accuracy: {correct / total * 100:.2f}%")
    return correct / total

model = SimpleCNN().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

auc_list = []
avg_loss = 0.0
for epoch in range(100):
    model.train()
    total_loss = 0
    total_batches = 0
    for images, masks, true_labels in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        # candidate_labels = [cl.to(device) for cl in candidate_labels]
        logits = model(images)

        # loss = pll_loss(logits, candidate_labels)
        loss = partial_label_loss(logits=logits, mask=masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
    accuracy = evaluate(model, test_loader)
    wandb.run.log({
                "accuracy": accuracy,
                "loss": total_loss / total_batches
            })
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
