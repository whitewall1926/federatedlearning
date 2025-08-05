import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import wandb


def generate_partial_labels(true_labels, num_classes=10, num_candidates=10):
    partial_labels = []
    for y in true_labels:
        candidates = {y}
        while len(candidates) < num_candidates:
            r = random.randint(0, num_classes - 1)
            candidates.add(r)
        partial_labels.append(sorted(list(candidates)))
    # print(partial_labels)
    return partial_labels

def generate_partial_labels_prob(true_labels, num_classes=10, rho=0.0):
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

class Client:
    def __init__(self, 
                 client_id, 
                 datas, 
                 labels):
        self.client_id = client_id
        self.train_dataset = TensorDataset(datas, torch.as_tensor(labels))
        self.partial_train_dataset = None
        self.train_dataloader = None
        self.local_model = None
        self.loss = 10
        
    def get_loss(self):
        return self.loss

    
    def partial_label_loss(self,logits, mask):
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

    def pll_loss(self, logits, candidate_labels):
        batch_size = logits.size(0)
        losses = []
        for i in range(batch_size):
            logits_i = logits[i]
            candidates = candidate_labels[i]
            log_probs = torch.log_softmax(logits_i, dim=0)
            loss_i = -log_probs[candidates].mean()
            losses.append(loss_i)
        return torch.stack(losses).mean()
    
    def cal_gini(self):
        labels = self.train_dataset.tensors[1].numpy().tolist()
        labels_num = [ labels.count(c) for c in range(10)]
        n = len(labels_num)
        labels_num.sort()
        
        sum_labels = sum(labels_num)
        denomirator = sum(labels_num) * 2 * n
        nomirator = 0
        pre = 0
        for i, num in enumerate(labels_num):
            nomirator += num * i - pre + (sum_labels - pre - num) - (n - i - 1) * num
            pre += num
        
        return nomirator / denomirator

        

    def local_train(self, 
                    net, 
                    global_paramers, 
                    lr=0.001,
                    local_epochs = 1, 
                    local_batch_size = 32,
                    beta = 0.99,
                    rho=0.0,
                    device='cuda',
                    iwds_enabled = True):
        net = net.to(device)
        net.load_state_dict(global_paramers)

        
        label_count = {}
        for _, label in self.train_dataset:
            if label.item() not in label_count:
                label_count[label.item()] = 1
            else:
                label_count[label.item()] += 1
        
        
        wegiths = []
        for _, label in self.train_dataset:
            wegiths.append((1 - beta) / (1 - beta**(label_count[label.item()])))
        sum_weights = torch.tensor(wegiths).sum()
        probability = []
        for  _, label in self.train_dataset:
            probability.append(wegiths[len(probability)] / sum_weights)
        sampler = WeightedRandomSampler(
            weights=probability,
            num_samples=len(probability),
            replacement= True
        )
        if self.partial_train_dataset == None:
            self.partial_train_dataset = MNIST_PartialLabel(self.train_dataset, rho=rho, num_candidates=5)
            
            label_counts = torch.zeros([10])
            for i in range(len(self.partial_train_dataset)):
                _, mask, true_label = self.partial_train_dataset[i]
                for j in range(10):
                    if mask[j] == 1:
                        label_counts[j] += 1
            
            counts_np = label_counts.numpy()
            classes = np.arange(10)

            # 创建图表
            plt.figure(figsize=(10, 6))
            bars = plt.bar(classes, counts_np, color='skyblue', edgecolor='black')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')

            # 美化图表
            plt.title('Class Distribution', fontsize=14)
            plt.xlabel('Class Label', fontsize=12)
            plt.ylabel('Sample Count', fontsize=12)
            plt.xticks(classes)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            wandb.run.log({
                "dataset": wandb.Image(plt.gcf())
             })
            # plt.show()



        if iwds_enabled == True:
            print(f'client {self.client_id} uses the method of iwds to sample')
            self.train_data_loader = DataLoader(self.partial_train_dataset, 
                                                batch_size = local_batch_size, 
                                                sampler=sampler)
        else:
            print(f'client {self.client_id} uses the method of random_sample')
            self.train_data_loader = DataLoader(self.partial_train_dataset, 
                                                batch_size = local_batch_size, 
                                                shuffle=True)
        loss_func = torch.nn.CrossEntropyLoss()
        # opti = torch.optim.Adam(net.parameters(), lr=0.001)
        opti = torch.optim.SGD(net.parameters(), lr=lr)
        avg_loss = 0.0
        for epoch in range(local_epochs):
            running_loss = 0.0
            total_batches = 0
            for data, mask, label in self.train_data_loader:
                data, label = data.to(device), label.to(device)
                mask = mask.to(device)
                opti.zero_grad()
                output = net(data)
                # loss = loss_func(output, label)

                # loss = self.pll_loss(output, candidate_labels)
                loss = self.partial_label_loss(output, mask)
                loss.backward()
                opti.step()

                running_loss += loss.item()
                total_batches += 1
            avg_loss = running_loss / total_batches
            
        self.loss = avg_loss 
        self.local_model = net
        return net.state_dict()

