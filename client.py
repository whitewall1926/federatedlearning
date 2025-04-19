import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
class Client:
    def __init__(self, 
                 client_id, 
                 datas, 
                 labels):
        self.client_id = client_id
        self.train_dataset = TensorDataset(datas, torch.as_tensor(labels))
        self.train_dataloader = None
        self.local_model = None
        
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
        if iwds_enabled == True:
            print(f'client {self.client_id} uses the method of iwds to sample')
            self.train_data_loader = DataLoader(self.train_dataset, 
                                                batch_size = local_batch_size, 
                                                sampler=sampler)
        else:
            print(f'client {self.client_id} uses the method of random_sample')
            self.train_data_loader = DataLoader(self.train_dataset, 
                                                batch_size = local_batch_size, 
                                                shuffle=True)
        loss_func = torch.nn.CrossEntropyLoss()
        # opti = torch.optim.Adam(net.parameters(), lr=0.001)
        opti = torch.optim.SGD(net.parameters(), lr=lr)
        for epoch in range(local_epochs):
            for data, label in self.train_data_loader:
                data, label = data.to(device), label.to(device)
                opti.zero_grad()
                output = net(data)
                loss = loss_func(output, label)
                loss.backward()
                opti.step()
                
        self.local_model = net
        return net.state_dict()

