# -*- coding: utf-8 -*-

import numpy as np
from client import Client
from server import Server
from torchvision import datasets, transforms
import torch
from model import  get_model, VGG9
from utils import (
    plot, 
    create_long_tail_split_noniid, 
    create_dirichlet_split_noniid,
    load_config
)
from torch.utils.data import DataLoader, TensorDataset 
import json
import os

import wandb
import textwrap

def main():
    
    config = load_config()
    
    lr_list = [0.001]
    
    for lr in lr_list:
        #save_path = 'log_data/config.json'
        #with open(save_path, 'w', encoding="utf-8") as f:
        #    json.dump(config, f, ensure_ascii=False, indent=4)
        config['optimizer']['learning_rate'] = lr
        name_gini = f"""mnist_lr{config['optimizer']['learning_rate']}
                    _long_tail_alpha_{config["dataset"]['alpha']}_random_resample_dynamic_random_clients_clients{config["dataset"]['alpha']}_randseed{config['system']['seed']}"""
        name_loss = f"""mnist_lr{config['optimizer']['learning_rate']}
                    _long_tail_alpha_{config["dataset"]['alpha']}_random_resample_random_clients_clients{config["dataset"]['alpha']}_randseed{config['system']['seed']}"""
        
        with wandb.init(
            project = "test-fl",
            entity= "whitewall_9-jinan-university",
            name= name_gini,
            config=config,
            notes=f"to observe the gini value",
            # tags= [''],
        ) as run:
            print(run.config)
            np.random.seed(run.config["system"]["seed"])
            print(f'current config is {run.config}')

            # cifar10_transforms = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # ])
            # cifar10_train_dataset = datasets.cifar10(root='./data', train=True, download=True, transforms=transforms)
            # cifar10_test_dataset = datasets.cifar10(root='./data', train=False, download=True, transforms=transforms)

            
            # ciar10_transform_train = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),   # 数据增强：随机裁剪
            #     transforms.RandomHorizontalFlip(),      # 随机水平翻转
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), 
            #                         (0.2023, 0.1994, 0.2010)),
            # ])

            # cifar10_transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), 
            #                         (0.2023, 0.1994, 0.2010)),
            # ])
            minst_tranform = transforms.Compose([
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            if config['dataset']['name'] == 'mnist':
                train_dataset =  datasets.MNIST('./mnist_dataset', train=True, download=True)
                test_dataset = datasets.MNIST('./mnist_dataset', train=False, download=True)
                print(f'Dataset: minst')
            elif config['dataset']['name'] == 'fashion_mnist':
                train_dataset = datasets.FashionMNIST('./fashion_minist_dataset', train=True, download=True)
                test_dataset = datasets.FashionMNIST('./fashion_minist_dataset', train=False, download=True)
                print(f'Dataset: fashion_mnist')
            elif config['dataset']['name'] == 'cifar10':
                train_dataset = datasets.CIFAR10(root='./data', train=True,
                                        download=True)
                test_dataset = datasets.CIFAR10(root='./data', train=False,
                                       download=True)

            

            train_data = train_dataset.data.to(torch.float)
            train_labels = train_dataset.targets.numpy()

            test_data = test_dataset.data.to(torch.float)
            test_labels = test_dataset.targets.numpy()
            
            # mean = (train_data.mean()) / (train_data.max() - train_data.min())
            # std = (train_data.std() / (train_data.max() - train_data.min()))
            
            
            train_data = minst_tranform(train_data)
            test_data = minst_tranform(test_data)
            train_data_size = train_data.shape[0]
            test_data_size = test_data.shape[0]
            test_data_loader = DataLoader(TensorDataset(test_data, 
                                                        torch.as_tensor(test_labels)), 
                                                        batch_size = run.config['train']['batch_size'])

            
            clients_train_data, clients_train_label = create_long_tail_split_noniid(train_data=train_data,
                                                                                train_labels=train_labels,
                                                                                alpha=run.config['dataset']['alpha'],
                                                                                clients_number=run.config['train']['num_clients'],
                                                                                seed=run.config['system']['seed'],
                                                                                config=run.config)
            # clients_train_data, clients_train_label = create_dirichlet_split_noniid(
            #     train_data=train_data,
            #     train_labels=train_labels,
            #     alpha=config['dataset']['alpha'],
            #     clients_number=config['train']['num_clients'],
            #     seed=config['system']['seed']  # dataset productivity
            # )
            
            device = run.config['system']['device']
            clients = []
            model = get_model()
            # model = VGG9(num_classes=10)
            for i in range(10):
                clients.append(Client(i, clients_train_data[i], clients_train_label[i]))
            server = Server(rounds=run.config['train']['global_rounds'], 
                            clients=clients, 
                            test_dataloader=test_data_loader,  
                            global_model=model.to(device=device),
                            config=run.config,
                            client_ratio=run.config['train']['client_ratio'])
                            
            accuracy_history = server.train(beta=run.config['strategy']['iwds']['beta'],
                                            beta_zero=run.config['strategy']['iwds']['beta_zero'],
                                            rou=run.config['strategy']['iwds']['rou'],
                                            log_dir=run.config['system']['log_dir'],
                                            local_epoch=run.config['train']['local_epoch'],
                                            local_batch_size=run.config['train']['batch_size'],
                                            learning_rate=run.config['optimizer']['learning_rate'],
                                            limit_rounds=run.config['strategy']['client_gini']['limit_rounds'])
            
            accuracy_history = [acc.cpu().item() 
                                if isinstance(acc, torch.Tensor) else acc 
                                for acc in accuracy_history]
        
if __name__ == '__main__':
    main()
