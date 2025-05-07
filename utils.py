import json
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.distributions.dirichlet import Dirichlet
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pylab as plt
from torch.utils.tensorboard import SummaryWriter
import wandb

def load_config(config_path=r'./config.json'):
    """
    加载JSON格式的配置文件
    
    Args:
        config_path: JSON配置文件路径
        
    Returns:
        包含配置参数的字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def plot(accuracy_history, save_path=None):
    """
    绘制准确率历史曲线
    
    Args:
        accuracy_history: 准确率历史数据列表
        save_path: 图像保存路径，如果为None则显示图像
    """
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_history, 'b-', linewidth=2)
    plt.title('模型准确率随训练轮数的变化')
    plt.xlabel('训练轮数')
    plt.ylabel('准确率')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def add_confusion_matrix(cm, 
                         title='Confusion matrix', 
                         cmap=plt.cm.Blues,
                         log_dir='log_data',
                         tag='unknown'):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    cm = np.array(cm) 
    tick_marks = np.arange(len(cm))
    classes = np.arange(len(cm))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('class')
    plt.xlabel('client')
    plt.tight_layout()
    # writer = SummaryWriter(log_dir=log_dir)
    # writer.add_figure(tag=tag, figure=plt.gcf(), global_step=0)
    # writer.close()
    wandb.run.log({
        "dataset": wandb.Image(plt.gcf())
    })

def create_long_tail_split_noniid(train_data, 
                                  train_labels, 
                                  config,
                                  alpha=1, 
                                  clients_number=10, 
                                  seed=42,
                                  log_dir='log_data',
                                  ):
    """
    使用长尾分布划分数据集 (Non-IID)
    
    Args:
        train_data: 训练数据 (Tensor)
        train_labels: 训练标签 (Tensor 或 numpy 数组)
        alpha: 长尾分布参数
        clients_number: 客户端数量
        seed: 随机数种子，保证结果一致性
        
    Returns:
        clients_train_data: 每个客户端的数据
        clients_train_label: 每个客户端的标签
    """
    # 设置随机数种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'alpha: {alpha}')
    classes = np.max(train_labels) + 1
    clients_train_data = {}
    clients_train_label = {}
    batch_idxs = []
    
    for i in range(clients_number):
        batch_idxs.append([])
    for id in range(classes):
        all_idxs = []
        for j, label in enumerate(train_labels):
            if label == id:
                all_idxs.append(j)
        idxs = np.random.permutation(len(all_idxs))
        num = 0
        client_id = 0
        for j in range(len(idxs)):
            if client_id == id:
                if num + 1 <= int(alpha * len(idxs)):
                    batch_idxs[client_id].append(all_idxs[idxs[j]])
                    num += 1
                else:
                    client_id += 1
                    num = 0
            else:
                if (num + 1) * (clients_number - 1) <= len(idxs) - int(alpha * len(idxs)):
                    batch_idxs[client_id].append(all_idxs[idxs[j]])
                    num += 1
                else:
                    num = 0
                    client_id += 1
    conf_matrix = [[] for i in range(clients_number)]


    for i in range(clients_number):
        clients_train_data[i] = train_data[batch_idxs[i]]
        clients_train_label[i] = train_labels[batch_idxs[i]]
        distribution = [list(clients_train_label[i]).count(j) for j in range(classes)]

        label_counts = torch.bincount(torch.tensor(clients_train_label[i]), minlength=classes)
        conf_matrix[i].extend(label_counts.tolist())

        print(distribution)
    add_confusion_matrix(conf_matrix, 
                         log_dir=log_dir,
                         tag=f'{config["dataset"]["name"]}/long_tail_distribution{alpha}')
    return clients_train_data, clients_train_label


def create_dirichlet_split_noniid(train_data, 
                                  train_labels, 
                                  config,
                                  alpha=1, 
                                  clients_number=10, 
                                  seed=42,
                                  log_dir='log_data',
                                  ):
    """
    使用 Dirichlet 分布划分数据集 (Non-IID)
    
    Args:
        train_data: 训练数据 (Tensor)
        train_labels: 训练标签 (Tensor 或 numpy 数组)
        alpha: Dirichlet 分布的浓度参数 (越小越不均匀)
        clients_number: 客户端数量
        seed: 随机数种子，保证结果一致性
        
    Returns:
        clients_train_data: 每个客户端的数据
        clients_train_label: 每个客户端的标签
    """
    # 设置随机数种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    classes = np.max(train_labels) + 1
    label_distribution = Dirichlet(torch.tensor([alpha] * clients_number))
    class_indices = [np.where(train_labels == i)[0] for i in range(classes)]
    
    clients_train_data = {i: [] for i in range(clients_number)}
    clients_train_label = {i: [] for i in range(clients_number)}
    
    for c in range(classes):
        proportions = label_distribution.sample().numpy()
        proportions = (proportions / proportions.sum()) * len(class_indices[c])
        proportions = proportions.astype(int)

        # 确保每个客户端至少分配一个样本
        for i in range(len(proportions)):
            if proportions[i] == 0:
                proportions[i] = 1
        # 调整总数以匹配类别样本总数
        while sum(proportions) > len(class_indices[c]):
            proportions[np.argmax(proportions)] -= 1
        while sum(proportions) < len(class_indices[c]):
            proportions[np.argmin(proportions)] += 1

        split_indices = np.split(class_indices[c], np.cumsum(proportions)[:-1])
        
        for client_id, indices in enumerate(split_indices):
            clients_train_data[client_id].extend(train_data[indices])
            clients_train_label[client_id].extend(train_labels[indices])
    
    for client_id in range(clients_number):
        if len(clients_train_data[client_id]) > 0:
            clients_train_data[client_id] = torch.stack(clients_train_data[client_id])
            clients_train_label[client_id] = torch.tensor(clients_train_label[client_id])
        else:
            print(f"警告: 客户端 {client_id} 没有分配到数据！")
    
    # 打印每个客户端分配到的各个类标签的数量
    conf_matrix = [[] for _ in range(clients_number)]

    for client_id in range(clients_number):
        label_counts = torch.bincount(clients_train_label[client_id], minlength=classes)
        conf_matrix[client_id].extend(label_counts.tolist())
        print(f"客户端 {client_id} 的标签分布: {label_counts.tolist()}")

    add_confusion_matrix(conf_matrix, 
                         log_dir=log_dir,
                         tag=f'{config["dataset"]["name"]}/dirichlet{alpha}')
    
    return clients_train_data, clients_train_label
