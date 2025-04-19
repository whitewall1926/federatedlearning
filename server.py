# -*- coding: utf-8 -*-

from numpy import copy
import torch
from torch.utils.data import DataLoader
from torch import nn
import copy
from torch.utils.tensorboard import SummaryWriter
import wandb
class Server:
    def __init__(self, 
                 rounds, 
                 clients, 
                 test_dataloader, 
                 global_model,
                 client_ratio=0.5,
                 config = None):
        
        self.rounds =  rounds
        self.clients = clients
        self.global_model = global_model
        self.test_dataloader = test_dataloader
        self.client_ratio = client_ratio
        self.w = {}
        if config is not None:
            self.config = config
    
    def select_clients_dynamic(self, roud, limit_rounds=0):

        selected_clients = int(self.client_ratio * len(self.clients))
        
        if self.config['strategy']['client_gini']['enabled'] == False:
            print(f'server side: clients: random selection')
            rand_indxs = torch.randperm(len(self.clients)).tolist()
            print(rand_indxs[:selected_clients])
            return rand_indxs[:selected_clients]
        
        print(f'server side: clinets: gini decay to  random selection')
        gini_clients = []
        for client_id in range(10):
            gini_clients.append((self.clients[client_id].cal_gini(), client_id))

        # weight = [1.0 - gini[0] for gini in gini_clients]
        weight = [gini[0] for gini in gini_clients]
        sum_weight = sum(weight)
        N = len(gini_clients)
        prob = [w / sum_weight for w in weight]
        alpha = (1  - roud / self.config['train']['global_rounds'])
        prob_t = torch.tensor([(1 - alpha) * p + p  / N for p in prob])
        print(prob_t)
        rand_indxs = list(torch.multinomial(prob_t, selected_clients, replacement=False))
        print(rand_indxs)
       
        return rand_indxs
        
    def select_clients(self, roud, limit_rounds=0):

        selected_clients = int(self.client_ratio * len(self.clients))
        if limit_rounds == 0:
            print(f'server side: clients: random selection')
        else:
            print(f'server side: clinets: gini before {limit_rounds}, then random selection')
        if roud >= limit_rounds:
            rand_indxs = torch.randperm(len(self.clients)).tolist()
            print(rand_indxs[:selected_clients])
            return rand_indxs[:selected_clients]
        else:
            gini_clients = []
            for client_id in range(10):
                gini_clients.append((self.clients[client_id].cal_gini(), client_id))

            if self.config['strategy']['client_gini']['reverse'] == True:
                gini_clients.sort(reverse=True)
                print('using gini reverse')
            else:
                print('using gini')
                gini_clients.sort()
            
            rand_indxs = []
            for i in range(selected_clients):
                rand_indxs.append(gini_clients[i][1])
            # print(f'gini score:{gini_clients}')
            print(rand_indxs)
            return rand_indxs
        

        
        return rand_indxs[:selected_clients]


    def train(self, 
              beta=0.99, 
              beta_zero = 0.9999, 
              rou=0.992,
              log_dir = 'log_data',
              local_epoch=1,
              local_batch_size = 32,
              learning_rate=0.001,
              limit_rounds=0
              ):
        
        accuracy_history = []
        # writer = SummaryWriter(log_dir)

        accuracy, loss = self.eval_model(-1)
        accuracy_history.append(accuracy)
        for t in range(self.rounds):
            global_parameters = self.global_model.state_dict()
            beta_t = beta + (beta_zero - beta) * rou**t
            print(f'beta_t: {beta_t:.10f}')

            # rand_indxs = torch.randperm(len(self.clients)).tolist()
            # selected_clients = int(self.client_ratio * len(self.clients))
            # print(rand_indxs[:selected_clients])

            # rand_indxs = self.select_clients(t, limit_rounds=limit_rounds)
            rand_indxs = self.select_clients_dynamic(t, limit_rounds=limit_rounds)

            for i in rand_indxs:
                client = self.clients[i]
                

                client.local_train(self.global_model, 
                                   global_parameters,
                                   local_epochs = local_epoch,
                                   local_batch_size = local_batch_size,
                                   beta=beta_t,
                                   lr=learning_rate,
                                   iwds_enabled = self.config['strategy']['iwds']['enabled'])
                self.w[i] = client.local_model.state_dict()
            self.aggregate_model_parameters(rand_indxs)
            accuracy, loss = self.eval_model(t)
            wandb.run.log({
                "accuracy": accuracy,
                "loss": loss
            })
            # if self.config['dataset']['distribution'] == 'dirichlet':
            #     writer.add_scalar(tag=f'dirichlet_alpha{self.config["dataset"]["alpha"]}/train/accuracy', scalar_value=accuracy, global_step=t, walltime=15)
            # elif self.config['dataset']['distribution'] == 'long_tail':
            #     writer.add_scalar(tag=f'long_tail_alpha{self.config["dataset"]["alpha"]}/train/accuracy', scalar_value=accuracy, global_step=t, walltime=15)
            # else:
            #     raise ValueError('warning: the distribution is None')
                # print(f'warning: the distribution is None')
            accuracy_history.append(accuracy)
        # writer.close()
        return accuracy_history
    
    def aggregate_model_parameters(self, rand_indxs):
        w_avg = copy.deepcopy(self.w[rand_indxs[0]])
        dataset_sum = 0
        for i in rand_indxs:
            dataset_sum += len(self.clients[i].train_dataset)
        
        for k in w_avg.keys():
            # w_avg[k] *= len(self.clients[rand_indxs[0]].train_dataset)
            for i in range(1, len(rand_indxs)):
                w_avg[k] += self.w[rand_indxs[i]][k]
                # w_avg[k] += self.w[rand_indxs[i]][k] * len(self.clients[rand_indxs[i]].train_dataset)
            # w_avg[k] /= dataset_sum
            w_avg[k] /= len(rand_indxs)
        self.global_model.load_state_dict(w_avg, strict = True)

        return w_avg
    
    def eval_model(self, round):
        total_test_loss = 0
        self.global_model.eval()
        loss_func = nn.CrossEntropyLoss()
        with torch.no_grad():
            sum_acuu = 0
            num = 0
            for data, label in self.test_dataloader:
                data, label = data.to('cuda'), label.to('cuda')
                output = self.global_model(data)
                loss = loss_func(output, label)
                
                total_test_loss += loss.item()
                output = torch.argmax(output, dim=1)
                sum_acuu += (output == label).float().mean()
                num += 1
            accuracy = sum_acuu / num
            avg_loss = total_test_loss / num
        print(f'server side--- {round + 1} roud: accuracy: {accuracy}, loss: {avg_loss}')
        return accuracy, avg_loss