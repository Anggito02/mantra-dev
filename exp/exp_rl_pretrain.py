import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from models.ddpg import Actor

class Exp_RL_Pretrain():
    def __init__(self, args, device, obs_dim, act_dim, hidden_dim, states, train_error, cls_weights, valid_states, valid_error):        
        self.args = args
        self.device = device

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.states = states
        self.train_error = train_error
        self.cls_weights = cls_weights
        self.valid_states = valid_states
        self.valid_error = valid_error

    def forward(self):
        best_train_model = torch.LongTensor(self.train_error.argmin(1)).to(self.device)
        best_valid_model = torch.LongTensor(self.valid_error.argmin(1)).to(self.device)

        actor = Actor(self.obs_dim, self.act_dim, self.hidden_dim).to(self.device)
        best_actor = Actor(self.obs_dim, self.act_dim, self.hidden_dim).to(self.device)
        cls_weights = torch.FloatTensor([1/self.cls_weights[w] for w in range(self.act_dim)]).to(self.device)

        L = len(self.states)
        batch_size = 512
        batch_num  = int(np.ceil(L / batch_size))
        optimizer  = torch.optim.Adam(actor.parameters(), lr=3e-4)
        loss_fn    = nn.CrossEntropyLoss(weight=cls_weights)  # weighted CE loss
        best_acc   = 0
        patience   = 0
        max_patience = self.args.RL_max_patience
        for epoch in trange(self.args.RL_pretrain_epoch, desc='[Pretrain]'):
            epoch_loss = []
            shuffle_idx = np.random.permutation(np.arange(L))
            for i in trange(batch_num, desc='[Pretrain Step]'):
                batch_idx = shuffle_idx[i*batch_size: (i+1)*batch_size]
                optimizer.zero_grad()
                batch_out = actor(self.states[batch_idx])
                loss = loss_fn(batch_out, best_train_model[batch_idx])
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            with torch.no_grad():
                pred = actor(self.valid_states)
                pred_idx = pred.argmax(1)
                acc = (pred_idx == best_valid_model).sum() / len(pred)
            print(f'# epoch {epoch+1}: loss = {np.average(epoch_loss):.5f}\tacc = {acc:.3f}')

            # early stop w.r.t. validation acc
            if acc > best_acc:
                best_acc = acc
                patience = 0
                # update best model
                for param, target_param in zip(
                        actor.parameters(), best_actor.parameters()):
                    target_param.data.copy_(param.data)
            else:
                patience += 1
            
            if patience == max_patience:
                break

        with torch.no_grad():
            pred = best_actor(self.valid_states)
            pred_idx = pred.argmax(1)
            acc = (pred_idx == best_valid_model).sum() / len(pred)    
        print(f'valid acc for pretrained actor: {acc:.3f}') 
        return best_actor