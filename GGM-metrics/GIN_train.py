import utils.experiment_logging as helpers
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import traceback
import logging
from evaluation.models.gin.gin import GIN as model
import pickle
import dgl

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Experiment args')

        self.parser.add_argument('--num_layers', default=5, type=int, help='The number of prop. rounds in GIN')
        self.parser.add_argument('--num_mlp_layers', default=2, type=int, help='The number of layers in the MLPs used in graph prop')
        self.parser.add_argument('--hidden_dim', default=64, type=int, help='The graph embed dimensionality at each round of graph prop. Final graph embed size is hidden_dim * (num_layers - 1)')

        self.parser.add_argument('--graph_pooling_type', default='sum', type=str, help='The method for aggregating node embeddings into graph embedding')
        self.parser.add_argument('--neighbor_pooling_type', default='sum', type=str, help='The method for aggregating neighborhood embeddings into a single node embedding')
        self.parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout (default: 0.5)')

        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
        self.parser.add_argument('--lr_step_size', default=3e5, type=int)
        self.parser.add_argument('--gamma', default=1, type=int)
        self.parser.add_argument('--epochs', default=200, type=int)

        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--res_dir', type=str, default='GIN_train')
        self.parser.add_argument('--no_cuda', action='store_true')

    def parse(self):
        args = self.parser.parse_args()
        args.input_dim = 1
        args.output_dim = 10
        args.learn_eps = False
        return args

class GINDataloader(Dataset):
    # Lobster: 0
    # Proteins: 1
    # Ego: 2
    # Community: 4
    # Grid: 5
    def __init__(self, device, set='train'):
        self.dataset = pickle.load(
            open('data/graphs/datasets/gin_train/{}.h5'.format(set), 'rb'))
        self.dataset = [(g.to(device), label.to(device))
                        for (g, label) in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ix):
        (g, label) = self.dataset[ix]
        feats = g.in_degrees() + g.out_degrees()
        feats = feats.unsqueeze(1).type(torch.float32)
        g.ndata['attr'] = feats.to(g.device)
        return (g, label)

    def collate_batch(self, batch):
        gs = []
        labels = []
        for (g, label) in batch:
            gs.append(g)
            labels.append(label)

        bg = dgl.batch(gs)
        labels = torch.tensor(labels).to(bg.device)
        return bg, labels

class Trainer():
    def __init__(self, helper):
        self.helper = helper
        args = self.helper.args
        self.printer = helpers.Printer(
            keys=['train_loss', 'valid_loss', 'valid_acc'], epochs=args.epochs)
        self.model = model(**args).to(args.device)
        self.opt = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.opt, step_size=args.lr_step_size, gamma=args.gamma)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader, self.valid_loader = self.get_dataloaders(args)

    def get_dataloaders(self, args):
        train_set = GINDataloader(args.device)
        valid_set = GINDataloader(args.device, set='valid')

        print('Train size:', len(train_set))
        print('Valid size:', len(valid_set))

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=train_set.collate_batch)#, pin_memory=True)
        valid_loader = DataLoader(
            valid_set, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=train_set.collate_batch)#, pin_memory=True)
        return train_loader, valid_loader

    def train(self):
        args = self.helper.args
        for epoch in range(args.epochs):
            train_loss = self.train_step(self.train_loader)
            valid_loss, valid_acc = self.eval_model(self.valid_loader)
            _, train_acc = self.eval_model(self.train_loader)
            res = {'train_loss': train_loss, 'train_acc': train_acc.item(),
                   'valid_loss': valid_loss, 'valid_acc': valid_acc.item()}
            self.helper.save_results(res)
            if epoch % 20 == 0:
                self.helper.checkpoint_model(self.model, self.opt, self.scheduler)
            self.helper.checkpoint_model_if_best(
                res, self.model, self.opt, self.scheduler,
                criteria='valid_loss', objective='minimize')
            self.printer.print(epoch, res)
            self.scheduler.step()

    def train_step(self, dataloader):
        self.model.train()
        running_loss = 0
        total_iters = len(dataloader)

        for (graphs, labels) in dataloader:
            feat = graphs.ndata['attr']
            outputs = self.model(graphs, feat)

            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

            # backprop
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        running_loss = running_loss / total_iters
        return running_loss

    def eval_model(self, dataloader):
        self.model.eval()
        running_loss = 0
        running_correct = 0
        total_iters = len(dataloader)
        count = 0

        for (graphs, labels) in dataloader:
            count += len(labels)
            feat = graphs.ndata['attr']

            outputs = self.model(graphs, feat)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum()

        loss = running_loss / total_iters
        acc = running_correct / count
        return loss, acc

if __name__ == '__main__':
    config = Config()
    config = config.parse()

    helper = helpers.ExperimentHelper(config, results_dir=config.res_dir) # For logging results mostly
    torch.manual_seed(helper.args.seed)
    helper.args.device = torch.device('cpu') if helper.args.no_cuda else torch.device('cuda')
    trainer = Trainer(helper)

    try:
        trainer.train()
    except:
        traceback.print_exc()
        logging.exception('')
    finally:
        helper.end_experiment()
