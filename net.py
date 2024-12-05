import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

n_feature = 21
n_output = 1
n_hidden1 = 30
n_hidden2 = 30
n_hidden3 = 60

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(n_feature, n_hidden1)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        # self.layer3 = nn.Linear(n_hidden2, n_hidden3)
        # self.output = nn.Linear(n_hidden3, n_output)
        self.output = nn.Linear(n_hidden2, n_output)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        # x = self.activation(self.layer3(x))
        x = self.output(x)
        return nn.Sigmoid()(x)


class Trainer():
    def __init__(self, learning_rate=1e-3, n_epoch=200):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.model = Net()
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # stats
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.best_model_accuracy = None
        self.best_model_loss = None
        self.best_epoch_accuracy = 0
        self.best_epoch_loss = 0
        self.max_accuracy = 0
        self.min_loss = np.inf
        self.patience = 10

    def train(self, x_train_df, y_train_df, x_test_df, y_test_df):
        x_train = torch.tensor(x_train_df.values, dtype=torch.float32)
        x_test = torch.tensor(x_test_df.values, dtype=torch.float32)
        y_train = torch.tensor(y_train_df.values.reshape(-1, 1), dtype=torch.float32)
        y_test = torch.tensor(y_test_df.values.reshape(-1, 1), dtype=torch.float32)

        epoch = 1
        while self.patience > 0:
        # for epoch in range(self.n_epoch):
            y_pred_train = self.model(x_train)
            train_loss = self.loss_fn(y_pred_train, y_train)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            # losses
            y_pred_test = self.model(x_test)
            test_loss = self.loss_fn(y_pred_test, y_test)
            self.train_losses.append(train_loss.item())
            self.test_losses.append(test_loss.item())
            # accuracies
            train_accuracy = (y_pred_train.round() == y_train).float().mean().item() * 100
            test_accuracy = (y_pred_test.round() == y_test).float().mean().item() * 100
            self.train_accuracies.append(train_accuracy)
            self.test_accuracies.append(test_accuracy)
            # best model
            if test_accuracy > self.max_accuracy:
                self.max_accuracy = test_accuracy
                self.best_model_accuracy = copy.deepcopy(self.model.state_dict())
                self.best_epoch_accuracy = epoch
            if test_loss < self.min_loss:
                self.min_loss = test_loss
                self.best_model_loss = copy.deepcopy(self.model.state_dict())
                self.best_epoch_loss = epoch
            # early stopping
            if test_loss > self.min_loss:
                self.patience -= 1
            else:
                self.patience = 10
            if self.patience < 1:
                break
            epoch += 1

    def plot_result(self, filename):
        plot1 = plt.subplot(1, 2, 1)
        plot1.plot(self.train_losses, label='Train Loss', color='blue')
        plot1.plot(self.test_losses, label='Test Loss', color='green')
        plot1.plot(np.argmin(self.test_losses), np.min(self.test_losses), 'rv')
        plot1.set_xlabel('Epoch')
        plot1.set_ylabel('Cross Entropy Loss')
        plot1.legend()
        plot1.grid()

        plot2 = plt.subplot(1, 2, 2)
        plot2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        plot2.plot(self.test_accuracies, label='Test Accuracy', color='green')
        plot2.plot(np.argmax(self.test_accuracies), np.max(self.test_accuracies), 'rv')
        plot2.set_xlabel('Epoch')
        plot2.set_ylabel('Accuracy (%)')
        plot2.legend()
        plot2.grid()

        plt.suptitle(f'{filename.capitalize()}')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./plots/{filename}.png')
        plt.clf()


