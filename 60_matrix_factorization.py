import glob
import math
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
import torch.nn.functional as F


class MF(nn.Module):
    def __init__(self, input_items, input_users):
        super(MF, self).__init__()

        self.l_b1 = nn.Embedding(num_embeddings=input_items, embedding_dim=512)
        self.l_b2 = nn.Linear(
            in_features=512, out_features=512, bias=True)
        self.l_b3 = nn.Linear(
            in_features=512, out_features=1, bias=True)

        self.l_a1 = nn.Embedding(num_embeddings=input_users, embedding_dim=512)
        self.l_a2 = nn.Linear(
            in_features=512, out_features=512, bias=True)
        self.l_a3 = nn.Linear(
            in_features=512, out_features=1, bias=True)

    def encode_item(self, x):
        x = F.relu(self.l_b1(x))
        x = F.relu(self.l_b2(x))
        x = F.relu(self.l_b3(x))
        return x

    def encode_user(self, x):
        x = F.relu(self.l_a1(x))
        x = F.relu(self.l_a2(x))
        x = F.relu(self.l_a3(x))
        return x

    def decode(self, x):
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        return x

    def forward(self, inputs):
        item_vec, user_vec = inputs
        item_vec = self.encode_item(item_vec)
        batch_size = list(item_vec.size())[0]

        user_vec = self.encode_user(user_vec)
        # print(user_vec.size())
        doted = torch.bmm(user_vec.view(batch_size, 1, 1),
                          item_vec.view(batch_size, 1, 17770))
        doted = doted.view(batch_size, 17770)
        print(doted.size())
        # exit()
        return doted


def myLoss(output, target):
    # loss = nn.MSELoss()(output, target)
    loss = torch.sqrt(torch.mean((output-target)**2))
    return loss


def generate():
    train_movies = pickle.load(open('works/dataset/train_movies.pkl', 'rb'))
    train_users = pickle.load(open('works/dataset/train_users.pkl', 'rb'))
    print(train_users.shape)

    BATCH = 4
    for i in range(0, train_users.shape[0], BATCH):
        yield (train_movies[i:i+BATCH], train_users[i:i+BATCH])


if __name__ == '__main__':
    movie_index = json.load(open('./works/defs/smovie_index.json'))
    user_index = json.load(open('./works/defs/user_index.json'))
    print(len(user_index))
    model = MF(len(movie_index), len(user_index)).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for index, (train_movies, train_users) in enumerate(generate()):
        print(train_movies.shape)
        print(train_users.shape)
        inputs_movie = Variable(torch.from_numpy(
            train_movies.todense())).float().cuda()
        inputs_user = Variable(torch.from_numpy(
            train_users.todense())).float().cuda()
        predict = model([inputs_movie, inputs_user])

        loss = myLoss(predict, inputs_movie)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        if index % 100 == 0:
            inputs = Variable(torch.from_numpy(
                testData)).float()
            model.to('cpu')
            loss = myLoss(inputs, model(inputs))
            print(math.sqrt(loss.data.cpu().numpy()))
            del inputs
            model.to('cuda')
        '''
        #torch.save(model.state_dict(), f'conv_autoencoder_{epoch:04d}.pth')
