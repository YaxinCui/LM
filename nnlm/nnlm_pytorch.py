from input_data import *
from torch import nn, optim
from torch.autograd import Variable

from input_data import *

import numpy as numpy
import argparse
import time
import math
import torch

# 参考，https://github.com/Vincentkitty/NNLM_by_pytorch/blob/master/NNLM/NNLM.py
# 参考，https://github.com/lsvih/nnlm
# 参考，https://www.jianshu.com/p/be242ed3f314
# 参考，https://github.com/L1aoXingyu/pytorch-beginner/blob/master/06-Natural%20Language%20Process/N-Gram.py


class NNLM(nn.Module):
    def __init__(self, vocab_size, word_dim, win_size, hidden_num):
        # 单词个数，词向量长度，窗口大小，隐藏层数量
        super(NNLM, self).__init__()
        # 将词映射成向量
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.win_size = win_size
        self.hidden_num = hidden_num

        self.embeds = nn.Embedding(vocab_size, word_dim)    # embeds [vocab_size, word_dim]
        torch.nn.init.uniform(self.embeds.weight, a=-2, b=2)

        self.layer1 = nn.Sequential(
                nn.Linear(win_size * word_dim, hidden_num),
                nn.Tanh()
            )
            # []
        
        self.layer2 = nn.Sequential(
                nn.Linear(hidden_num, vocab_size),
                nn.Softmax()
            )
#        torch.nn.init.normal(self.layer1[0].weight, std=1.0 / math.sqrt(hidden_num))
#        torch.nn.init.normal(self.layer1[0].bias, std=1.0 / math.sqrt(hidden_num))
#        torch.nn.init.normal(self.layer2[0].weight, std=1.0 / math.sqrt(win_size * word_dim))
#        torch.nn.init.normal(self.layer2[0].bias, std=1.0 / math.sqrt(hidden_num))

    def forward(self, x):
        x = self.embeds(x)
        x = x.view(-1, self.win_size * self.word_dim)
#        x = x.view(-1, 1)
        out = self.layer1(x)
        out = self.layer2(out)
#        print("weight:", self.embeds.weight[0][:20])
#        embeddings_norm = torch.sqrt(torch.sum(torch.mul(self.embeds.weight, self.embeds.weight), dim=1))
        #print("embed shape", embeddings_norm.shape)
        #print("embed type", embeddings_norm.type)
        #print("weight shape", self.embeds.weight.shape)
        #print("weight type", self.embeds.weight.type)
#        self.embeds.from_pretrained(self.embeds.weight / embeddings_norm.unsqueeze(1))
        
        return out
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='nnlm/data/',
                       help='data directory containing input.txt')
                       
    parser.add_argument('--data_name', type=str, default='input.en.txt',
                       help='data directory containing input.en.txt')
    parser.add_argument('--batch_size', type=int, default=15,
                       help='minibatch size')
    parser.add_argument('--win_size', type=int, default=5,
                       help='context sequence length')
    parser.add_argument('--hidden_num', type=int, default=64,
                       help='number of hidden layers')
    parser.add_argument('--word_dim', type=int, default=50,
                       help='number of word embedding')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='number of epochs')
    parser.add_argument('--grad_clip', type=float, default=10.,
                       help='clip gradients at this value')

    args = parser.parse_args() # 参数集合

    data_loader = TextLoader(args.data_dir, args.data_name, args.batch_size, args.win_size)
    args.vocab_size = data_loader.vocab_size

    criterion = nn.CrossEntropyLoss()
    model = NNLM(args.vocab_size, args.word_dim, args.win_size, args.hidden_num)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for e in range(args.num_epochs):
        data_loader.reset_batch_pointer()
        for b in range(data_loader.num_batches):
            start = time.time()
            x, target = data_loader.next_batch()
            x = torch.from_numpy(x).long()
            target = torch.from_numpy(target.squeeze()).long()
            output = model(x)


            #print("output  ", output[10:20], " \n shape ", output.shape)
            #print("target ", target[10:20], " \n shape ", target.shape)
            
            train_loss = criterion(output, target)
            end = time.time()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                        b, data_loader.num_batches,
                        e, train_loss, end - start))
        # 保存词向量至nnlm_word_embeddings.npy文件
        np.save('nnlm_word_embeddings.en', model.embeds.weight.detach().numpy())



if __name__ == '__main__':
    main()