from input_data import *
from torch import nn, optim
from torch.autograd import Variable

import numpy as np
import argparse
import time
import math
import torch

# 环境，pytorch1.3
# 参考，https://github.com/Vincentkitty/NNLM_by_pytorch/blob/master/NNLM/NNLM.py
# 参考，https://github.com/lsvih/nnlm
# 参考，https://www.jianshu.com/p/be242ed3f314
# 参考，https://github.com/L1aoXingyu/pytorch-beginner/blob/master/06-Natural%20Language%20Process/N-Gram.py

# 总结
# 之前使用Softmax的损失值一直在8.5，一直降不下来，直到使用LogSoftmax（）才下降到4.5左右，在目标比较稀疏的时候，使用LogSoftmax效果更好
# 除了损失函数使用Softmax外，参数的初始化对损失值下降也有很大影响
# 对embeds的初始值设置为正态分布，比均匀分布梯度下降得更快
# 建议使用grad_clip，使得梯度截断不是很大
# batch_size设置太多，也会影响梯度下降速度

# debug过程，删去tensorflow版nnlm的各种优化，可以得到各种优化效果到底是什么效果

class NNLM(nn.Module):
    def __init__(self, vocab_size, word_dim, win_size, hidden_num, grad_clip):
        # 单词个数，词向量长度，窗口大小，隐藏层数量
        super(NNLM, self).__init__()
        # 将词映射成向量
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.win_size = win_size
        self.hidden_num = hidden_num

        self.embeds = nn.Embedding(vocab_size, word_dim)    # embeds [vocab_size, word_dim] ，不能有max_norm，否则报错
    #    torch.nn.init.uniform(self.embeds.weight, a=-1, b=1)
        torch.nn.init.normal(self.embeds.weight) # 使用正态分布，梯度下降得更快
        
        self.layer1 = nn.Sequential(
                nn.Linear(win_size * word_dim, hidden_num),
                nn.Tanh()
            )
            # []
        
        self.layer2 = nn.Linear(hidden_num, vocab_size)
        self.layer3 = nn.Linear(win_size * word_dim, vocab_size)
        self.layer4 = nn.LogSoftmax(dim=1)
        a = [win_size * word_dim * hidden_num]
        
        """
        # 有没有初始化，收敛速度差不多
        nn.init.normal(self.layer1[0].weight/(torch.sqrt(torch.Tensor(a))))
        a = [hidden_num * vocab_size]
        nn.init.normal(self.layer2.weight/(torch.sqrt(torch.Tensor(a))))
        a = [win_size * word_dim * vocab_size]
        nn.init.normal(self.layer3.weight/(torch.sqrt(torch.Tensor(a))))
        """

#        self.layer4 = nn.LogSoftmax()

    def forward(self, x):
        x = self.embeds(x)
        x = x.view(-1, self.win_size * self.word_dim)
#        x = x.view(-1, 1)
        hidden = self.layer1(x)
        out = self.layer2(hidden) + self.layer3(x)
        out = self.layer4(out)
        
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
    model = NNLM(args.vocab_size, args.word_dim, args.win_size, args.hidden_num, args.grad_clip)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)

    model.train()
    for e in range(args.num_epochs):
        data_loader.reset_batch_pointer()
        for b in range(data_loader.num_batches):
            start = time.time()
            x, target = data_loader.next_batch()
            x = torch.from_numpy(x).long()
            target = torch.from_numpy(target.squeeze()).long()
            
            if torch.cuda.is_available():
                x = x.cuda()
                target = target.cuda()

            output = model(x)
            
            train_loss = criterion(output, target)
            # 截断梯度，正则化，防止梯度爆炸
            nn.utils.clip_grad_norm(model.parameters(), max_norm=args.grad_clip, norm_type=2)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            end = time.time()
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                        b, data_loader.num_batches,
                        e, train_loss, end - start))
        # 保存词向量
        embeds = model.embeds.weight.detach().numpy()
        np.save(args.data_dir+"nnlm_"+str(args.data_name).replace(".txt", "") + ".npy", embeds)


if __name__ == '__main__':
    main()