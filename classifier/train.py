import math
import os
import random
import time
import torch
import torch.nn as nn
from classifier.data import (
    n_letters, 
    n_categories, 
    all_categories, 
    category_lines,
    line_to_tensor
)
from classifier.model import RNN

def train_model(cont=False):
    rnn = None

    # continue training the model
    if cont:
        # load the model if it exists
        if os.path.exists('models/classifier.pth'):
            kwargs, state = torch.load('models/classifier.pth', weights_only=True)
            if state:
                rnn = RNN(**kwargs)
                rnn.load_state_dict(state)
        else:
            raise Exception('Model not found, cannot continue training')
    
    else:
        # number of hidden units
        n_hidden = 128
        rnn = RNN(n_letters, n_hidden, n_categories)

    # convert output to category
    # meaning: get the category with the highest probability
    # for example, if the output is [0.1, 0.2, 0.7],
    # then the category is the third one
    def category_from_output(output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return all_categories[category_i]

    # randomly choose an element from a list
    def random_choice(l):
        return l[random.randint(0, len(l) - 1)]

    # randomly choose a category and a line, along with their tensors
    def random_training_example():
        category = random_choice(all_categories)
        line = random_choice(category_lines[category])
        category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
        line_tensor = line_to_tensor(line)
        return category, line, category_tensor, line_tensor

    # use negative log likelihood loss
    criterion = nn.NLLLoss()
    # learning rate
    learning_rate = 0.005

    # training function for one example
    def train(category_tensor, line_tensor):
        hidden = rnn.init_hidden()

        rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item()


    # number of iterations (epochs)
    n_iters = 100000
    # print every n iterations
    print_every = 5000
    # trace the loss every n iterations (for plotting)
    plot_every = 1000

    # keep track of the loss
    current_loss = 0
    all_losses = []

    # time elapsed (in minutes and seconds)
    def time_since(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    # start time
    start = time.time()

    # train the model
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = random_training_example()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        if iter % print_every == 0:
            guess = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # save the model: configuration and weights
    os.makedirs('models', exist_ok=True)
    torch.save([rnn.kwargs, rnn.state_dict()], 'models/classifier.pth')