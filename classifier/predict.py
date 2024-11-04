import os
import torch
from classifier.data import line_to_tensor, all_categories
from classifier.model import RNN

# predict the category of an input line
def predict(input_line, n_predictions=3):

    # load the model
    (kwargs, state), wtf = torch.load('models/classifier.pth', weights_only=True) if os.path.exists('models/classifier.pth') else (None, None), None
    rnn = None
    if state:
        rnn = RNN(**kwargs)
        rnn.load_state_dict(state)

    if rnn is None:
        print('Model not found, please train the model first')
        return
    
    # evaluate the prediction of an input line
    def evaluate(line_tensor):
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        return output

    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(line_to_tensor(input_line))

        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

        return predictions