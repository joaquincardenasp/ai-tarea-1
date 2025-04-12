import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size=3, hidden1_size= 4, hidden2_size=8, output_size= 1):
        super(MLP, self).__init__()

        self.hidden_layer_1 = nn.Linear(input_size, hidden1_size) # input -> hidden1
        self.hidden_layer_2 = nn.Linear(hidden1_size, hidden2_size) # hidden1 -> hidden2
        self.output_layer = nn.Linear(hidden2_size, output_size) # hidden2 -> output

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.sigmoid(self.hidden_layer_1(x))
        x = self.sigmoid(self.hidden_layer_2(x))
        x = self.sigmoid(self.output_layer(x)) # Usar para salida sigmoid
        #x = self.output_layer(x) # Usar para salida softmax
        return x