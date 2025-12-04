import torch
import torch.nn as nn

class PerSQ(nn.Module):
    def __init__(self, input_size, hidden_units=[50, 30, 20], dropout_rate=0.2):
        super(PerSQ, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_units[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_units[0], hidden_units[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_units[1], hidden_units[2], batch_first=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_units[2], 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Layer 1
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        
        # Layer 2
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        
        # Layer 3
        out, (hn, cn) = self.lstm3(out)
        # We take the output of the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        
        # Output Layer
        out = self.fc(out)
        
        return out
