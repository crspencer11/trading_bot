import torch
import torch.nn as nn
import torch.optim as optim

# Check for MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define a basic Long Short-Term Memory model
class ModelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(ModelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 10, device=device)  # move to MPS
        c0 = torch.zeros(1, x.size(0), 10, device=device)  # same
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Create model and move to MPS
model = ModelLSTM(input_size=1, hidden_size=10, output_size=1).to(device)

# Create dummy data
X_train = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5]]], dtype=torch.float32).to(device)
y_train = torch.tensor([[0.6]], dtype=torch.float32).to(device)

# Define loss function, in this case mean standard error, and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train for a few epochs
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training Complete!")
