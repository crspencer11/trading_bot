def main():
    
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

if __name__ == "__main__":
    main()
