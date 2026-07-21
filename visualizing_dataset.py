import matplotlib.pyplot as plt

def get_length_of_data(dataset):
    return len(dataset)

def get_data(dataset, index):
    inputs, target = dataset[index]
    return inputs, target

def draw_sample(sample_tensor):
    # Squeeze it so it changes from [28,28,1] to [28,28]
    plt.imshow(sample_tensor.squeeze(), cmap="gray")
    plt.show()

def visualize_loss_backwards():
    import torch
    import torch.nn as nn

    # 1. Create a simple model with one linear layer
    model = nn.Linear(2, 1)
    X = torch.randn(1, 2)
    y = torch.tensor([[1.0]])

    # 2. Forward pass
    pred = model(X)

    # 3. Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred, y)

    # --- Let's trace the pointers! ---
    print("1. Loss's creator:", loss.grad_fn)
    # Output: <MseLossBackward0 object at ...> (Points to the loss calculation)

    print("2. The step before loss:", loss.grad_fn.next_functions[0][0])
    # Output: <AddmmBackward0 object at ...> (Points to the model's linear layer calculation!)

    print("3. Inside the linear layer, pointing to the weights:")
    # This points directly to your model's weights and biases in memory!
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
    # Output: <AccumulateGrad object at ...> (The function that writes to model.weight.grad!)

if __name__ == "__main__":
    visualize_loss_backwards()