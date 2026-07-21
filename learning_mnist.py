import torch.cuda
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import visualizing_dataset
from PIL import Image

def download_data(download=True):
    training_dataset = datasets.MNIST(
        root="datasets/mnist",
        train=True,
        download=download,
        transform=transforms.ToTensor()
    )

    testing_dataset = datasets.MNIST(
        root="datasets/mnist",
        train=False,
        download=download,
        transform=transforms.ToTensor()
    )

    return training_dataset, testing_dataset

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(model, loss_fn, optimizer, train_dataloader, device):
    model.train()
    for batch, (X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()


        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}, [{current:>5d} / {len(train_dataloader.dataset) :>5d}]")


def test(model, loss_fn, dataloader, device):
    model.to(device)
    model.eval()
    test_loss, correct = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

def use_model(model_path, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MyNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28,28))
    ])
    image_tensor = transform(image).to(device)

    with torch.inference_mode():
        out = model(image_tensor)
        prediction = torch.argmax(out).item()
        print(f"Model predict this image is a: {prediction}")


if __name__ == "__main__":

    # Prepare the goods
    train_data, test_data = download_data(False)

    # Packing the goods into nice boxes
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}")

    # create the model, the loss and optimizer
    model = MyNN().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    epochs = 17
    for i in range(epochs):
        print (f"============== Epoch {i} ===================")
        train(model, loss, optimizer, train_dataloader, device)
        test(model, loss, test_dataloader, device)

    print("Done")
    torch.save(model.state_dict(), "models/mnist/mnist_base_model.pth")
    print("Saved Model")


    # Test 4
    use_model("models/mnist/mnist_base_model.pth",
              "models/mnist/tests/testA.jpg")

    # Test 1
    use_model("models/mnist/mnist_base_model.pth",
              "models/mnist/tests/testB.jpg")

    # Test 6
    use_model("models/mnist/mnist_base_model.pth",
              "models/mnist/tests/testC.jpg")



