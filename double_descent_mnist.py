import time, json
import torch
import torch.nn as nn
import torchvision.datasets


def run_epoch(model, images, labels, batch_size, optimizer=None):
    num_correct = 0
    squared_loss = 0.0

    assert len(images) % batch_size == 0

    for i in range(0, len(images), batch_size):
        # Get consecutive batches
        inputs = images[i : i + batch_size]
        targets = torch.eye(10)[labels[i : i + batch_size]]

        # Compute predictions
        predictions = model(inputs)

        # Count how many predictions were correct
        correct = torch.argmax(targets, dim=1) == torch.argmax(predictions, dim=1)
        num_correct += torch.sum(correct).item()

        # Compute mean squared error and sum of squared errors
        loss = torch.mean(torch.square(predictions - targets))
        squared_loss += torch.sum(torch.square(predictions - targets)).item()

        # Train network if we got an optimizer
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    squared_loss /= len(images)
    accuracy = num_correct / len(images)

    return squared_loss, accuracy


def train(H, train_images, train_labels, test_images, test_labels, device):
    batch_size = 100
    d = 28 * 28
    K = 10
    num_epochs = 200
    lr = 0.1

    # Create neural network
    model = nn.Sequential(
        nn.Flatten(), nn.Linear(d, H), nn.ReLU(), nn.Linear(H, K),
    ).to(device)

    # Check whether the network has the correct number of parameters
    num_params = int(sum(W.numel() for W in model.parameters()))
    expected_num_params = (d + 1) * H + (H + 1) * K
    assert num_params == expected_num_params

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    start_time = time.perf_counter()
    for epoch in range(1, 1 + num_epochs):
        # Train neural network (with optimizer)
        train_squared_loss, train_accuracy = run_epoch(
            model, train_images, train_labels, batch_size, optimizer
        )

        # Test neural network (without optimizer)
        test_squared_loss, test_accuracy = run_epoch(
            model, test_images, test_labels, batch_size
        )

        # Log results
        log = {
            "epoch": epoch,
            "d": d,
            "K": K,
            "H": H,
            "batch_size": batch_size,
            "num_params": num_params,
            "test_accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
            "test_squared_loss": test_squared_loss,
            "train_squared_loss": train_squared_loss,
            "time": time.perf_counter() - start_time,
        }

        print(log)

        with open("log.json", "a+") as f:
            f.write(json.dumps(log) + "\n")


def main():
    # Seed random number generator to make results somewhat reproducible
    torch.manual_seed(0)

    # Load dataset
    mnist_train = torchvision.datasets.MNIST(".", train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(".", train=False, download=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only train on first 4000 images
    train_images = mnist_train.data[:4000].to(device) / 255.0
    train_labels = mnist_train.targets[:4000].to(device)
    test_images = mnist_test.data.to(device) / 255.0
    test_labels = mnist_test.targets.to(device)

    # Train several networks with different hidden layer sizes H
    for H in [3, 5, 10, 20, 30, 40, 50, 100, 150, 200, 300, 500, 800]:
        train(H, train_images, train_labels, test_images, test_labels, device)


if __name__ == "__main__":
    main()
