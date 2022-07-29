import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    num_params = []
    train_squared_loss = []
    test_squared_loss = []

    # Load train and test loss at epoch 200 from log.json
    with open("log.json") as f:
        for line in f:
            log = json.loads(line)

            if log["epoch"] == 200:
                num_params.append(log["num_params"])
                train_squared_loss.append(log["train_squared_loss"])
                test_squared_loss.append(log["test_squared_loss"])

            print(log)

    plt.figure(figsize=(8, 4))
    plt.title("Unsuccessful attempt at reproducing double descent ")

    # Plot train and test loss
    plt.semilogx(np.array(num_params) / 1000, test_squared_loss, "D-", label="Test")
    plt.semilogx(np.array(num_params) / 1000, train_squared_loss, label="Train")

    # Plot vertical black bar
    x = [40, 40]
    y = [0, max(test_squared_loss)]
    plt.plot(x, y, "--", color="black", label="No peak here")

    plt.legend()
    xticks = [3, 10, 40, 100, 300, 800]
    plt.xticks(ticks=xticks, labels=xticks)
    plt.minorticks_off()
    plt.ylabel("Squared loss")
    plt.xlabel("Number of parameters/weights ($\\times 10^3$)")
    plt.tight_layout()
    plt.savefig("double_descent.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
