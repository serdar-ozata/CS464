import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('bmh')


class LinearRegression(object):
    def __init__(self, data: np.array, labels: np.array, val_split_ratio: float = 0.30, learning_rate: float = 0.01):
        # add bias term to the data
        data = np.c_[data, np.ones((data.shape[0], 1))]

        # standardize
        mu = np.sum(data, axis=0)
        data -= mu
        sigma = np.sum(data ** 2, axis=0)
        data /= sigma

        # uncomment this if you want to get the shuffled version
        # shuffle
        # shuffler = np.arange(data.shape[0])
        # np.random.shuffle(shuffler)
        # data = data[shuffler]
        # labels = labels[shuffler]

        # init data
        index = int(data.shape[0] * val_split_ratio)
        self.train_data = data[index:, :]
        self.train_labels = labels[index:, :]
        self.test_data = data[:index, :]
        self.test_labels = labels[:index, :]
        self.learning_rate = learning_rate

        # init tetha
        self.tetha = np.zeros((data.shape[1], 1))

    def solve(self):
        self.tetha = np.linalg.pinv(self.train_data.T @ self.train_data) @ self.train_data.T @ self.train_labels

    def iterate(self):
        self.tetha -= self.learning_rate * self.gradient()

    def gradient(self):
        return (2 / self.train_data.shape[0]) * self.train_data.T @ (self.train_data @ self.tetha - self.train_labels)

    def validation_error(self):
        # error
        error = (self.test_data @ self.tetha - self.test_labels)
        n = error.shape[0]

        # MSE, MEAN, VAR
        MSE = (1 / n) * error.T @ error
        MEAN = (1 / n) * np.sum(error.reshape((-1,)))
        normalized_error = error - MEAN
        VAR = (1 / n) * normalized_error.T @ normalized_error
        return MSE[0, 0], MEAN, VAR[0, 0], error.reshape((-1,))

    def validation_binary_error(self):
        predictions = self.test_data @ self.tetha
        labels = self.test_labels

        predictions = predictions.reshape((-1,))
        labels = labels.reshape((-1,))

        previous_labels = labels[1:]

        predictions = predictions[:-1]
        labels = labels[:-1]

        predicted_increase = predictions > previous_labels
        predicted_decrease = predictions < previous_labels
        label_increase = labels > previous_labels
        label_decrease = labels < previous_labels
        correct_predictions = predicted_increase * label_increase + predicted_decrease * label_decrease

        return np.sum(correct_predictions.reshape((-1,))) / correct_predictions.size

    def plot_validation_prediction(self):
        prediction = (self.test_data @ self.tetha).reshape((-1,))
        labels = (self.test_labels).reshape((-1,))

        # sorting so we see something more meaningful

        plt.figure(0, figsize=(8, 4))
        plt.clf()
        plt.plot(prediction, '-b')
        plt.plot(labels, '-r')
        plt.legend(['predictions', 'labels'])
        plt.xlabel("Sample Index")
        plt.ylabel("Close Value")
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv(".\\data\\ASIANPAINT.csv", index_col=None, header=0)
    df['Date'] = pd.to_numeric(df.Date.str.replace('-', ''))
    df.drop(columns=['Symbol', 'Series'], inplace=True)
    df.dropna(axis='index', how='any', inplace=True)
    data = df.to_numpy()[:-1]
    labels = df['Close'].to_numpy()[1:].reshape((-1, 1))

    lr = LinearRegression(data, labels)
    validation_error = lr.validation_error()
    print(f"(Liner Regression)> Validation MSE: {int(validation_error[0])}")
    print(f"(Liner Regression)> Validation ERROR MEAN: {int(validation_error[1])}")
    print(f"(Liner Regression)> Validation ERROR VARIANCE: {int(validation_error[2])}")
    print(f"(Liner Regression)> Validation Accuracy: {round(lr.validation_binary_error(), 2)}")
    lr.plot_validation_prediction()
    fig = sns.displot(validation_error[3], kind="kde", fill=True, height=4, aspect=2)
    fig.set_axis_labels("Error (label - prediction)", "Density")
    plt.tight_layout()
    plt.show()

    print("(Liner Regression)> Solving...")
    lr.solve()
    validation_error = lr.validation_error()
    print(f"(Liner Regression)> Validation MSE: {int(validation_error[0])}")
    print(f"(Liner Regression)> Validation ERROR MEAN: {int(validation_error[1])}")
    print(f"(Liner Regression)> Validation ERROR VARIANCE: {int(validation_error[2])}")
    print(f"(Liner Regression)> Validation Accuracy: {round(lr.validation_binary_error(), 2)}")
    lr.plot_validation_prediction()
    fig = sns.displot(validation_error[3], kind="kde", fill=True, height=4, aspect=2)
    fig.set_axis_labels("Error (label - prediction)", "Density")
    plt.tight_layout()
    plt.show()
