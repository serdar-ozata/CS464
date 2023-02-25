# predicts whether an electric system is prone to error or not
from datetime import time
import pandas
import numpy as np
from sklearn import linear_model
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import time


def question1():
    models = [LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=1, alpha=10 ** -4, epoch=100,
                          question1_enabled=True),
              LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=64, alpha=10 ** -4, epoch=100,
                          question1_enabled=True),
              LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=train_perc, alpha=10 ** -4, epoch=100,
                          question1_enabled=True)]
    accuracies = [model.accuracy for model in models]
    for i in range(len(models)):
        print("Exec: " + str(models[i].executionTime))
    for batchIdx in range(3):
        plt.plot(accuracies[batchIdx])
    plt.legend(["full-batch gradient ascent", "mini-batch gradient ascent", "stochastic gradient ascent"])
    printPlot(models)


def question2():
    models = [LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=64, alpha=10 ** -4, epoch=100,
                          init_technique=0, question1_enabled=True),
              LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=64, alpha=10 ** -4, epoch=100,
                          init_technique=1, question1_enabled=True),
              LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=64, alpha=10 ** -4, epoch=100,
                          init_technique=2, question1_enabled=True)]
    accuracies = [model.accuracy for model in models]
    for batchIdx in range(3):
        plt.plot(accuracies[batchIdx])
    plt.legend(["Gaussian", "Uniform", "Zero"])
    printPlot(models)


def question3():
    models = [LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=train_perc, alpha=1, epoch=100,
                          question1_enabled=True),
              LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=train_perc, alpha=10 ** -3, epoch=100,
                          question1_enabled=True),
              LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=train_perc, alpha=10 ** -4, epoch=100,
                          question1_enabled=True),
              LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=train_perc, alpha=10 ** -5, epoch=100,
                          question1_enabled=True)]
    accuracies = [model.accuracy for model in models]
    for batchIdx in range(4):
        plt.plot(accuracies[batchIdx])
    plt.legend(["1", "1e-3", "1e-4", "1e-5"])
    printPlot(models)


def question4():
    model = LogisticReg(train_x, train_y, mean=0, sd=1, batch_size=train_perc, alpha=1, epoch=100,
                        question1_enabled=True)
    accuracy = model.accuracy
    plt.plot(accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def printPlot(models):
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
    models[0].estimate(eval_x, eval_y, True)
    models[1].estimate(eval_x, eval_y, True)
    models[2].estimate(eval_x, eval_y, True)


def measureSklearn():
    lg = linear_model.LogisticRegression()
    lg.fit(train_x, train_y)
    print(lg.score(eval_x, eval_y))


def normalize(mat):
    r_max = mat.max()
    r_min = mat.min()
    mat -= r_min
    mat /= r_max - r_min


class LogisticReg:
    def __init__(self, train_x, train_y, mean, sd, batch_size, alpha, epoch, init_technique=0, question1_enabled=False):
        start_time = time.time()
        if init_technique == 0:
            self.ws = mean + sd * np.random.randn(train_x.shape[1] + 1)
        elif init_technique == 1:
            size = train_x.shape[1] + 1
            self.ws = np.random.uniform(size=size)
        else:
            self.ws = np.ones(train_x.shape[1] + 1)
        split_limit = train_x.shape[0] // batch_size
        self.alpha = alpha
        self.epoch = epoch
        self.bs = batch_size
        if question1_enabled:
            self.accuracy = np.empty(epoch)
        for ep in range(epoch):
            for i in range(batch_size):
                end = (i + 1) * split_limit
                start = i * split_limit
                self.learn(train_x[start: end, :], train_y[start: end])
            if question1_enabled:
                self.accuracy[ep] = self.estimate(eval_x, eval_y, False)
        self.executionTime = time.time() - start_time

    def estimate(self, eval_x, eval_y, plot_conf_mat: bool):
        predictions = np.empty(shape=eval_y.shape)

        for r in range(eval_x.shape[0]):
            ws_sum = self.ws[0]
            for c in range(eval_x.shape[1]):
                ws_sum += self.ws[c + 1] * eval_x[r][c]
            result = ws_sum > 0
            predictions[r] = result
        y_true = eval_y.astype(dtype=bool)
        y_pred = predictions.astype(dtype=bool)
        if plot_conf_mat:
            mat = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred)
            mat.plot()
            plt.show()
            report = classification_report(y_true=y_true, y_pred=y_pred, target_names=["Stable", "Unstable"])
            print("Batch Size: " + str(self.bs) + ", Alpha: " + str(self.alpha) + ", Epoch: " + str(self.epoch))
            print(report)
        return accuracy_score(y_true=y_true, y_pred=y_pred)

    def learn(self, dt_x, dt_y):
        x_length = len(self.ws) - 1
        ws = np.array(self.ws)
        for j in range(dt_x.shape[0]):
            exp = getExpOfP(self.ws, dt_x[j])
            p = exp / (1 + exp)
            ws[0] += self.alpha * (dt_y[j] - p)
            for i in range(x_length):
                ws[i + 1] += self.alpha * dt_x[j][i] * (dt_y[j] - p)
        self.ws = ws


def getExpOfP(ws, xs):
    ws_sum = ws[0]
    for c in range(xs.shape[0]):
        ws_sum += ws[c + 1] * xs[c]
    return np.exp(ws_sum)


filename = "./data/dataset.csv"
p_data = pandas.read_csv(filename)
data = p_data.to_numpy()
train_perc = 7 * len(data) // 10
train_perc += 64 - (train_perc % 64)
train_x = data[:train_perc, :data.shape[1] - 1]
train_y = data[:train_perc, data.shape[1] - 1]

eval_x = data[train_perc:, :data.shape[1] - 1]
eval_y = data[train_perc:, data.shape[1] - 1]
normalize(train_x)
normalize(eval_x)
print("Data read")

# question1()

# measureSklearn()
