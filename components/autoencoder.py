import numpy as np
from chainer import Variable, FunctionSet, optimizers
import chainer.functions  as F
import cPickle as pickle

import brica1

class SLP(FunctionSet):
    def __init__(self, n_input, n_output):
        super(SLP, self).__init__(
            transform=F.Linear(n_input, n_output)
        )

    def forward(self, x_data, y_data):
        x = Variable(x_data)
        t = Variable(y_data)
        y = F.sigmoid(self.transform(x))
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        return loss, accuracy

    def predict(self, x_data):
        x = Variable(x_data)
        y = F.sigmoid(self.transform(x))
        return y.data

class Autoencoder(FunctionSet):
    def __init__(self, n_input, n_output):
        super(Autoencoder, self).__init__(
            encoder=F.Linear(n_input, n_output),
            decoder=F.Linear(n_output, n_input)
        )

    def forward(self, x_data):
        x = Variable(x_data)
        t = Variable(x_data)
        x = F.dropout(x)
        h = F.sigmoid(self.encoder(x))
        y = F.sigmoid(self.decoder(h))
        loss = F.mean_squared_error(y, t)
        return loss

    def encode(self, x_data):
        x = Variable(x_data)
        h = F.sigmoid(self.encoder(x))
        return h.data

class SLPComponent(brica1.Component):
    def __init__(self):
        super(SLPComponent, self).__init__()

    def setup(self, n_input, n_output):
        self.model = SLP(n_input, n_output)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

        self.make_in_port("input", n_input)
        self.make_in_port("target", 1)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)
        self.make_out_port("accuracy", 1)

    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)
        t_data = self.inputs["target"].astype(np.int32)

        self.optimizer.zero_grads()
        loss, accuracy = self.model.forward(x_data, t_data)
        loss.backward()
        self.optimizer.update()
        self.results["loss"] = loss.data
        self.results["accuracy"] = accuracy.data

        y_data = self.model.predict(x_data)
        self.results["output"] = y_data

if __name__ == "__main__":
    f = open("autoencoder.pkl", "wb")
    component = SLPComponent()
    pickle.dump(component, f)
    f.close()
