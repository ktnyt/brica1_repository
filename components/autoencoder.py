import numpy as np
from chainer import Variable, FunctionSet, optimizers
import chainer.functions  as F
import cPickle as pickle

import brica1

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

class AutoencoderComponent(brica1.Component):
    def __init__(self, n_input, n_output):
        super(AutoencoderComponent, self).__init__()

    def setup(self, properties):
        self.model = Autoencoder(properties["n_input"], properties["n_output"])
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

        self.make_in_port("input", n_input)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)

    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)

        self.optimizer.zero_grads()
        loss = self.model.forward(x_data)
        loss.backward()
        self.optimizer.update()
        self.results["loss"] = loss.data

        y_data = self.model.encode(x_data)
        self.results["output"] = y_data


f = open("autoencoder.pkl", "wb")
component = AutoencoderComponent
pickle.dump(component, f)
f.close()


