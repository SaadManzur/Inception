from tensorflow import keras


class Data(object):

    def __init__(self, x, y, size, outputs=1, batch_size=None):
        self.x = x
        self.y = y
        self.size = size
        self.batch_size = batch_size
        self.generator = None
        self.number_of_outputs = outputs

    def generate(self):
        self.generator = keras.preprocessing.ImageDataGenerator()

        while True:
            xi, yi = self.generator.flow(self.x, self.y, batch_size=self.batch_size)

            outputs = {'output': yi}

            for i in range(1, self.number_of_outputs):
                outputs['auxiliary_' + str(i)] = yi

            yield xi, outputs

    def steps_per_epoch(self):
        return self.size // self.batch_size
