import nsml
import keras


class CustomNSML(keras.callbacks.Callback):
    def __init__(self, epoch_total):
        self.epoch_total = epoch_total

    def on_epoch_end(self, epoch, logs=None):
        print(logs)

        train_loss, train_acc = logs['loss'], logs['acc']
        nsml.report(summary=True, epoch=epoch, epoch_total=self.epoch_total, loss=train_loss, acc=train_acc)
        nsml.save(epoch)

