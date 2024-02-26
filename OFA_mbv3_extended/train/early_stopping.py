__all__ = ["LossEarlyStoppingMeter", "AccEarlyStoppingMeter"]


class LossEarlyStoppingMeter:

    def __init__(self, patience):
        self.patience = patience
        self.best_val_loss = None
        self.counter = 0

    def update(self, new_val_loss):

        should_stop = False

        if self.best_val_loss is None:
            self.best_val_loss = new_val_loss

        else:

            if self.best_val_loss < new_val_loss:
                self.counter += 1
                if self.counter >= self.patience:
                    should_stop = True
            else:
                self.counter = 0
                self.best_val_loss = new_val_loss

        return should_stop

    def reset(self, patience):
        self.patience = patience
        self.best_val_loss = None
        self.counter = 0


class AccEarlyStoppingMeter:

    def __init__(self, patience):
        self.patience = patience
        self.best_val_acc = None
        self.counter = 0

    def update(self, new_val_acc):

        should_stop = False

        if self.best_val_acc is None:
            self.best_val_acc = new_val_acc

        else:

            if self.best_val_acc > new_val_acc:
                self.counter += 1
                if self.counter >= self.patience:
                    should_stop = True
            else:
                self.counter = 0
                self.best_val_acc = new_val_acc

        return should_stop

    def reset(self, patience):
        self.patience = patience
        self.best_val_acc = None
        self.counter = 0

