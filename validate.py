import numpy as np
import net


class CrossValidation:
    def __init__(self, x_df, y_df, k_fold=5):
        self.x_df = x_df
        self.y_df = y_df
        self.k_fold = k_fold

    def validate(self):
        n_targets = self.y_df.shape[1]
        n_samples = self.x_df.shape[0]
        fold_size = n_samples // 5
        loss_per_target = np.zeros(n_targets)

        for i in range(n_targets):
            y = self.y_df.iloc[:, i]
            mean_loss = 0
            for j in range(self.k_fold):
                start = j * fold_size
                end = (j + 1) * fold_size
                x_train = self.x_df.drop(self.x_df.index[start:end])
                y_train = y.drop(y.index[start:end])
                x_test = self.x_df.iloc[start:end]
                y_test = y.iloc[start:end]

                trainer = net.Trainer()
                trainer.train(x_train, y_train, x_test, y_test)
                mean_loss += trainer.min_loss
            loss_per_target[i] = mean_loss / self.k_fold

        return loss_per_target, loss_per_target.mean()

