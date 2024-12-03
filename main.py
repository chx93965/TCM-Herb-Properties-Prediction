import pandas as pd
import net
from validate import CrossValidation
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./data/dataset.csv')

# 75% training, 25% testing
train = df.sample(frac=0.75, axis=0)
test = df.drop(train.index)


# taste 6 columns
y_sour_train = train.iloc[:, -1]
y_salty_train = train.iloc[:, -2]
y_astringent_train = train.iloc[:, -3]
y_sweet_train = train.iloc[:, -4]
y_pungent_train = train.iloc[:, -5]
y_bitter_train = train.iloc[:, -6]

y_sour_test = test.iloc[:, -1]
y_salty_test = test.iloc[:, -2]
y_astringent_test = test.iloc[:, -3]
y_sweet_test = test.iloc[:, -4]
y_pungent_test = test.iloc[:, -5]
y_bitter_test = test.iloc[:, -6]


# attribution 13 columns
y_others_train = train.iloc[:, -7]
y_pericardium_train = train.iloc[:, -8]
y_small_intestine_train = train.iloc[:, -9]
y_bladder_train = train.iloc[:, -10]
y_triple_burner_train = train.iloc[:, -11]
y_gallbladder_train = train.iloc[:, -12]
y_lung_train = train.iloc[:, -13]
y_stomach_train = train.iloc[:, -14]
y_large_intestine_train = train.iloc[:, -15]
y_heart_train = train.iloc[:, -16]
y_kidney_train = train.iloc[:, -17]
y_spleen_train = train.iloc[:, -18]
y_liver_train = train.iloc[:, -19]

y_others_test = test.iloc[:, -7]
y_pericardium_test = test.iloc[:, -8]
y_small_intestine_test = test.iloc[:, -9]
y_bladder_test = test.iloc[:, -10]
y_triple_burner_test = test.iloc[:, -11]
y_gallbladder_test = test.iloc[:, -12]
y_lung_test = test.iloc[:, -13]
y_stomach_test = test.iloc[:, -14]
y_large_intestine_test = test.iloc[:, -15]
y_heart_test = test.iloc[:, -16]
y_kidney_test = test.iloc[:, -17]
y_spleen_test = test.iloc[:, -18]
y_liver_test = test.iloc[:, -19]


# temperature 5 columns
y_cold_train = train.iloc[:, -20]
y_cool_train = train.iloc[:, -21]
y_even_train = train.iloc[:, -22]
y_warm_train = train.iloc[:, -23]
y_hot_train = train.iloc[:, -24]

y_cold_test = test.iloc[:, -20]
y_cool_test = test.iloc[:, -21]
y_even_test = test.iloc[:, -22]
y_warm_test = test.iloc[:, -23]
y_hot_test = test.iloc[:, -24]

# features 21 columns
x_train = train.iloc[:, :21]
x_test = test.iloc[:, :21]



# standardize
sc = StandardScaler()
x_train = pd.DataFrame(sc.fit_transform(x_train))
x_test = pd.DataFrame(sc.transform(x_test))

# cross validation
y = train.iloc[:, 21:]
cross_val = CrossValidation(x_train, y)
loss_per_target, mean_loss = cross_val.validate()
print(loss_per_target.round(2))
print(f'Cross Validation Mean Loss: {mean_loss}')

# '''
# Hot
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_hot_train, x_test, y_hot_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('hot')
#
# '''
# Warm
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_warm_train, x_test, y_warm_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('warm')
#
# '''
# Even
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_even_train, x_test, y_even_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('even')
#
# '''
# Cool
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_cool_train, x_test, y_cool_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('cool')
#
# '''
# Cold
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_cold_train, x_test, y_cold_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('cold')
#
# '''
# Liver
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_liver_train, x_test, y_liver_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('liver')
#
# '''
# Spleen
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_spleen_train, x_test, y_spleen_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('spleen')
#
# '''
# Kidney
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_kidney_train, x_test, y_kidney_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('kidney')
#
# '''
# Heart
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_heart_train, x_test, y_heart_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('heart')
#
# '''
# Large Intestine
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_large_intestine_train, x_test, y_large_intestine_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('large_intestine')
#
# '''
# Stomach
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_stomach_train, x_test, y_stomach_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('stomach')
#
# '''
# Lung
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_lung_train, x_test, y_lung_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('lung')
#
# '''
# Gallbladder
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_gallbladder_train, x_test, y_gallbladder_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('gallbladder')
#
# '''
# Triple Burner
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_triple_burner_train, x_test, y_triple_burner_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('triple_burner')
#
# '''
# Bladder
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_bladder_train, x_test, y_bladder_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('bladder')
#
# '''
# Small Intestine
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_small_intestine_train, x_test, y_small_intestine_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('small_intestine')
#
# '''
# Pericardium
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_pericardium_train, x_test, y_pericardium_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('pericardium')
#
# '''
# Others
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_others_train, x_test, y_others_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('others')
#
# '''
# Bitter
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_bitter_train, x_test, y_bitter_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('bitter')
#
# '''
# Pungent
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_pungent_train, x_test, y_pungent_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('pungent')
#
# '''
# Sweet
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_sweet_train, x_test, y_sweet_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('sweet')
#
# '''
# Astringent
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_astringent_train, x_test, y_astringent_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('astringent')
#
# '''
# Salty
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_salty_train, x_test, y_salty_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('salty')
#
# '''
# Sour
# '''
# trainer = net.Trainer()
# trainer.train(x_train, y_sour_train, x_test, y_sour_test)
# print(f'Max test accuracy at Epoch {trainer.best_epoch_accuracy}')
# print(f'Min test loss at Epoch {trainer.best_epoch_loss}')
# trainer.plot_result('sour')
#
#
#
#
