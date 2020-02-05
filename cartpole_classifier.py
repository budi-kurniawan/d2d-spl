import numpy as np
from sklearn.neural_network import MLPClassifier
from datetime import datetime
import time
import pickle

# scikit MLP: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

max_records = 162
hidden_layer_sizes = [80, 80] # the default # all 1.0, but trial 5 0.9375
#hidden_layer_sizes = [75, 75] # not very successful, got low on trial 9 (12.3) and trial 8 (18.71)
#hidden_layer_sizes = [80, 80, 80] # all 1.0, trial 9: 0.9697, results not good, got 12.72 and 27.66
#hidden_layer_sizes = [70, 70] # all 1.0, trial 8 0.9589. Results ok but trial 9 is 9.29
#hidden_layer_sizes = [60, 60] # all 1.0, except trial 2: 0.9896, Results ok, some good some bad
#hidden_layer_sizes = [50, 50] # 6 1.0, trials 3,4,8,9 0.9xxx, results ok, not as good as the default
#hidden_layer_sizes = [40, 40] # 7 1.0, except trials 1, 7, 9, results ok, not as good as the default
 
solver = 'lbfgs'
alpha = 0.0001

def create_classifier(trial_no):
    file = 'results/d2d_results/trainingset' + str(trial_no).zfill(2) + '.txt'
    pickled_path = 'results/cartpole-classifier' + str(trial_no).zfill(2) + '.p'
    X, y = prepare_data(file, max_records)
    classifier = MLPClassifier(solver=solver, alpha=alpha, random_state=1, max_iter=1_000_000, hidden_layer_sizes=hidden_layer_sizes).fit(X, y)
    pickle.dump(classifier, open(pickled_path, "wb"))
    score = classifier.score(X, y)
    print('classifier score for trial ' + str(trial_no) + ': ' + str(score))

def prepare_data(file, max_records=1_000_000):
    X = []
    y = []
    f = open(file, "r")
    for line in f:
        index1 = line.index('[')
        index2 = line.index(']', index1 + 1)
        input = line[index1 + 1 : index2]
        input = list(input.split(','))
        input = [float(i) for i in input]
        index1 = line.index('[', index2 + 1)
        index2 = line.index(']', index1 + 1)
        output = line[index1 + 1 : index2]
        output = list(output.split(','))
        output = [float(i) for i in output]
        if sum(output) == 0:
            continue
        X.append(input)
        y.append(np.argmax(output))
        if len(X) == max_records:
            break
    return X, y

def main():
    num_trials = 10
    for trial in range(num_trials):
        np.random.seed(trial)
        create_classifier(trial)

if __name__ == '__main__':
    main()