from main import get_k_random_samples
import numpy as np
from Normalize import Normalize
from TrainModel import TrainModel

max_queried = 500

class ActiveLearning(object):
    accuracies = []

    def __init__(self, initial_labeled_samples, model, selection_function):
        self.initial_labeled_samples = initial_labeled_samples
        self.model = model
        self.sample_selection_function = selection_function

    def run(self, X_train_full, y_train_full, X_test, y_test):
        (permutation, X_train, y_train) = get_k_random_samples(
            self.initial_labeled_samples, X_train_full, y_train_full)
        self.queried = self.initial_labeled_samples
        self.samplecount = [self.initial_labeled_samples]

        # assign the val set the rest of the 'unlabelled' training data
        X_val = np.array([])
        y_val = np.array([])
        X_val = np.copy(X_train_full)
        X_val = np.delete(X_val, permutation, axis=0)
        y_val = np.copy(y_train_full)
        y_val = np.delete(y_val, permutation, axis=0)
        print('Val set:', X_val.shape, y_val.shape, permutation.shape)
        print()

        normalizer = Normalize()
        X_train, X_val, X_test = normalizer.normalize(X_train, X_val, X_test)

        self.clf_model = TrainModel(self.model)
        (X_train, X_val, X_test) = self.clf_model.train(
            X_train, y_train, X_val, X_test, 'balanced')
        active_iteration = 1
        self.clf_model.get_test_accuracy(1, y_test)

        while self.queried < max_queried:

            active_iteration += 1

            # get validation probabilities

            probas_val = \
                self.clf_model.model_object.classifier.predict_proba(X_val)
            print('val predicted:',
                  self.clf_model.val_y_predicted.shape,
                  self.clf_model.val_y_predicted)
            print('probabilities:', probas_val.shape, '\n',
                  np.argmax(probas_val, axis=1))

            # select samples using a selection function

            uncertain_samples = \
                self.sample_selection_function.select(
                    probas_val, self.initial_labeled_samples)

            # normalization needs to be inversed and recalculated based on the new train and test set.

            X_train, X_val, X_test = normalizer.inverse(X_train, X_val, X_test)

            # get the uncertain samples from the validation set

            print('trainset before', X_train.shape, y_train.shape)
            X_train = np.concatenate((X_train, X_val[uncertain_samples]))
            y_train = np.concatenate((y_train, y_val[uncertain_samples]))
            print('trainset after', X_train.shape, y_train.shape)
            self.samplecount.append(X_train.shape[0])

            bin_count = np.bincount(y_train.astype('int64'))
            unique = np.unique(y_train.astype('int64'))
            print(
                'updated train set:',
                X_train.shape,
                y_train.shape,
                'unique(labels):',
                bin_count,
                unique,
            )

            X_val = np.delete(X_val, uncertain_samples, axis=0)
            y_val = np.delete(y_val, uncertain_samples, axis=0)
            print('val set:', X_val.shape, y_val.shape)
            print()

            # normalize again after creating the 'new' train/test sets
            normalizer = Normalize()
            X_train, X_val, X_test = normalizer.normalize(
                X_train, X_val, X_test)

            self.queried += self.initial_labeled_samples
            (X_train, X_val, X_test) = self.clf_model.train(
                X_train, y_train, X_val, X_test, 'balanced')
            self.clf_model.get_test_accuracy(active_iteration, y_test)

        print('final active learning accuracies',
              self.clf_model.accuracies)
