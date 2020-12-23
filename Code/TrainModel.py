import time
import numpy as np
from sklearn import metrics


class TrainModel:
    def __init__(self, model):
        self.accuracies = []
        self.model = model

    def print_model_type(self):
        print(self.model.model_type)

    # Huấn luyện bình thường và nhận xác suất cho tập validation
    # Sử dụng xác suất để chọn các mẫu không chắc chắn nhất

    def train(self, X_train, y_train, X_val, X_test, c_weight):
        print("Train set:", X_train.shape, 'y:', y_train.shape)
        print("Val set:", X_val.shape)
        print("Test set:", X_test.shape)
        t0 = time.time()
        (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted) = self.model.fit(
            X_train, y_train, X_val, X_test, c_weight)
        self.run_time = time.time() - t0
        # Trả về trong trường hợp sử dụng PCA
        # Với các trường hợp khác thì không cần
        return (X_train, X_val, X_test)

    # Đo độ chính xác trên tập test
    def get_test_accuracy(self, i, y_test):
        classif_rate = np.mean(
            self.test_y_predicted.ravel() == y_test.ravel())*100
        self.accuracies.append(classif_rate)
        print('--------------------------------')
        print('Iteration:', i)
        print('--------------------------------')
        print('y-test set:', y_test.shape)
        print('Example run in %.3f s' % self.run_time, '\n')
        print("Accuracy rate for %f " % (classif_rate))
        print("Classification report for classifier %s:\n%s\n" % (
            self.model.classifier, metrics.classification_report(y_test, self.test_y_predicted)))
        print("Confusion matrix:\n%s" %
              metrics.confusion_matrix(y_test, self.test_y_predicted))
        print('--------------------------------')
