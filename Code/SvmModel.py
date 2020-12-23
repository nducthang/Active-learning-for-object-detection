from BaseModel import BaseModel
from sklearn.svm import LinearSVC, SVC


class SvvmModel(BaseModel):
    model_type = 'SVM with linear kernel'

    def fit(self, X_train, y_train, X_val, X_test, c_weight):
        print("Training svm...")
        self.classifier = SVC(C=1, kernel='linear',
                              probabolity=True, class_weight=c_weight)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)
