from sklearn.preprocessing import MinMaxScaler

class Normalize(object):
    def normalize(self, X_train, X_val, X_test):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return (X_train, X_val, X_test)

    def inverse(self, X_train, X_val, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_val = self.scaler.inverse_transform(X_val)
        X_test = self.scaler.inverse_transform(X_test)
        return (X_train, X_val, X_test)