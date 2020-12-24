""" THUẬT TOÁN ACTIVE LEARNING
Require:
    - Tập các mẫu đã gán nhãn L
    - Tạp các mẫu chưa gán nhãn U
    - Model khởi tạo f0
    - Active learning metric v
Algorithm:
1. Chia U thành các batches
2. f <- f0
3. Nếu U vẫn trống hoặc chưa đạt điều kiện dừng thì:
    - Tính scores cho tất cả batches của U sử dụng f
    - U_best <- Batches điểm cao nhất trong U theo v
    - Y_best <- Gán nhãn cho U_best (người)
    - Train f sử dụng L và (U_best, Y_best)
    - U = U\U_best
    - L = L + (U_best, y_best)
"""

class ActiveLearning(object):
    def __init__(self, model, select_function):
        self.model = model
        self.select_function = select_function

    def run(self, X_train, y_train, X_test, y_test):
        pass

if __name__ == '__main__':
    pass
