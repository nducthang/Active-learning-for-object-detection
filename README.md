# Active-learning-for-object-detection
Active learning for deep object detection using YOLO

In this project, I study the active learning method for object detection. Specifically, I have active field experiments for weapon detection.

Today, the amount of data for deep learning is increasing, but the manpower to label it is not enough. Therefore, the problem is how to use as little data as possible and still achieve high efficiency for the model. That is why I study actively.

In the active learning method, the person labels the data, which is then fed into the training model. The model then goes on to pick out a small amount of unlabeled data that it thinks after being labeled, it gets better. And give it to the labeler. And the loop goes on like that.

Because of human interaction, object detection model requires speed to be fast and accuracy is also relative. Through experimentation I have chosen YOLO v5.

Theoretical details please see attached pdf report.
