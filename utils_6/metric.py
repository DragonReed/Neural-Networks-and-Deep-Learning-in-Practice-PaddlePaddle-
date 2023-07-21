import paddle
from paddle.metric import Metric
class Accuracy(Metric):
    def __init__(self):
        # 用于统计正确的样本个数
        self.num_correct = 0
        # 用于统计样本的总数
        self.num_count = 0

    def compute(self, logits, labels):
        
        # 获取最大元素值的索引
        preds = paddle.argmax(logits, axis=-1)
        labels = labels.squeeze(-1)
        # 获取本批数据中预测正确的样本个数
        batch_correct = paddle.sum(paddle.cast(preds==labels, dtype="float32")).numpy()[0]
        batch_count = len(labels)
        return batch_correct, batch_count

    def update(self, batch_correct, batch_count):
        # 更新num_correct 和 num_count
        self.num_correct += batch_correct
        self.num_count += batch_count
    
    def accumulate(self):
        # 使用累计的数据，计算总的指标
        if self.num_count == 0:
            return 0
        return self.num_correct/self.num_count

    def reset(self):
        self.num_correct = 0
        self.num_count = 0

    def name(self):
        return "Accuracy"