class AlexNetConfig:
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        self.num_classes = num_classes
        self.dropout = dropout
