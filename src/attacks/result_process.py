from utils import Pickler


class EfficientResult:
    def __init__(self, result=None):
        if result is not None:
            self.pop = result.pop
            self.initial_state = result.initial_state
