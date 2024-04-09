import numpy as np

class Layer:
    """
    Abstract class for layers
    """

    def __init__(self, seed: int = 100):
        self.seed = seed
    
    # Overwrite if layer has weights
    def reset_weights(self):
        pass
    
    # Overwrite if layer has schedulers
    def reset_schedulers(self):
        pass

    # Must be overwritten
    def find_output_shape(self) -> tuple | int:
        raise NotImplementedError
    
    # Must be overwritten
    def feed_forward(self, input: np.ndarray):
        raise NotImplementedError

    # Must be overwritten
    def backpropagate(self, dC_doutput: np.ndarray, lmbd: float):
        raise NotImplementedError
