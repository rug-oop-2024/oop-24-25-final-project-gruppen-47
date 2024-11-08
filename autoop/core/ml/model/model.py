
from abc import abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model:
    def map(self, input_feature, target_feature):
        raise NotImplementedError("This should be implemented by you.")
    
