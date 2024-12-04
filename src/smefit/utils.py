# -*- coding: utf-8 -*-
import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.float32, np.float64, np.int32, np.int64)):
            return o.item()  # Convert to native Python type
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
