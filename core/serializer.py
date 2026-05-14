import json
import numpy as np
from typing import Any


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return convert_numpy(obj.tolist())
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        if hasattr(obj, 'item'):
            val = obj.item()
            if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
                return None
            return val
        return super().default(obj)


def convert_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, (str, int, bool)):
        return obj
    elif hasattr(obj, 'item'):
        val = obj.item()
        if isinstance(val, (float, np.floating)):
            if np.isnan(val) or np.isinf(val):
                return None
        return val
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj
