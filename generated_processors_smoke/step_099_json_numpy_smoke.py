import numpy as np

def process(payload):
    return {"observation": "ok", "metrics": {"supports": np.bool_(True), "score": np.float64(0.0)}}
