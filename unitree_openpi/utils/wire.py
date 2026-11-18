"""OpenPI-compatible MessagePack NumPy encoding, plus a JSON diagnostic codec.

Implements the public ndarray/scalar wire format without pickle. JSON carries
images as base64 JPEGs; MessagePack carries uint8 RGB ndarrays as OpenPI expects.
"""

import base64
import json
import math

import msgpack
import numpy as np


def _pack(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind not in "biuf":
            raise ValueError(f"unsupported numpy dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {b"__ndarray__": True, b"data": value.tobytes(),
                    b"dtype": value.dtype.str, b"shape": value.shape}
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"unsupported MessagePack object: {type(value).__name__}")


def _unpack(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("unsafe or unsupported numpy dtype")
        if b"__npgeneric__" in value:
            return dtype.type(value[b"data"])
        shape = value[b"shape"]
        if (not isinstance(shape, (list, tuple)) or len(shape) > 8 or
                any(type(n) is not int or n < 0 for n in shape)):
            raise ValueError("invalid ndarray dimensions")
        data = value[b"data"]
        if math.prod(shape) * dtype.itemsize != len(data):
            raise ValueError("ndarray byte length does not match shape")
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    return value


def packb(value):
    return msgpack.packb(value, default=_pack, use_bin_type=True)


def unpackb(payload):
    return msgpack.unpackb(payload, object_hook=_unpack, raw=False, strict_map_key=False)


def json_encode(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            if item.dtype == np.uint8 and item.ndim == 3 and item.shape[2] == 3:
                import cv2
                ok, encoded = cv2.imencode(".jpg", item[:, :, ::-1])
                if not ok:
                    raise ValueError("JPEG encoding failed")
                return {"encoding": "jpeg_base64", "color_space": "RGB",
                        "shape": list(item.shape),
                        "data": base64.b64encode(encoded).decode("ascii")}
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    return json.dumps(value, default=convert, allow_nan=False)


def json_decode(payload):
    def convert(item):
        if item.get("encoding") == "jpeg_base64":
            import cv2
            data = base64.b64decode(item["data"], validate=True)
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("invalid JPEG")
            return bgr[:, :, ::-1].copy()
        return item
    return json.loads(payload, object_hook=convert)
