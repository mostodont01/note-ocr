import base64
import io

import cv2
import numpy as np

from notocr.native_host import decode_capture, handle, read_message, write_message


def data_url(img):
    ok, png = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()


def test_message_framing():
    buf = io.BytesIO()
    write_message(buf, {"a": 1, "ü": "x"})
    buf.seek(0)
    assert read_message(buf) == {"a": 1, "ü": "x"}
    assert read_message(buf) is None


def test_crop_uses_device_pixel_ratio():
    img = np.zeros((200, 400, 3), np.uint8)
    img[40:80, 100:180] = 255
    out = decode_capture(data_url(img), {"x": 50, "y": 20, "w": 40, "h": 20}, 2.0)
    assert out.shape[:2] == (40, 80) and out.min() == 255


def test_open_mode_saves_snippet(tmp_path):
    opened = []
    img = np.full((100, 300, 3), 255, np.uint8)
    reply = handle({"action": "recognise", "image": data_url(img), "rect": None, "dpr": 1, "mode": "open"},
                   open_app=opened.append, out_dir=tmp_path)
    assert reply["ok"] and opened and opened[0].exists()
