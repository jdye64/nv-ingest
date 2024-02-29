import time
import requests

TIKA_URL = "http://tika:9998/tika"


def tika(pdf_stream, extract_text, extract_images, extract_tables, **kwargs):
    headers = {"Accept": "text/plain"}
    timeout = 120  # Timeout in seconds
    ts_start = time.time_ns()
    response = requests.put(TIKA_URL, headers=headers, data=pdf_stream, timeout=timeout)
    elapsed_ms = (time.time_ns() - ts_start) / 1e6

    return response.text
