import requests

TIKA_URL = "http://tika:9998/tika"


def tika(pdf_stream, extract_text, extract_images, extract_tables, **kwargs):
    headers = {"Accept": "text/plain"}
    timeout = 120  # Timeout in seconds
    response = requests.put(TIKA_URL, headers=headers, data=pdf_stream, timeout=timeout)

    return response.text
