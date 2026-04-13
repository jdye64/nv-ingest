from corpus_suite_poc import __version__


def health_payload() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
