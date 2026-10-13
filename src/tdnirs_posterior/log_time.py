"""Wall-clock timestamps with local UTC offset for console and JSON records."""
from datetime import datetime


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
