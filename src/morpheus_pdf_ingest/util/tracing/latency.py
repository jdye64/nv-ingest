import logging
import time
from functools import wraps

# Ensure the logging is configured; for example, to log to console at DEBUG level
logging.basicConfig(level=logging.DEBUG)


# Define ANSI color codes
class ColorCodes:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'  # Added Blue
    RESET = '\033[0m'


# Function to apply color to a message
def colorize(message, color_code):
    return f"{color_code}{message}{ColorCodes.RESET}"


def latency_logger(name=None):
    """
    A decorator to log the elapsed time of function execution and,
    if available, the latency based on 'latency::ts_send' metadata in a ControlMessage object.

    Parameters
    ----------
    name : str, optional
        Custom name to use in the log message. Defaults to the function's name.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Ensure there's at least one argument and it has the required methods
            if args and hasattr(args[0], 'has_metadata') and hasattr(args[0], 'set_metadata'):
                message = args[0]
                start_time = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

                result = func(*args, **kwargs)

                end_time = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                elapsed_time = (end_time - start_time) / 1e6  # Convert ns to ms

                log_name = name if name else func.__name__

                if message.has_metadata('latency::ts_send'):
                    ts_send = int(message.get_metadata('latency::ts_send'))
                    latency = (start_time - ts_send) / 1e6  # Also in ms
                    logging.debug(colorize(f"{log_name} since ts_send: {latency} msec.", ColorCodes.BLUE))

                logging.debug(colorize(f"{log_name} elapsed time {elapsed_time} msec.", ColorCodes.BLUE))

                message.set_metadata('latency::ts_send', str(time.clock_gettime_ns(time.CLOCK_MONOTONIC)))
                return result
            else:
                raise ValueError("The first argument must be a ControlMessage object with metadata capabilities.")

        return wrapper

    return decorator
