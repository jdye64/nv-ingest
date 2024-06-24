# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import logging
from datetime import datetime
from functools import wraps

# Ensure the logging is configured; for example, to log to console at DEBUG level
logging.basicConfig(level=logging.DEBUG)


# Define ANSI color codes
class ColorCodes:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"  # Added Blue
    RESET = "\033[0m"


# Function to apply color to a message
def colorize(message, color_code):
    return f"{color_code}{message}{ColorCodes.RESET}"


def latency_logger(name=None):
    """
    A decorator to log the elapsed time of function execution. If available, it also logs
    the latency based on 'latency::ts_send' metadata in a ControlMessage object.

    Parameters
    ----------
    name : str, optional
        Custom name to use in the log message. Defaults to the function's name.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Ensure there's at least one argument and it has timestamp handling capabilities
            if args and hasattr(args[0], "get_timestamp"):
                message = args[0]
                start_time = datetime.now()

                result = func(*args, **kwargs)

                end_time = datetime.now()
                elapsed_time = end_time - start_time

                func_name = name if name else func.__name__

                # Log latency from ts_send if available
                if message.filter_timestamp("latency::ts_send"):
                    ts_send = message.get_timestamp("latency::ts_send")
                    latency_ms = (start_time - ts_send).total_seconds() * 1e3
                    logging.debug(f"{func_name} since ts_send: {latency_ms} msec.")

                message.set_timestamp("latency::ts_send", datetime.now())
                message.set_timestamp(f"latency::{func_name}::elapsed_time", elapsed_time)
                return result
            else:
                raise ValueError("The first argument must be a ControlMessage object with metadata " "capabilities.")

        return wrapper

    return decorator
