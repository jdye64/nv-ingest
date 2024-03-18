# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from functools import wraps


def filter_by_task(required_tasks, forward_func=None):
    """
    A decorator that checks if the first argument to the wrapped function (expected to be a
    ControlMessage object)
    contains any of the tasks specified in `required_tasks`. If the message does not contain any
    listed task,
    the message is returned directly without calling the wrapped function, unless a forwarding
    function is provided, in which case it calls that function on the ControlMessage.

    Parameters
    ----------
    required_tasks : list
        A list of task keys to check for in the ControlMessage.
    forward_func : callable, optional
        A function to be called with the ControlMessage if no required task is found. Defaults to
        None.

    Returns
    -------
    callable
        The wrapped function, conditionally called based on the task check.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if args and hasattr(args[0], "has_task"):
                message = args[0]
                if any(message.has_task(task) for task in required_tasks):
                    return func(*args, **kwargs)
                elif forward_func:
                    # If a forward function is provided, call it with the ControlMessage
                    forward_func(message)
                else:
                    # If no forward function is provided, return the message directly
                    return message
            else:
                raise ValueError(
                    "The first argument must be a ControlMessage object with task handling "
                    "capabilities."
                )

        return wrapper

    return decorator
