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
    A decorator that checks if the first argument to the wrapped function (expected to be a ControlMessage object)
    contains any of the tasks specified in `required_tasks`. Each task can be a string of the task name or a tuple
    of the task name and task properties. If the message does not contain any listed task and/or task properties,
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
            if args and hasattr(args[0], "get_tasks"):
                message = args[0]
                tasks = message.get_tasks()
                for required_task in required_tasks:
                    if isinstance(required_task, str) and (required_task in tasks):
                        return func(*args, **kwargs)
                    if isinstance(required_task, tuple) or isinstance(required_task, list):
                        required_task, *required_task_props_list = required_task
                        if required_task not in tasks:
                            continue
                        task_props = tasks.get(required_task, [None])
                        if not task_props:
                            continue
                        task_props = task_props.pop()
                        for required_task_props in required_task_props_list:
                            if _is_subset(task_props, required_task_props):
                                return func(*args, **kwargs)
                if forward_func:
                    # If a forward function is provided, call it with the ControlMessage
                    return forward_func(message)
                else:
                    # If no forward function is provided, return the message directly
                    return message
            else:
                raise ValueError(
                    "The first argument must be a ControlMessage object with task handling " "capabilities."
                )

        return wrapper

    return decorator


def _is_subset(superset, subset):
    match subset:
        case str():
            return subset in superset
        case dict():
            return all(key in superset and _is_subset(val, superset[key]) for key, val in subset.items())
        case list() | set():
            return all(any(_is_subset(subitem, superitem) for superitem in superset) for subitem in subset)
        case _:
            return subset == superset
