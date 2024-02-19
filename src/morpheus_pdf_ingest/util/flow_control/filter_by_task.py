# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps


def filter_by_task(required_tasks, forward_func=None):
    """
    A decorator that checks if the first argument to the wrapped function (expected to be a ControlMessage object)
    contains any of the tasks specified in `required_tasks`. If the message does not contain any listed task,
    the message is returned directly without calling the wrapped function, unless a forwarding function is provided,
    in which case it calls that function on the ControlMessage.

    Parameters
    ----------
    required_tasks : list
        A list of task keys to check for in the ControlMessage.
    forward_func : callable, optional
        A function to be called with the ControlMessage if no required task is found. Defaults to None.

    Returns
    -------
    callable
        The wrapped function, conditionally called based on the task check.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if args and hasattr(args[0], 'has_task'):
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
                raise ValueError("The first argument must be a ControlMessage object with task handling capabilities.")
        return wrapper

    return decorator
