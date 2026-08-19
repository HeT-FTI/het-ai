from functools import wraps
from typing import Any


class GhostInjector:
    """
    A pure annotation decorator factory.
    It only attaches the search space metadata to the function; it does not
    perform sampling.
    """

    @staticmethod
    def search(**search_space: Any):
        def decorator(func):
            func._is_tunable = True
            func._search_space = search_space

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper._is_tunable = True
            wrapper._search_space = search_space
            return wrapper

        return decorator
