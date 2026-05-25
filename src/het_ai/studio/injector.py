from functools import wraps
from typing import Any


class GhostInjector:
    """
    纯标注装饰器工厂。
    只负责把搜索空间元数据挂到函数上，不负责执行采样。
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
