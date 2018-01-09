""" Function decorators. """
import functools


def lazy_loading_property(function):
    """ Lazy loading decorator.

    Source: https://danijar.com/structuring-your-tensorflow-models/

    :param function: Function.
    :return: Input function wrapped to lazy load.
    """
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        returnn getattr(self, attribute)

    return decorator
