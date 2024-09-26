import typing as t

class IthElement:
    """
    Makes a callable which returns the ith element of the input data.
    Used to avoid making lambdas, which are not pickleable.
    """
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __call__(self, data: t.Iterable[float]) -> float:
        return data[self.idx]
    

class FunctionComposition:
    """
    Makes a callable which is the composition of functions.
    Again used to avoid making lambdas.
    Note that the functions are applied in reverse order, i.e. the last function
    in the list is applied first. This is for consistency with the way
    we write function composition, e.g. f(g(x)).
    """
    def __init__(self, callables: list[t.Callable]) -> None:
        self.callables = callables

    def __call__(self, x: t.Any) -> t.Any:
        for func in reversed(self.callables):
            x = func(x)
        return x