import copy
import random
from abc import ABC
from typing import Callable, Sequence, override
import src.utilities as utilities

"""
functions.py
Standard implementation of function classes.

Example of usage:
    my_function = DerivableFunction(lambda x, y: x ** 2 + y ** 2, (lambda x, y: 2 * x, lambda x, y: 2 * y))
    my_function.apply(1, 2) -> 5
    my_function.get_gradient_at(1, 2) -> (2, 4)
"""


class Function:
    def __init__(self, function: Callable[..., float]):
        self.function = function
        self.tracking = False
        self.times_used = 0

    def apply(self, *args: float) -> float:
        if self.tracking:
            self.times_used += 1
        return self.function(*args)

    def get_arg_count(self) -> int:
        return self.function.__code__.co_argcount

    def start_tracking(self) -> None:
        self.tracking = True

    def stop_tracking(self) -> None:
        self.tracking = False

    def get_call_data(self) -> dict[str, int]:
        return {"to_function": self.times_used}


class DirectionalFunction:
    def __init__(self, function: Function, starting_point: tuple[float, ...], direction: tuple[float, ...]):
        self.function = function
        self.starting_point = starting_point
        self.direction = direction

    def apply(self, coefficient: float) -> float:
        return self.function.apply(*utilities.element_wise_addition(self.starting_point, self.direction, -coefficient))


class HyperFunction(Function):
    def __init__(self, function: Callable[[tuple[float, ...], float, ...], float]):
        super().__init__(function)
        self.object = None
        self.property = None

    def set_object(self, obj: tuple[float, ...], prop: float):
        self.object = obj
        self.property = prop

    def apply(self, *args: float) -> float:
        if self.tracking:
            self.times_used += 1
        return self.function(self.object, self.property, *args)

    @override
    def get_arg_count(self):
        return super().get_arg_count() - 1


class DerivableFunction(Function):
    def __init__(self, function: Callable[..., float], gradient: tuple[Callable[..., float], ...]):
        super().__init__(function)
        self._gradient = gradient
        self.times_gradient_used = False

    def get_gradient_at(self, *args: float) -> tuple[float, ...]:
        if self.tracking:
            self.times_gradient_used += 1
        return tuple(dF(*args) for dF in self._gradient)

    def get_call_data(self) -> dict[str, int]:
        result = super().get_call_data()
        result["to_gradient"] = self.times_gradient_used
        return result

    def get_directional(self, point: tuple[float, ...]) -> DirectionalFunction:
        return DirectionalFunction(self, point, self.get_gradient_at(*point))

class AutomatedDerivableFunction(DerivableFunction):
    @staticmethod
    def _get_partial(function: Function, x: tuple[float, ...], coord: int, epsilon: float):
        x_shift = x[:coord] + (x[coord] + epsilon,) + x[coord + 1:]
        return (function.apply(*x_shift) - function.apply(*x)) / epsilon

    def __init__(self, function: Function, derivable_start: bool = True, epsilon: float = 10 ** -8):
        super().__init__(function.apply,
                         tuple(
                             lambda *x: AutomatedDerivableFunction._get_partial(function, x, i, epsilon)
                             for i in range(function.get_arg_count()))
                         if derivable_start else ())
        self.__arg_count = function.get_arg_count()

    def get_arg_count(self) -> int:
        return self.__arg_count


class BatchAutomatedDerivableFunction(AutomatedDerivableFunction):
    def __init__(self, function: HyperFunction, objects: Sequence[tuple[tuple[float, ...], float]],
                 batch_size: int, regular_func: DerivableFunction, epsilon: float = 10 ** -8):
        super().__init__(function, False)
        self.objects = objects
        self.function = function
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.batch_choices = list(range(len(objects)))
        self.regular_func = regular_func
        self.__new_batch()

    def get_batch_gradient_at(self, object_numbers: list[int], hyper_parameters: tuple[float, ...]) -> (
            tuple)[float, ...]:
        if self.tracking:
            self.times_gradient_used += 1
        gradients = []
        for i in object_numbers:
            self.function.set_object(self.objects[i][0], self.objects[i][1])
            gradient_for_object = tuple(
                AutomatedDerivableFunction._get_partial(self.function, hyper_parameters, j, self.epsilon)
                for j in range(len(hyper_parameters)))
            gradients.append(gradient_for_object)
        gradients.append(self.regular_func.get_gradient_at(*hyper_parameters))
        return tuple(sum(elements) for elements in zip(*gradients))

    def __new_batch(self):
        self.batch_choice = random.sample(self.batch_choices, self.batch_size)

    @override
    def apply(self, *args: float) -> float:
        self.__new_batch()
        return self.__apply_batch(self.batch_choice, args)

    def __apply_batch(self, batch_numbers: list[int], point: tuple[float, ...]) -> float:
        result = 0
        self.times_used += 1
        for batch_num in batch_numbers:
            self.function.set_object(self.objects[batch_num][0], self.objects[batch_num][1])
            result += self.function.apply(*point)
        return result / max(len(batch_numbers), 1) + self.regular_func.apply(*point)

    @override
    def get_gradient_at(self, *args: float) -> tuple[float, ...]:
        self.__new_batch()
        eg = self.get_batch_gradient_at(self.batch_choice, args)
        rg = self.regular_func.get_gradient_at(*args)
        return tuple(sum(elements) for elements in zip(eg, rg))

    @override
    def get_directional(self, point: tuple[float, ...]) -> DirectionalFunction:
        to_result = Function(lambda *x: self.__apply_batch(copy.deepcopy(self.batch_choice), x))
        grad = self.get_batch_gradient_at(self.batch_choice, point)
        return DirectionalFunction(to_result, point, grad)


class L(DerivableFunction, ABC):
    def __init__(self, arg_count: int, lamda: float, function: Callable[..., float],
                 gradient: tuple[Callable[..., float], ...]):
        super().__init__(function, gradient)
        self.arg_count = arg_count
        self.lamda = lamda

    @override
    def get_arg_count(self) -> int:
        return self.arg_count


class L2(L):
    def __init__(self, arg_count: int, lamda: float):
        super().__init__(arg_count, lamda, (lambda *args: (lamda * sum(map((lambda x: x ** 2), args[1:]))) / 2),
                         tuple((lambda *w: lamda * w[i]) for i in range(arg_count)))


class L1(L):
    def __init__(self, arg_count: int, lamda: float):
        super().__init__(arg_count, lamda, (lambda *args: (lamda * sum(map((lambda x: abs(x)), args[1:]))) / 2),
                         tuple((lambda *w: lamda * L1.sign(w[i])) for i in range(arg_count)))

    @staticmethod
    def sign(a: int) -> int:
        if a < 0:
            return -1
        if a == 0:
            return 0
        return 1

class Elastic(L):
    def __init__(self, arg_count: int, lamda: float):
        super().__init__(arg_count, lamda, (lambda *args: (lamda * sum(map((lambda x: x ** 2 + abs(x)), args[1:]))) / 2),
                         tuple((lambda *w: lamda * w[i] * L1.sign(w[i])) for i in range(arg_count)))


class NoiseFunction(Function):
    def __init__(self, function: Callable[..., float], creativity: int = 20):
        assert creativity > 0
        super().__init__(function)
        self.cache: dict[tuple[float, ...], float] = dict()
        self.creativity = creativity

    def apply(self, *args: float) -> float:
        result = super().apply(*args)
        if args in self.cache:
            offset = self.cache[args]
        else:
            offset = (random.randint(-self.creativity, self.creativity) + random.random())
            self.cache[args] = offset
        return result + offset
