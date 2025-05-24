import copy
import random
import typing
from typing import Callable, override
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
        assert self.get_arg_count() == len(args)
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


class DerivableFunction(Function):
    def __init__(self, function: Callable[..., float], gradient: tuple[Callable[..., float], ...]):
        super().__init__(function)
        self.gradient = gradient
        self.times_gradient_used = False

    def get_gradient_at(self, *args: float) -> tuple[float, ...]:
        if self.tracking:
            self.times_gradient_used += 1
        return tuple(dF(*args) for dF in self.gradient)

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
    def __init__(self, function: HyperFunction, objects: typing.Sequence[tuple[tuple[float, ...], float]],
                 batch_size: int, epsilon: float = 10 ** -8):
        super().__init__(function, False)
        self.objects = objects
        self.function = function
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.batch_choices = list(range(len(objects)))
        self.batch_choice = random.sample(self.batch_choices, batch_size)

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

        return tuple(sum(elements) for elements in zip(*gradients))

    def __new_batch(self):
        self.batch_choice = random.sample(self.batch_choices, self.batch_size)

    @override
    def apply(self, *args: float) -> float:
        self.__new_batch()
        return self.__apply_batch(self.batch_choice, args)

    def __apply_batch(self, batch_numbers: list[int], point: tuple[float, ...]) -> float:
        result = 0
        for batch_num in batch_numbers:
            self.function.set_object(self.objects[batch_num][0], self.objects[batch_num][1])
            result += self.function.apply(*point)
        return result

    @override
    def get_gradient_at(self, *args: float) -> tuple[float, ...]:
        self.__new_batch()
        return self.get_batch_gradient_at(self.batch_choice, args)

    @override
    def get_directional(self, point: tuple[float, ...]) -> DirectionalFunction:
        to_result = Function(lambda *x: self.__apply_batch(copy.deepcopy(self.batch_choice), x))
        grad = self.get_batch_gradient_at(self.batch_choice, point)
        return DirectionalFunction(to_result, point, grad)


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
