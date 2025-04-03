import numpy
from scipy.optimize import *

from src.functions import Function, NoiseFunction
from src.report import Report


def func(args, *_):
    x, y = args
    return 4 * x ** 2 + y ** 2 + 3 * y


def grad(args, *_):
    x, y = args
    dx = 8 * x
    dy = 2 * y + 3
    return numpy.asarray((dx, dy))


def to_float_array(a) -> list[tuple[float, ...]]:
    result = []
    for elem in a:
        result.append(tuple(float(num) for num in elem))
    return result


def main():
    x_0 = numpy.asarray((10, 10))
    result = fmin_cg(func, x_0, fprime=grad, retall=True)
    scipy_report = Report(Function(lambda x, y: func([x, y])),
                          to_float_array(result[1]), False,
                          dict(),
                          "Scipy")
    scipy_report.display()


if __name__ == '__main__':
    main()
