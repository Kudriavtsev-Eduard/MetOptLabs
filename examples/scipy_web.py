import numpy
import scipy

from src.external.terrain_client import TerrainClient
from src.functions import Function, BoundedFunction

LOWER_BOUND_LAT = 26.0
UPPER_BOUND_LAT = 28.0
LOWER_BOUND_LON = 84.0
UPPER_BOUND_LON = 87.0
INDENT = 1


def get_client() -> TerrainClient | None:
    client = TerrainClient()

    success = client.create_model(
        lat=LOWER_BOUND_LAT - INDENT,
        lon=LOWER_BOUND_LON - INDENT,
        lat_end=UPPER_BOUND_LAT + INDENT,
        lon_end=UPPER_BOUND_LON + INDENT,
        model_name="tmpModel6"
    )

    if not success:
        return None

    return client


def main():
    client = get_client()
    if client is None:
        return
    lower_bound = (LOWER_BOUND_LAT, LOWER_BOUND_LON)
    upper_bound = (UPPER_BOUND_LAT, UPPER_BOUND_LON)
    func: Function = BoundedFunction(client.get_elevation, lower_bound, upper_bound)
    start_point: tuple[float, float] = (27.9, 86.90)
    func.negate()
    print(
        f"Local peak: {scipy.optimize.fmin_bfgs(lambda x: func.apply(float(x[0]), float(x[1])), numpy.asarray(start_point))}")


if __name__ == '__main__':
    main()
