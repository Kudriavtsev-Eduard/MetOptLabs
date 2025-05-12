import sys
import typing
import requests
import base64
import time


class TerrainClient:
    DEFAULT_IP = "http://185.175.45.249:5000"

    def __init__(self, server_url: str = DEFAULT_IP, timeout: int = 60, retries: int = 3) -> None:
        self.server_url = server_url.rstrip('/')
        self.model_id = None
        self.timeout = timeout
        self.retries = retries

    def create_model(self, lat: float, lon: float, lat_end: float | None = None, lon_end: float | None = None,
                     model_name: str = None) -> bool:
        if not self.__check_server():
            return False
        print("Creating model...")
        if model_name is None:
            model_name = self.__get_default_model_name(lat, lon)
        payload = {
            "lat": lat,
            "lon": lon,
            "lat_end": lat_end,
            "lon_end": lon_end,
            "model_name": model_name
        }
        response = self.__get_request_response("create_model", {'json': payload}, is_post=True)
        if response is None:
            return False
        print(f"Response: {response.text}...")
        result = response.json()
        self.model_id = result.get("model_id", model_name)
        print(f"Model created: {self.model_id}")
        return True

    def set_model_id(self, model_id: str) -> None:
        self.model_id = model_id
        print(f"Model set: {self.model_id}")

    def get_elevation(self, lat: float, lon: float) -> float | None:
        params = {
            "model_id": self.model_id,
            "lat": lat,
            "lon": lon
        }
        return self.__get_elevation_by_params("get_elevation", params)

    def get_elevation_by_xy(self, x: float, y: float, method: str = 'linear') -> float | None:
        params = {
            "model_id": self.model_id,
            "x": x,
            "y": y,
            "method": method
        }
        return self.__get_elevation_by_params("get_elevation_by_xy", params)

    def save_visualization(self, title="Рельеф", save_path="visualization.png"):
        if self.model_id is None:
            print("Error: Model not set", file=sys.stderr)
            return False

        params = {
            "model_id": self.model_id,
            "title": title
        }

        print(f"Getting visualization...")
        response = self.__get_request_response("visualize", params)
        if response is None:
            return False

        img_data = base64.b64decode(response.json()["image"])

        with open(save_path, 'wb') as f:
            f.write(img_data)

        print(f"Image saved to {save_path}")
        return True

    def __get_request_response(self, api_folder: str | None, params: dict[str, typing.Any],
                               is_post: bool = False) -> requests.models.Response | None:
        arguments = {'url': self.server_url + ('/api/' + api_folder if api_folder is not None else ''),
                     'timeout': self.timeout,
                     **params}
        for attempt in range(self.retries):
            try:
                response = requests.post(**arguments) if is_post else requests.get(**arguments)
                response.raise_for_status()
                return response
            except requests.exceptions.Timeout:
                print(f"Get request timeout, attempts {attempt + 1}/{self.retries}", file=sys.stderr)
            except requests.exceptions.HTTPError as e:
                print(f"Error while getting elevation: {e}", file=sys.stderr)
            if attempt < self.retries - 1:
                time.sleep(2)
        return None

    def __check_server(self) -> bool:
        response = self.__get_request_response(None, {
            'verify': False,
            'headers': {
                'Connection': 'keep-alive',
                'Accept': 'application/json'
            }})
        if response is None:
            return False
        print(f"API Server available: {self.server_url}")
        models = response.json().get("models_available", tuple())
        if models:
            print(f"Available models: {', '.join(models)}")
        else:
            print("No available models at the time. One must be created.")
        return True

    @staticmethod
    def __get_default_model_name(lat: float, lon: float) -> str:
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        lat_str = f"{abs(int(lat)):02d}"
        lon_str = f"{abs(int(lon)):03d}"
        return f"{lat_dir}{lat_str}{lon_dir}{lon_str}"

    def __get_elevation_by_params(self, url: str, params: dict[str, typing.Any]) -> float | None:
        if self.model_id is None:
            print("Error. Model not set")
            return None
        response = self.__get_request_response(url, {"params": params})
        if response is not None:
            return response.json().get("elevation", None)
        return None
