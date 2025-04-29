import requests
import base64
import time
import urllib3
import json
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class TerrainClient:
    def __init__(self, server_url, timeout=30):

        self.server_url = server_url.rstrip('/')
        self.model_id = None
        self.timeout = timeout

    def _check_server(self):

        try:
            response = requests.get(
                f"{self.server_url}/",
                timeout=self.timeout,
                verify=False,
                headers={
                    'Connection': 'keep-alive',
                    'Accept': 'application/json'
                }
            )
            if response.status_code == 200:
                print(f"Сервер API доступен: {self.server_url}")
                models = response.json().get("models_available", [])
                if models:
                    print(f"Доступные модели: {', '.join(models)}")
                else:
                    print("На сервере нет доступных моделей. Необходимо создать новую.")
                return True
            else:
                print(f"Сервер вернул код ответа: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при проверке сервера: {e}")
            return False

    def create_model(self, lat, lon, lat_end=None, lon_end=None, model_name=None, retries=3):

        if not self._check_server():
            print("Ошибка: сервер API недоступен")
            return False

        if model_name is None:
            lat_dir = 'N' if lat >= 0 else 'S'
            lon_dir = 'E' if lon >= 0 else 'W'
            lat_str = f"{abs(int(lat)):02d}"
            lon_str = f"{abs(int(lon)):03d}"
            model_name = f"{lat_dir}{lat_str}{lon_dir}{lon_str}"

        payload = {
            "lat": lat,
            "lon": lon,
            "lat_end": lat_end,
            "lon_end": lon_end,
            "model_name": model_name
        }

        try:
            print(f"Попытка / создания модели...")
            print(f"Отправляем запрос: {self.server_url}/api/create_model")
            print(f"Данные запроса: {json.dumps(payload, indent=2)}")

            response = requests.post(
                f"{self.server_url}/api/create_model",
                json=payload,
                timeout=self.timeout
            )

            print(f"Код ответа: {response.status_code}")
            print(f"Ответ сервера: {response.text[:200]}...")

            response.raise_for_status()

            result = response.json()
            self.model_id = result.get("model_id", model_name)
            print(f"Модель успешно создана: {self.model_id}")
            return True
        except requests.exceptions.Timeout:
            print(f"Время ожидания истекло.")

        except Exception as e:
            print(f"Ошибка создания модели: {e}")

        print("Все попытки создания модели завершились неудачно")
        return False

    def set_model_id(self, model_id):
        self.model_id = model_id
        print(f"Установлена модель: {self.model_id}")
        return True

    def get_elevation(self, lat, lon, retries=3):
        self.model_id = "Caucasus"

        params = {
            "model_id": self.model_id,
            "lat": lat,
            "lon": lon
        }

        for attempt in range(retries):
            try:
                response = requests.get(
                    f"{self.server_url}/api/get_elevation",
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["elevation"]
            except requests.exceptions.Timeout:
                print(f"Таймаут при получении высоты. Попытка {attempt + 1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(2)
            except Exception as e:
                print(f"Ошибка получения высоты: {e}")
                if attempt < retries - 1:
                    time.sleep(2)

        print("Все попытки получения высоты завершились неудачно")
        return None

    def get_elevation_by_xy(self, x, y, method='linear', retries=3):
        self.model_id = "Caucasus"

        params = {
            "model_id": self.model_id,
            "x": x,
            "y": y,
            "method": method
        }

        for attempt in range(retries):
            try:
                response = requests.get(
                    f"{self.server_url}/api/get_elevation_by_xy",
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["elevation"]
            except requests.exceptions.Timeout:
                print(f"Таймаут при получении высоты по x,y. Попытка {attempt + 1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(2)
            except Exception as e:
                print(f"Ошибка получения высоты по x,y: {e}")
                if attempt < retries - 1:
                    time.sleep(2)

        print("Все попытки получения высоты по x,y завершились неудачно")
        return None

    def save_visualization(self, title="Рельеф", save_path="visualization.png", retries=2):
        if not self.model_id:
            print("Ошибка: сначала нужно создать модель или установить ID модели")
            return False

        params = {
            "model_id": self.model_id,
            "title": title
        }

        try:
            print(f"Получение визуализации рельефа)...")
            response = requests.get(
                f"{self.server_url}/api/visualize",
                params=params,
                timeout=60
            )
            response.raise_for_status()

            img_data = base64.b64decode(response.json()["image"])

            with open(save_path, 'wb') as f:
                f.write(img_data)

            print(f"Изображение сохранено в {save_path}")
            return True
        except requests.exceptions.Timeout:
            print(f"Таймаут при получении визуализации./{retries}")

        except Exception as e:
            print(f"Ошибка получения визуализации: {e}")

        print("Все попытки получения визуализации завершились неудачно")
        return False


if __name__ == "__main__":
    client = TerrainClient("http://185.175.45.249:5000", timeout=30)

    if not client._check_server():
        print("Сервер API недоступен. Завершение работы.")
        exit(1)

    success = client.create_model(
        lat=43.0,
        lon=42.0,
        lat_end=44.0,
        lon_end=43.0,
        model_name="Caucasus"
    )

    if not success:
        print("Не удалось создать модель. Завершение работы.")
        exit(1)

    elevation = client.get_elevation(43.35, 42.45)
    if elevation is not None:
        print(f"Высота Эльбруса: {elevation} метров")
