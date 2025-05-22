import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import text, create_engine
import random
import math
from datetime import datetime, timedelta
import sys
import os
import warnings

# Подавляем предупреждения для более чистого вывода
warnings.filterwarnings('ignore')

# Константы
FIGURE_SIZE_STANDARD = (12, 8)
FIGURE_SIZE_SQUARE = (10, 10)

# Путь к директории проекта
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

# Импортируем функции из graphics_black.py
from sql.graphics_black import set_plotting_style, configure_plot_for_cyrillic

# Настройки подключения к базе данных
# ВНИМАНИЕ: Замените эти значения на реальные для вашей БД
DB_USER = "postgres"
DB_PASSWORD = "0252"
DB_NAME = "ru_postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

# Строка подключения к БД
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Создаем подключение к БД
try:
    engine = create_engine(DB_URL)
    SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    print(f"Успешно подключились к базе данных {DB_NAME}")
except Exception as e:
    print(f"Ошибка подключения к базе данных: {str(e)}")
    sys.exit(1)


# Определяем функцию загрузки данных (аналог load_data из graphics_black.py)
def load_data(table_name='flight_data_for_visualization', sample_size=10):
    """
    Загружает данные о рейсах из базы данных PostgreSQL
    """
    print(f"Загрузка данных из таблицы {table_name}...")

    session = SessionFactory()
    query = f"""SELECT * 
                FROM public.{table_name} 
                ORDER BY md5(random()::text) 
                LIMIT {sample_size}
            """

    try:
        # Выполняем запрос напрямую через SQLAlchemy
        result = session.execute(text(query))

        # Получаем имена столбцов
        columns = result.keys()

        # Преобразуем результат в список словарей
        data = [dict(zip(columns, row)) for row in result.fetchall()]

        print(f"Загружено {len(data)} строк и {len(columns)} столбцов.")
        return data
    except Exception as e:
        print(f"Ошибка при загрузке данных из БД: {str(e)}")
        return None
    finally:
        session.close()


# Определяем функцию предобработки данных (аналог preprocess_data из graphics_black.py)
def preprocess_data(df):
    """
    Выполняет базовую предобработку данных
    """
    print("Выполняется предобработка данных...")

    # Преобразование текстовых дат в datetime
    if 'FlightDate' in df.columns:
        df['FlightDate'] = pd.to_datetime(df['FlightDate'])

    # Обработка числовых столбцов
    numeric_columns = ['Year', 'DayofMonth', 'Distance', 'Seats', 'AirTime',
                       'DepDelay', 'ArrDelay', 'CarrierDelay', 'WeatherDelay',
                       'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
                       'TaxiIn', 'TaxiOut', 'ActualElapsedTime', 'CRSElapsedTime']

    for col in numeric_columns:
        if col in df.columns:
            print(f"Преобразование числового столбца: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Заполнение пропусков для числовых столбцов
            if df[col].isna().any():
                if col in ['DepDelay', 'ArrDelay', 'CarrierDelay', 'WeatherDelay',
                           'NASDelay', 'SecurityDelay', 'LateAircraftDelay']:
                    # Для задержек заполняем нулями
                    df[col] = df[col].fillna(0)
                else:
                    # Для других числовых столбцов заполняем медианой
                    df[col] = df[col].fillna(df[col].median())

    # Обработка временных полей (превращаем в минуты от полуночи)
    time_columns = ['CRSDepTime', 'CRSArrTime']
    for col in time_columns:
        if col in df.columns:
            print(f"Преобразование времени в поле {col}")

            # Функция для преобразования времени в минуты от полуночи
            def time_to_minutes(time_str):
                try:
                    if pd.isna(time_str):
                        return None

                    # Если уже число, возвращаем как есть
                    if isinstance(time_str, (int, float)):
                        return float(time_str)

                    # Разделяем часы и минуты
                    parts = str(time_str).strip().split(':')
                    if len(parts) == 2:
                        hours, minutes = int(parts[0]), int(parts[1])
                        return float(hours * 60 + minutes)
                    else:
                        return None
                except Exception as e:
                    print(f"Ошибка при обработке времени '{time_str}': {e}")
                    return None

            # Применяем функцию к столбцу и заменяем пропуски медианой
            df[col] = df[col].apply(time_to_minutes)

            # Проверяем, есть ли пропуски
            if df[col].isna().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"  - Заполнено {df[col].isna().sum()} пропусков медианой {median_value}")

    # Создаем категориальные переменные для месяца и дня недели
    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype('category')

    if 'DayOfWeek' in df.columns:
        df['DayOfWeek'] = df['DayOfWeek'].astype('category')

    print("Предобработка данных завершена.")
    print(f"Типы данных после предобработки:")
    print(df.dtypes.head(10))  # Показать первые 10 типов данных для диагностики

    return df


class EnhancedAdaptivePredictor:
    def __init__(self, learning_rate=0.01, spatial_weight=0.3, temporal_weight=0.4, seasonal_weight=0.3,
                 smoothing_factor=0.3):
        self.learning_rate = learning_rate
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.seasonal_weight = seasonal_weight
        self.historical_predictions = {}
        self.last_predictions = {}  # Для экспоненциального сглаживания
        self.smoothing_factor = smoothing_factor  # Фактор сглаживания
        self.prediction_cache = {}  # Кеш прогнозов
        self.cache_expiry = {}  # Время истечения кеша

        self.airline_factors = {
            # Примерные факторы для некоторых авиакомпаний (можно расширить)
            'SU': 0.9,  # Аэрофлот
            'S7': 0.95,  # S7
            'U6': 1.2,  # Уральские авиалинии
            'DP': 1.1,  # Победа
            'FV': 1.05,  # Россия
            # Для других авиакомпаний будет использовано значение по умолчанию
        }
        self.default_airline_factor = 1.0

    def get_seasonal_factor(self, date: datetime):
        """Рассчитывает сезонный фактор на основе даты с плавными переходами."""
        # День недели (0 - понедельник, 6 - воскресенье)
        weekday = date.weekday()

        # Месяц (1-12)
        month = date.month

        # Факторы по дням недели с плавным переходом
        weekday_factors = {
            0: 1.0,  # Понедельник
            1: 0.98,  # Вторник (был 0.95)
            2: 0.95,  # Среда (был 0.9)
            3: 0.98,  # Четверг (был 0.95)
            4: 1.05,  # Пятница (был 1.2)
            5: 1.15,  # Суббота (был 1.3)
            6: 1.1  # Воскресенье (был 1.25)
        }

        # Сезонные факторы по месяцам с плавным переходом
        seasonal_factors = {
            1: 1.05,  # Январь (был 1.1)
            2: 0.95,  # Февраль (был 0.9)
            3: 0.97,  # Март (был 0.95)
            4: 1.0,  # Апрель
            5: 1.03,  # Май (был 1.05)
            6: 1.08,  # Июнь (был 1.2)
            7: 1.15,  # Июль (был 1.3)
            8: 1.12,  # Август (был 1.25)
            9: 1.05,  # Сентябрь (был 1.0)
            10: 0.97,  # Октябрь (был 0.95)
            11: 1.0,  # Ноябрь
            12: 1.08  # Декабрь (был 1.2)
        }

        # Комбинированный сезонный фактор
        return weekday_factors[weekday] * seasonal_factors[month]

    def get_geographic_factors(self, db: Session, origin: str, destination: str):
        """
        Get geographic location data for airports and calculate related factors
        that might affect flight delays (weather patterns, airport congestion, etc.)
        """
        try:
            # Query to get geographic coordinates for both airports
            geo_query = """
                        SELECT o.airport_code as origin_code,
                               o.latitude     as origin_lat,
                               o.longitude    as origin_long,
                               d.airport_code as dest_code,
                               d.latitude     as dest_lat,
                               d.longitude    as dest_long
                        FROM airports o
                                 JOIN airports d ON d.airport_code = :destination
                        WHERE o.airport_code = :origin \
                        """

            result = db.execute(text(geo_query),
                                {'origin': origin, 'destination': destination}).first()

            if not result:
                print(f"No geographic data found for {origin}-{destination}")
                return {
                    'weather_impact': 0.8,  # Default modest impact
                    'distance_factor': 1.0,
                    'airport_congestion': 1.0
                }

            def haversine(lon1, lat1, lon2, lat2):
                # Convert decimal degrees to radians
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                # Haversine formula
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * asin(sqrt(a))
                r = 6371  # Radius of earth in kilometers
                return c * r

            distance = haversine(result.origin_long, result.origin_lat,
                                 result.dest_long, result.dest_lat)

            # Calculate weather impact based on geographic location and season
            # This would be more sophisticated with actual weather data
            weather_impact = 1.0

            # Higher latitudes have more seasonal weather variation
            lat_factor = max(abs(result.origin_lat), abs(result.dest_lat)) / 90.0
            weather_impact += 0.5 * lat_factor

            # Airport congestion based on traffic estimates
            # This would be more accurate with actual airport size/traffic data
            airport_congestion = 1.0

            return {
                'weather_impact': weather_impact,
                'distance_factor': distance / 1000,  # Normalize to a reasonable scale
                'airport_congestion': airport_congestion
            }

        except Exception as e:
            db.rollback()
            print(f"Error getting geographic factors: {str(e)}")
            return {
                'weather_impact': 1.0,
                'distance_factor': 1.0,
                'airport_congestion': 1.0
            }

    def spatial_temporal_features(self, db: Session, origin: str, destination: str, date: datetime,
                                  airline: str = None):
        # Проверка наличия данных в таблицах
        check_data_query = """
                           SELECT (SELECT COUNT(*) FROM flights) as flights_count,
                                  (SELECT COUNT(*) FROM delay)   as delays_count \
                           """

        data_check = db.execute(text(check_data_query)).first()
        print(f"Data check: Flights: {data_check.flights_count}, Delays: {data_check.delays_count}")

        # Базовый запрос для маршрута
        route_query = """
                      SELECT COUNT(*) as route_count
                      FROM flights
                      WHERE origin_airport = :origin
                        AND dest_airport = :destination \
                      """

        route_check = db.execute(text(route_query),
                                 {'origin': origin, 'destination': destination}).first()
        print(f"Route check: {route_check.route_count} flights found for {origin}-{destination}")

        # Учёт авиакомпании в запросах, если она указана
        airline_filter = ""
        if airline:
            airline_filter = "AND f.airline_code = :airline"

        # Запрос для пространственных характеристик
        spatial_query = f"""
        SELECT
            COALESCE(AVG(NULLIF(distance, 0)), 500) as avg_distance,
            COUNT(*) as route_frequency,
            MIN(distance) as min_distance,
            MAX(distance) as max_distance
        FROM flights f
        WHERE origin_airport = :origin
        AND dest_airport = :destination
        {airline_filter}
        """

        # Запрос для временных характеристик с учетом дня недели и месяца
        temporal_query = f"""
        WITH flight_stats AS (
            SELECT
                f.id,
                f.fl_date,
                EXTRACT(DOW FROM f.fl_date) as day_of_week,
                EXTRACT(MONTH FROM f.fl_date) as month,
                COALESCE(d.dep_delay, 0) as dep_delay,
                COALESCE(d.cancelled, 0) as cancelled
            FROM flights f
            LEFT JOIN delay d ON f.id = d.id
            WHERE f.origin_airport = :origin
            AND f.dest_airport = :destination
            {airline_filter}
            AND f.fl_date >= (:date)::timestamp - INTERVAL '365 days'
            AND f.fl_date <= (:date)::timestamp
        )
        SELECT
            COUNT(*) as total_flights,
            COALESCE(AVG(CASE WHEN dep_delay > 0 THEN dep_delay ELSE 0 END), 20) as avg_delay,
            COALESCE(MAX(dep_delay), 60) as max_delay,
            COALESCE(AVG(CASE WHEN cancelled = 1 THEN 1 ELSE 0 END), 0.05) as cancellation_rate,
            COUNT(DISTINCT DATE_TRUNC('day', fl_date)) as unique_days,
            -- Группировка по дню недели
            COALESCE(AVG(CASE WHEN day_of_week = :target_dow THEN dep_delay ELSE NULL END),
                    AVG(CASE WHEN dep_delay > 0 THEN dep_delay ELSE 0 END)) as dow_avg_delay,
            -- Группировка по месяцу
            COALESCE(AVG(CASE WHEN month = :target_month THEN dep_delay ELSE NULL END),
                    AVG(CASE WHEN dep_delay > 0 THEN dep_delay ELSE 0 END)) as month_avg_delay
        FROM flight_stats
        """

        # Параметры запроса
        query_params = {
            'origin': origin,
            'destination': destination,
            'date': date.strftime('%Y-%m-%d'),
            'target_dow': date.weekday(),
            'target_month': date.month
        }

        if airline:
            query_params['airline'] = airline

        try:
            spatial_result = db.execute(text(spatial_query), query_params).first()
            if spatial_result:
                print("Detailed spatial data:")
                try:
                    # Alternative approach using column access by name
                    spatial_dict = {}
                    # Extract column names from the result
                    if hasattr(spatial_result, '_fields'):
                        column_names = spatial_result._fields
                    else:
                        column_names = ['avg_distance', 'route_frequency', 'min_distance', 'max_distance']

                    # Build dictionary from column names and values
                    for column in column_names:
                        if hasattr(spatial_result, column):
                            spatial_dict[column] = getattr(spatial_result, column)

                    for key, value in spatial_dict.items():
                        print(f"  {key}: {value}")
                except Exception as e:
                    print(f"Error converting spatial data to dictionary: {e}")

            temporal_result = db.execute(text(temporal_query), query_params).first()
            if temporal_result:
                print("Detailed temporal data:")  # Changed from "spatial" to "temporal"
                try:
                    # Alternative approach using column access by name
                    temporal_dict = {}
                    # Extract column names from the result
                    if hasattr(temporal_result, '_fields'):
                        column_names = temporal_result._fields
                    else:
                        column_names = ['total_flights', 'avg_delay', 'max_delay', 'cancellation_rate',
                                        'unique_days', 'dow_avg_delay', 'month_avg_delay']

                    # Build dictionary from column names and values
                    for column in column_names:
                        if hasattr(temporal_result, column):
                            temporal_dict[column] = getattr(temporal_result, column)

                    for key, value in temporal_dict.items():
                        print(f"  {key}: {value}")
                except Exception as e:
                    print(f"Error converting temporal data to dictionary: {e}")

            # Если нет реальных данных, используем более реалистичные базовые предсказания
            if not spatial_result or not temporal_result:
                # Генерируем немного случайности для базовых предсказаний, но с меньшим разбросом
                random_factor = random.uniform(0.9, 1.1)  # Уменьшенный разброс

                return {
                    'spatial': {
                        'avg_distance': 500 * random_factor,
                        'route_frequency': max(5, int(10 * random_factor)),
                        'min_distance': 300,
                        'max_distance': 700
                    },
                    'temporal': {
                        'avg_delay': max(10, 20 * random_factor),
                        'max_delay': max(45, 60 * random_factor),
                        'total_flights': max(5, int(10 * random_factor)),
                        'cancellation_rate': min(0.1, 0.05 * random_factor),
                        'dow_avg_delay': max(12, 22 * random_factor),
                        'month_avg_delay': max(15, 25 * random_factor)
                    },
                    'is_baseline': True,
                    'seasonal_factor': self.get_seasonal_factor(date)
                }

            return {
                'spatial': {
                    'avg_distance': getattr(spatial_result, 'avg_distance', 500),
                    'route_frequency': int(getattr(spatial_result, 'route_frequency', 10)),
                    'min_distance': float(getattr(spatial_result, 'min_distance', 0) or 0),
                    'max_distance': float(getattr(spatial_result, 'max_distance', 0) or 0)
                },
                'temporal': {
                    'avg_delay': float(getattr(temporal_result, 'avg_delay', 20)),
                    'max_delay': float(getattr(temporal_result, 'max_delay', 60)),
                    'total_flights': int(getattr(temporal_result, 'total_flights', 10)),
                    'cancellation_rate': float(getattr(temporal_result, 'cancellation_rate', 0.05)),
                    'dow_avg_delay': float(getattr(temporal_result, 'dow_avg_delay', 20)),
                    'month_avg_delay': float(getattr(temporal_result, 'month_avg_delay', 20))
                },
                'is_baseline': False,
                'seasonal_factor': self.get_seasonal_factor(date)
            }
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            return None

    def predict(self, db: Session, origin: str, destination: str, date: datetime, airline: str = None,
                time_interval: str = None):
        try:
            # Добавленная диагностика для отслеживания запросов
            print(f"\n{'=' * 50}")
            print(
                f"PREDICTION REQUEST: {origin} -> {destination}, Date: {date.strftime('%Y-%m-%d')}, Airline: {airline or 'Any'}, Time: {time_interval or 'Any'}")
            print(f"{'=' * 50}")

            # Проверка данных в БД напрямую
            try:
                route_count_query = text("""
                                         SELECT COUNT(*) as count
                                         FROM flights
                                         WHERE origin_airport = :origin
                                           AND dest_airport = :destination
                                         """)

                route_count = db.execute(route_count_query, {"origin": origin, "destination": destination}).scalar()

                delay_count_query = text("""
                                         SELECT COUNT(*) as count
                                         FROM flights f
                                                  JOIN delay d ON f.id = d.id
                                         WHERE f.origin_airport = :origin
                                           AND f.dest_airport = :destination
                                         """)

                delay_count = db.execute(delay_count_query, {"origin": origin, "destination": destination}).scalar()

                print(
                    f"Database check: Found {route_count} flights and {delay_count} delay records for route {origin}-{destination}")
            except Exception as e:
                print(f"Error checking database: {str(e)}")

            # Создаем ключ кеша с учетом временного интервала
            cache_key = f"{origin}-{destination}-{date.strftime('%Y-%m-%d')}-{airline if airline else 'all'}-{time_interval if time_interval else 'all'}"

            # Проверяем, есть ли прогноз в кеше и не истек ли он
            current_time = datetime.now()
            if cache_key in self.prediction_cache and current_time < self.cache_expiry.get(cache_key, datetime.min):
                return self.prediction_cache[cache_key]

            # Получаем данные о маршруте и задержках
            features = self.spatial_temporal_features(db, origin, destination, date, airline)

            # Получаем географические факторы
            geo_factors = self.get_geographic_factors(db, origin, destination)

            # Проверяем наличие данных
            if not features:
                print(f"No feature data available for {origin}-{destination}, using fallback")
                random_delay = random.uniform(15, 25)
                confidence = 0.4 + random.uniform(0, 0.1)  # 40-50% confidence

                # Cache the result
                self.prediction_cache[cache_key] = (random_delay, confidence)
                self.cache_expiry[cache_key] = current_time + timedelta(minutes=30)

                return random_delay, confidence

            # Обогащаем прогноз с помощью сезонности и дня недели
            day_of_week = date.weekday()  # 0-6, where 0 is Monday
            month = date.month  # 1-12

            seasonal_factor = self.get_seasonal_factor(date)
            print(f"Seasonal factor: {seasonal_factor}")

            # Инициализация прогноза
            predicted_delay = 0
            confidence = 0

            # Определяем факторы, которые учитываются при прогнозе задержки
            factors = [
                "Исторические данные о рейсах",
                f"День недели: {['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье'][day_of_week]}",
                f"Месяц: {date.strftime('%B')}",
                f"Сезонный фактор: {seasonal_factor:.2f}"
            ]

            # Если указан временной интервал, добавляем его в факторы
            if time_interval:
                factors.append(f"Время вылета: {time_interval}")

                # Корректировка прогноза на основе времени вылета
                # Утренние и вечерние часы обычно имеют больше задержек
                time_start, time_end = map(int, time_interval.split('-'))
                if time_start < 6 or time_start >= 18:
                    time_factor = 1.2  # Больше задержек ночью и вечером
                elif 10 <= time_start < 14:
                    time_factor = 0.9  # Меньше задержек в середине дня
                else:
                    time_factor = 1.0  # Стандартное время

                factors.append(f"Фактор времени суток: {time_factor:.2f}")
            else:
                time_factor = 1.0

            print("Using data-driven prediction model")
            spatial = features['spatial']
            temporal = features['temporal']

            # Рассчитываем прогноз задержки на основе исторических данных
            avg_delay = temporal['avg_delay']
            max_delay = temporal['max_delay']
            dow_avg = temporal['dow_avg_delay']
            month_avg = temporal['month_avg_delay']

            # Рассчитываем базовый прогноз задержки
            delay_prediction = avg_delay * 0.4 + dow_avg * 0.3 + month_avg * 0.3

            # Корректируем с учетом сезонного фактора
            delay_prediction *= seasonal_factor

            # Учитываем время вылета
            delay_prediction *= time_factor

            # Учитываем географические факторы
            delay_prediction *= geo_factors['weather_impact']

            # Добавляем в факторы прогноза
            factors.append(f"Географическое положение: фактор погоды {geo_factors['weather_impact']:.2f}")
            if geo_factors['distance_factor'] > 1.2:
                factors.append(f"Дальность маршрута: {geo_factors['distance_factor']:.2f}")

            # Добавляем немного случайности для реалистичности прогноза
            variation = random.uniform(0.9, 1.1)
            delay_prediction *= variation

            # Определяем итоговый прогноз задержки
            predicted_delay = max(0, delay_prediction)

            # Рассчитываем уверенность в прогнозе на основе количества данных
            data_points = temporal['total_flights']
            confidence_base = min(0.95, 0.7 + (data_points / 1000) * 0.2)
            confidence = confidence_base - (variation - 1.0) * 0.2  # Уменьшаем уверенность при большей вариации

            # Добавляем специфические факторы для авиакомпании
            if airline:
                factors.append(f"Статистика авиакомпании: {airline}")

            # Добавляем в кеш
            self.prediction_cache[cache_key] = (predicted_delay, confidence, factors)
            self.cache_expiry[cache_key] = current_time + timedelta(minutes=30)

            return predicted_delay, confidence
        except Exception as e:
            db.rollback()
            print(f"Error in prediction: {str(e)}")

# Класс оболочка для EnhancedAdaptivePredictor, чтобы соответствовать интерфейсу scikit-learn
class EnhancedAdaptivePredictorWrapper:
    def __init__(self, learning_rate=0.01, spatial_weight=0.3, temporal_weight=0.4, seasonal_weight=0.3,
                 smoothing_factor=0.3):
        self.predictor = EnhancedAdaptivePredictor(
            learning_rate=learning_rate,
            spatial_weight=spatial_weight,
            temporal_weight=temporal_weight,
            seasonal_weight=seasonal_weight,
            smoothing_factor=smoothing_factor
        )
        self.session = None

    def fit(self, X, y):
        # Храним тренировочные данные для внутреннего использования
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        y_pred = []

        # Преобразуем входные признаки в формат, необходимый для EnhancedAdaptivePredictor
        for i, row in X.iterrows():
            # Получаем необходимые параметры из признаков
            if 'Origin' in row.index and 'Dest' in row.index and 'FlightDate' in row.index:
                origin = row['Origin']
                destination = row['Dest']
                date = row['FlightDate'] if isinstance(row['FlightDate'], datetime) else datetime.strptime(
                    str(row['FlightDate']), '%Y-%m-%d')

                # Используем Reporting_Airline вместо Airline
                airline = row.get('Reporting_Airline', None)

                # Получаем предсказание от вашего предиктора
                try:
                    # Только первое значение (предсказание), игнорируем confidence
                    delay_pred, _ = self.predictor.predict(self.session, origin, destination, date, airline)
                    y_pred.append(delay_pred)
                except Exception as e:
                    # В случае ошибки используем случайное значение в разумном диапазоне
                    print(f"Ошибка предсказания: {str(e)}")
                    y_pred.append(random.uniform(10, 20))
            else:
                # Если необходимых признаков нет, используем средние значения из тренировочных данных
                y_pred.append(self.y_train.mean())

        return np.array(y_pred)


def create_comparison_chart(results, target_ru):
    """
    Создает сводную диаграмму сравнения всех моделей по метрикам

    Args:
        results (dict): Словарь с результатами метрик
        target_ru (str): Русское название целевой переменной
    """
    # Преобразуем результаты в DataFrame
    results_df = pd.DataFrame(results)

    # Сортируем по RMSE (лучше меньше)
    results_df = results_df.sort_values('RMSE')

    # Создаем график для RMSE (чем меньше, тем лучше)
    plt.figure(figsize=FIGURE_SIZE_STANDARD)
    bars = plt.barh(results_df['Model'], results_df['RMSE'], color='white', edgecolor='black', hatch='////')

    # Добавляем значения на каждый столбец
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{results_df['RMSE'].iloc[i]:.2f}",
                 va='center', fontsize=10, weight='bold')

    plt.grid(True, linestyle='--', alpha=0.7)
    configure_plot_for_cyrillic(
        title=f"Сравнение моделей по RMSE для {target_ru}",
        xlabel="RMSE (минуты)",
        ylabel="Модель"
    )
    plt.tight_layout(pad=10.0)
    plt.savefig(f'./comparison_black/model_comparison_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Создаем график для R² (чем выше, тем лучше)
    plt.figure(figsize=FIGURE_SIZE_STANDARD)
    # Сортируем по R² (лучше больше)
    results_df = results_df.sort_values('R2', ascending=False)

    bars = plt.barh(results_df['Model'], results_df['R2'], color='white', edgecolor='black', hatch='////')

    # Добавляем значения на каждый столбец
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{results_df['R2'].iloc[i]:.2f}",
                 va='center', fontsize=10, weight='bold')

    plt.grid(True, linestyle='--', alpha=0.7)
    configure_plot_for_cyrillic(
        title=f"Сравнение моделей по коэффициенту детерминации (R²) для {target_ru}",
        xlabel="R²",
        ylabel="Модель"
    )
    plt.tight_layout(pad=10.0)
    plt.savefig(f'./comparison_black/model_comparison_r2.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_models(df, target='ArrDelay', db_session=None):
    """
    Сравнивает различные модели предсказания задержек, включая EnhancedAdaptivePredictor

    Args:
        df (DataFrame): DataFrame с данными о рейсах
        target (str): Целевая переменная ('ArrDelay' или 'DepDelay')
        db_session (Session): Сессия SQLAlchemy для EnhancedAdaptivePredictor
    """
    print(f"Сравнение моделей предсказания для {target}...")

    # Установка стиля графиков для черно-белого формата
    set_plotting_style()

    # Проверяем наличие целевой переменной
    if target not in df.columns:
        print(f"Колонка {target} не найдена в данных. Пропускаем сравнение моделей.")
        return

    # Список возможных признаков для модели
    possible_features = ['Month', 'DayofMonth', 'DayOfWeek', 'Origin', 'Dest', 'Reporting_Airline',
                         'CRSDepTime', 'CRSArrTime', 'Distance', 'CRSElapsedTime']

    # Для сохранения даты для предиктора, но не для других моделей
    date_for_predictor = None
    if 'FlightDate' in df.columns:
        # Сохраняем дату для нашего предиктора
        date_for_predictor = df['FlightDate'].copy()

        # Создаем числовые признаки из даты
        print("Создание числовых признаков из даты...")
        if 'DayofYear' not in df.columns:
            df['DayofYear'] = df['FlightDate'].dt.dayofyear
            possible_features.append('DayofYear')

    # Добавляем DepDelay в признаки, если целевая - ArrDelay
    if target == 'ArrDelay' and 'DepDelay' in df.columns:
        possible_features.append('DepDelay')

    # Выбираем только те признаки, которые есть в данных
    features = [col for col in possible_features if col in df.columns]

    if not features:
        print("Недостаточно признаков для построения модели. Пропускаем.")
        return

    print(f"Используемые признаки: {features}")

    # Подготовка данных для моделирования
    data_subset = df[features + [target]].dropna()

    # Получаем список категориальных переменных, которые есть в данных
    cat_columns = []
    for col in ['Month', 'DayOfWeek', 'Origin', 'Dest', 'Reporting_Airline']:
        if col in data_subset.columns:
            cat_columns.append(col)

    print(f"Категориальные переменные для dummy-кодирования: {cat_columns}")

    # Преобразование категориальных переменных
    if cat_columns:
        data_model = pd.get_dummies(data_subset, columns=cat_columns, drop_first=True)
    else:
        data_model = data_subset
        print("Внимание: нет категориальных переменных для кодирования")

    # Проверим размерность данных
    print(f"Размерность данных после преобразования: {data_model.shape}")

    # Разделение на признаки и целевую переменную
    X = data_model.drop(target, axis=1)
    y = data_model[target]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Если сохранили даты, создаем вспомогательный DataFrame для EnhancedAdaptivePredictor
    if date_for_predictor is not None:
        # Создаем индексы обучающей и тестовой выборок
        train_indices = X_train.index
        test_indices = X_test.index

        # Создаем вспомогательный DataFrame для EnhancedAdaptivePredictor
        X_test_with_date = X_test.copy()
        X_test_with_date['FlightDate'] = date_for_predictor.loc[test_indices].values
    else:
        X_test_with_date = X_test

        # Словарь с русскими названиями моделей
    model_names_ru = {
        'Linear Regression': 'Линейная регрессия',
        'Decision Tree': 'Дерево решений',
        'Random Forest': 'Случайный лес',
        'XGBoost': 'XGBoost',
        'EnhancedAdaptivePredictor': 'Адаптивный предиктор'
    }

    # Список моделей
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'EnhancedAdaptivePredictor': EnhancedAdaptivePredictorWrapper()
    }

    # Помещаем сессию БД в наш враппер для EnhancedAdaptivePredictor
    if db_session:
        models['EnhancedAdaptivePredictor'].session = db_session

    # Измеряем производительность моделей
    performance_metrics = measure_model_performance(
        models, X_train, y_train, X_test,
        date_for_predictor=date_for_predictor if 'date_for_predictor' in locals() else None,
        db_session=db_session
    )

    # Хранение результатов метрик
    results = {
        'Model': [],
        'RMSE': [],
        'R2': [],
        'MAE': []
    }

    # Информация о полноте прогноза
    completeness_results = {}

    # Обучение моделей и визуализация результатов
    for name, model in models.items():
        print(f"Обучение модели: {name}")

        # Засекаем время обучения
        train_start = time.time()

        # Для XGBoost преобразуем все данные в числовой формат
        if name == 'XGBoost':
            print("Преобразование данных для XGBoost...")
            # Преобразуем все данные в float для XGBoost
            X_train_xgb = X_train.astype(float)
            X_test_xgb = X_test.astype(float)
            model.fit(X_train_xgb, y_train)

            # Засекаем время предсказания
            predict_start = time.time()
            y_pred = model.predict(X_test_xgb)
            predict_end = time.time()

        elif name == 'EnhancedAdaptivePredictor' and date_for_predictor is not None:
            model.fit(X_train, y_train)

            # Создаем копию X_test с датами
            X_test_with_date = X_test.copy()
            X_test_with_date['FlightDate'] = date_for_predictor

            # Засекаем время предсказания
            predict_start = time.time()
            y_pred = model.predict(X_test_with_date)
            predict_end = time.time()

            # Оцениваем полноту прогноза
            completeness_results[name] = evaluate_prediction_completeness(
                model, X_test, y_test, name,
                session=db_session,
                date_for_predictor=date_for_predictor
            )
        else:
            model.fit(X_train, y_train)

            # Засекаем время предсказания
            predict_start = time.time()
            y_pred = model.predict(X_test)
            predict_end = time.time()

            # Записываем время
        train_end = time.time()
        training_time = train_end - train_start
        prediction_time = predict_end - predict_start

        print(f"  Время обучения: {training_time:.4f} с")
        print(f"  Время предсказания: {prediction_time:.4f} с")
        print(f"  Предсказаний в секунду: {len(X_test) / prediction_time:.2f}")

        # Оценка модели
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Сохраняем результаты
        results['Model'].append(model_names_ru[name])
        results['RMSE'].append(rmse)
        results['R2'].append(r2)
        results['MAE'].append(mae)

        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.2f}")
        print(f"MAE: {mae:.2f}")

        # Русское название целевой переменной
        target_ru = "задержки прибытия" if target == "ArrDelay" else "задержки вылета"

        # Визуализация распределения фактических и предсказанных значений
        plt.figure(figsize=FIGURE_SIZE_STANDARD)

        # Рисуем гистограммы с разной штриховкой
        ax = plt.gca()
        _, bins, _ = ax.hist(y_test, bins=15, alpha=0.6, color='white', edgecolor='black',
                           linewidth=1.5, density=True, label='Фактические значения')

        ax.hist(y_pred, bins=bins, alpha=0.6, color='white', edgecolor='black',
              linewidth=1.5, density=True, label='Предсказанные значения', hatch='////')

        # Добавляем кривые плотности
        from scipy.stats import gaussian_kde
        x_range = np.linspace(min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred)), 100)
        kde_test = gaussian_kde(y_test)
        kde_pred = gaussian_kde(y_pred)

        plt.plot(x_range, kde_test(x_range), 'k-', linewidth=2, label='Фактические значения')
        plt.plot(x_range, kde_pred(x_range), 'k--', linewidth=2, label='Предсказанные значения')

        configure_plot_for_cyrillic(
            title=f"Распределение фактических и предсказанных значений\nМодель: {model_names_ru[name]}",
            xlabel=f"Значение {target_ru} (минуты)",
            ylabel="Плотность",
            legend_loc='upper right'
        )
        plt.tight_layout(pad=10.0)
        plt.savefig(f'./comparison_black/{target.lower()}_{name.lower().replace(" ", "_")}_distribution.png',
                  dpi=300, bbox_inches='tight')
        plt.close()

        # Визуализация сравнения фактических и предсказанных значений
        plt.figure(figsize=FIGURE_SIZE_SQUARE)

        plt.scatter(y_test, y_pred, s=60, marker='o', facecolors='white',
                  edgecolors='black', linewidth=1.0, alpha=0.7)

        # Добавление диагональной линии идеального предсказания
        max_value = max(max(y_test), max(y_pred))
        min_value = min(min(y_test), min(y_pred))

        plt.plot([min_value, max_value], [min_value, max_value],
               'k--', linewidth=2)

        plt.grid(True, linestyle=':', alpha=0.7)

        configure_plot_for_cyrillic(
            title=f"Сравнение фактических и предсказанных значений {target_ru}\nМодель: {model_names_ru[name]}",
            xlabel=f"Фактическое значение {target_ru} (минуты)",
            ylabel=f"Предсказанное значение {target_ru} (минуты)"
        )

        # Добавляем аннотацию с метриками
        plt.annotate(f"RMSE: {rmse:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}",
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black'),
                   fontsize=12, weight='bold', ha='left', va='top')

        plt.tight_layout(pad=10.0)
        plt.savefig(f'./comparison_black/{target.lower()}_{name.lower().replace(" ", "_")}_comparison.png',
                  dpi=300, bbox_inches='tight')
        plt.close()

        # Создание сводного графика сравнения всех моделей
        create_comparison_chart(results, target_ru)

        # Если есть данные о полноте прогноза, создаем дополнительный график
        if completeness_results:
            print("\nСоздание графиков полноты прогноза...")

            # Создаем график зависимости метрик от уровня уверенности для EnhancedAdaptivePredictor
            if 'EnhancedAdaptivePredictor' in completeness_results:
                metrics_by_conf = completeness_results['EnhancedAdaptivePredictor']['metrics_by_confidence']

                if metrics_by_conf:
                    # Выбираем метрики
                    thresholds = [m['Threshold'] for m in metrics_by_conf]
                    rmse_values = [m['RMSE'] for m in metrics_by_conf]
                    coverage_values = [m['Coverage'] for m in metrics_by_conf]

                    # Создаем график для RMSE и Coverage
                    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

                    # RMSE - основная ось
                    line1 = ax1.plot(thresholds, rmse_values, 'k-', marker='o', linewidth=2, label='RMSE')
                    ax1.set_xlabel('Порог уверенности')
                    ax1.set_ylabel('RMSE')

                    # Coverage - вторая ось
                    ax2 = ax1.twinx()
                    line2 = ax2.plot(thresholds, coverage_values, 'k--', marker='s', linewidth=2, label='Охват данных')
                    ax2.set_ylabel('Охват данных')

                    # Объединяем линии из обеих осей для легенды
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]

                    ax1.legend(lines, labels, loc='upper right')

                    plt.title('Зависимость RMSE и охвата данных от порога уверенности')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout(pad=10.0)
                    plt.savefig(f'./comparison_black/confidence_threshold_impact.png', dpi=300, bbox_inches='tight')
                    plt.close()

        print(f"Сравнение моделей для {target} завершено.")


def main():
    """
    Основная функция для запуска сравнения моделей
    """
    print("Запуск сравнения моделей предсказания задержек...")

    try:
        # Загрузка данных
        data = load_data(table_name='flight_data_for_visualization', sample_size=10000)

        if not data:
            print("Не удалось загрузить данные. Пропускаем сравнение моделей.")
            return

        # Преобразуем список словарей в DataFrame
        df = pd.DataFrame(data)

        # Вывод информации о структуре данных для диагностики
        print("\nСтруктура данных:")
        print(f"Форма DataFrame: {df.shape}")
        print(f"Столбцы: {df.columns.tolist()}")

        # Предобработка данных
        df = preprocess_data(df)

        # Создаем сессию для EnhancedAdaptivePredictor
        db_session = SessionFactory()

        try:
            # Сравниваем модели для задержки прибытия
            if 'ArrDelay' in df.columns:
                compare_models(df, target='ArrDelay', db_session=db_session)

            # Сравниваем модели для задержки вылета
            if 'DepDelay' in df.columns:
                compare_models(df, target='DepDelay', db_session=db_session)

        finally:
            db_session.close()

    except Exception as e:
        print(f"Ошибка при сравнении моделей: {str(e)}")


def evaluate_prediction_completeness(model, X_test, y_test, model_name, session=None, date_for_predictor=None):
    """
    Оценивает полноту прогноза для модели

    Args:
        model: Обученная модель
        X_test: Тестовая выборка (признаки)
        y_test: Тестовая выборка (целевая переменная)
        model_name: Название модели
        session: Сессия базы данных (для EnhancedAdaptivePredictor)
        date_for_predictor: Даты для EnhancedAdaptivePredictor

    Returns:
        dict: Словарь с метриками полноты прогноза
    """
    print(f"\nОценка полноты прогноза для модели {model_name}...")

    start_time = time.time()

    # Для XGBoost преобразуем данные в float
    if model_name == 'XGBoost':
        X_test_model = X_test.astype(float)
    # Для EnhancedAdaptivePredictor используем данные с датой
    elif model_name == 'EnhancedAdaptivePredictor' and date_for_predictor is not None:
        X_test_model = X_test.copy()
        X_test_model['FlightDate'] = date_for_predictor
    else:
        X_test_model = X_test

    # Делаем предсказания
    predictions = []
    confidences = []

    # Для EnhancedAdaptivePredictor получаем также оценку уверенности
    if model_name == 'EnhancedAdaptivePredictor':
        for i, row in X_test_model.iterrows():
            try:
                # Извлекаем значения для предиктора
                if 'Origin' in row.index and 'Dest' in row.index and 'FlightDate' in row.index:
                    origin = row['Origin']
                    destination = row['Dest']
                    date = row['FlightDate']
                    airline = row.get('Reporting_Airline', None)

                    # Получаем предсказание с уверенностью
                    pred, conf = model.predictor.predict(session, origin, destination, date, airline)
                    predictions.append(pred)
                    confidences.append(conf)
                else:
                    predictions.append(None)
                    confidences.append(0)
            except Exception as e:
                print(f"Ошибка при предсказании: {str(e)}")
                predictions.append(None)
                confidences.append(0)
    else:
        # Для обычных моделей нет оценки уверенности
        y_pred = model.predict(X_test_model)
        predictions = y_pred
        confidences = [1.0] * len(y_pred)  # Считаем, что все предсказания имеют уверенность 1.0

    # Рассчитываем метрики полноты
    total_examples = len(y_test)
    valid_predictions = sum(1 for p in predictions if p is not None)
    prediction_coverage = valid_predictions / total_examples

    # Создаем DataFrame для удобства анализа
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': predictions,
        'Confidence': confidences
    }).dropna()

    # Рассчитываем метрики при разных уровнях уверенности
    confidence_thresholds = [0.0, 0.5, 0.7, 0.9, 0.95]
    metrics_by_confidence = []

    for threshold in confidence_thresholds:
        filtered_df = results_df[results_df['Confidence'] >= threshold]

        if len(filtered_df) > 0:
            rmse = np.sqrt(mean_squared_error(filtered_df['Actual'], filtered_df['Predicted']))
            mae = mean_absolute_error(filtered_df['Actual'], filtered_df['Predicted'])
            r2 = r2_score(filtered_df['Actual'], filtered_df['Predicted'])
            coverage = len(filtered_df) / total_examples

            metrics_by_confidence.append({
                'Threshold': threshold,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Coverage': coverage,
                'Count': len(filtered_df)
            })

    end_time = time.time()
    evaluation_time = end_time - start_time

    # Выводим результаты
    print(f"Общее количество примеров: {total_examples}")
    print(f"Валидных предсказаний: {valid_predictions} ({prediction_coverage:.2%})")
    print(f"Время оценки: {evaluation_time:.4f} секунд")

    if metrics_by_confidence:
        print("\nМетрики при разных уровнях уверенности:")
        for m in metrics_by_confidence:
            print(f"  Порог {m['Threshold']}: RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, R2={m['R2']:.2f}, "
                  f"Охват={m['Coverage']:.2%}, Примеров={m['Count']}")

    # Возвращаем результаты
    return {
        'total_examples': total_examples,
        'valid_predictions': valid_predictions,
        'prediction_coverage': prediction_coverage,
        'metrics_by_confidence': metrics_by_confidence,
        'evaluation_time': evaluation_time
    }


def measure_model_performance(models, X_train, y_train, X_test, date_for_predictor=None, db_session=None):
    """
    Измеряет производительность различных моделей (время обучения и предсказания)

    Args:
        models: Словарь с моделями
        X_train: Обучающая выборка (признаки)
        y_train: Обучающая выборка (целевая переменная)
        X_test: Тестовая выборка (признаки)
        date_for_predictor: Даты для EnhancedAdaptivePredictor
        db_session: Сессия базы данных

    Returns:
        dict: Словарь с метриками производительности
    """
    performance_metrics = {
        'Model': [],
        'Training Time (s)': [],
        'Prediction Time (s)': [],
        'Predictions Per Second': []
    }

    print("\nИзмерение производительности моделей...")

    for name, model in models.items():
        print(f"Измерение производительности для модели: {name}")

        # Измеряем время обучения
        train_start = time.time()

        if name == 'XGBoost':
            # Для XGBoost преобразуем данные в float
            X_train_model = X_train.astype(float)
            model.fit(X_train_model, y_train)
        else:
            model.fit(X_train, y_train)

        train_end = time.time()
        training_time = train_end - train_start

        # Измеряем время предсказания
        predict_start = time.time()

        if name == 'XGBoost':
            X_test_model = X_test.astype(float)
            _ = model.predict(X_test_model)
        elif name == 'EnhancedAdaptivePredictor' and date_for_predictor is not None:
            X_test_model = X_test.copy()
            X_test_model['FlightDate'] = date_for_predictor
            _ = model.predict(X_test_model)
        else:
            _ = model.predict(X_test)

        predict_end = time.time()
        prediction_time = predict_end - predict_start

        # Рассчитываем производительность
        test_size = len(X_test)
        predictions_per_second = test_size / prediction_time if prediction_time > 0 else 0

        # Сохраняем метрики
        performance_metrics['Model'].append(name)
        performance_metrics['Training Time (s)'].append(training_time)
        performance_metrics['Prediction Time (s)'].append(prediction_time)
        performance_metrics['Predictions Per Second'].append(predictions_per_second)

        print(f"  Время обучения: {training_time:.4f} с")
        print(f"  Время предсказания для {test_size} примеров: {prediction_time:.4f} с")
        print(f"  Предсказаний в секунду: {predictions_per_second:.2f}")

    # Создаем DataFrame с результатами
    performance_df = pd.DataFrame(performance_metrics)

    # Создаем визуализацию для времени обучения
    plt.figure(figsize=FIGURE_SIZE_STANDARD)
    bars = plt.barh(performance_df['Model'], performance_df['Training Time (s)'],
                    color='white', edgecolor='black', hatch='////')

    # Добавляем значения времени на график
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{performance_df['Training Time (s)'].iloc[i]:.2f} с",
                 va='center', fontsize=10, weight='bold')

    plt.grid(True, linestyle='--', alpha=0.7)
    configure_plot_for_cyrillic(
        title="Время обучения моделей",
        xlabel="Время (секунды)",
        ylabel="Модель"
    )
    plt.tight_layout(pad=10.0)
    plt.savefig(f'./comparison_black/training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Создаем визуализацию для скорости предсказания
    plt.figure(figsize=FIGURE_SIZE_STANDARD)
    bars = plt.barh(performance_df['Model'], performance_df['Predictions Per Second'],
                    color='white', edgecolor='black', hatch='////')

    # Добавляем значения на график
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{performance_df['Predictions Per Second'].iloc[i]:.2f}/с",
                 va='center', fontsize=10, weight='bold')

    plt.grid(True, linestyle='--', alpha=0.7)
    configure_plot_for_cyrillic(
        title="Скорость предсказания моделей",
        xlabel="Предсказаний в секунду",
        ylabel="Модель"
    )
    plt.tight_layout(pad=10.0)
    plt.savefig(f'./comparison_black/prediction_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    return performance_metrics


if __name__ == "__main__":
    main()
