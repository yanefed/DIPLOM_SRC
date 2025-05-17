#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
from datetime import datetime
import random
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

# Импорт функций из оригинального скрипта
from generate_flight_data import (
    calculate_distance,
    generate_flights,
    insert_data,
    generate_flight_airports
)

# Подключение к базе данных
db_host = os.environ.get('POSTGRES_HOST', 'localhost')
db_port = os.environ.get('DATABASE_PORT', '5432')
db_user = os.environ.get('POSTGRES_USER', 'postgres')
db_password = os.environ.get('POSTGRES_PASSWORD', '0252')
db_name = os.environ.get('POSTGRES_DB', 'ru_postgres')

connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)
Session = sessionmaker(bind=engine)


def get_existing_data():
    """Получает существующие данные из базы данных"""
    session = Session()
    try:
        # Получаем авиакомпании
        airlines_query = session.execute(text("SELECT airline_code, airline_name FROM airlines"))
        airlines = [{"airline_code": row[0], "airline_name": row[1]} for row in airlines_query]

        # Получаем аэропорты
        airports_query = session.execute(text("""
            SELECT airport_code, display_airport_name, airport_city, airport_fullname, 
                   airport_state, airport_country, latitude, longitude, id 
            FROM airports
        """))

        airports = []
        for row in airports_query:
            airports.append({
                "airport_code": row[0],
                "name": row[1],
                "city": row[2],
                "fullname": row[3],
                "state": row[4],
                "country": row[5],
                "lat": row[6],
                "lon": row[7],
                "id": row[8]
            })

        # Получаем самолеты
        planes_query = session.execute(text("""
            SELECT tail_num, plane_type, manufacture_year, number_of_seats, airline_code 
            FROM planes
        """))

        planes = []
        for row in planes_query:
            planes.append({
                "tail_num": row[0],
                "plane_type": row[1],
                "manufacture_year": row[2],
                "number_of_seats": row[3],
                "airline_code": row[4]
            })

        # Получаем максимальный ID рейса
        max_id_query = session.execute(text("SELECT MAX(id) FROM flights"))
        max_flight_id = max_id_query.scalar() or 0

        return {
            "airlines": airlines,
            "airports": airports,
            "planes": planes,
            "max_flight_id": max_flight_id
        }
    except Exception as e:
        print(f"Ошибка при получении существующих данных: {str(e)}")
        return {"airlines": [], "airports": [], "planes": [], "max_flight_id": 0}
    finally:
        session.close()


def generate_additional_data(num_flights, start_date, end_date):
    """Генерирует дополнительные данные для таблиц flights, delay и flight_airport"""
    print(f"Генерация {num_flights} дополнительных рейсов с {start_date} по {end_date}")

    # Получаем существующие данные
    existing_data = get_existing_data()
    airlines = existing_data["airlines"]
    airports = existing_data["airports"]
    planes = existing_data["planes"]
    max_flight_id = existing_data["max_flight_id"]

    print(f"Найдено {len(airlines)} авиакомпаний, {len(airports)} аэропортов, {len(planes)} самолетов")
    print(f"Последний ID рейса: {max_flight_id}")

    if not airlines or not airports or not planes:
        print("Ошибка: Недостаточно данных в базе для генерации новых рейсов")
        return

    # Преобразуем строковые даты в объекты datetime
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    # Генерируем рейсы с новыми ID
    print("Генерация рейсов и задержек...")
    flights, delays = generate_flights(airlines, airports, planes, num_flights, start_date_obj, end_date_obj)

    # Обновляем ID рейсов, чтобы не конфликтовать с существующими
    for i, flight in enumerate(flights):
        # Устанавливаем ID начиная с max_flight_id + 1
        flight["id"] = max_flight_id + i + 1

    # Обновляем ID в данных о задержках, чтобы они соответствовали ID рейсов
    for i, delay in enumerate(delays):
        delay["id"] = max_flight_id + i + 1

    # Вставляем данные в базу
    print("Вставка данных о рейсах...")
    insert_data(flights, "flights")

    print("Вставка данных о задержках...")
    insert_data(delays, "delay")

    # Генерируем и вставляем данные flight_airport
    print("Генерация и вставка данных flight_airport...")
    flight_airports = generate_flight_airports(flights, airports)
    insert_data(flight_airports, "flight_airport")

    print(f"Генерация данных завершена. Добавлено {len(flights)} рейсов.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Генерация дополнительных данных о рейсах')
    parser.add_argument('--flights', type=int, default=10000000,
                        help='Количество рейсов для генерации (по умолчанию: 50000)')
    parser.add_argument('--start_date', type=str, default='2024-01-01',
                        help='Начальная дата в формате YYYY-MM-DD (по умолчанию: 2024-01-01)')
    parser.add_argument('--end_date', type=str, default='2024-12-31',
                        help='Конечная дата в формате YYYY-MM-DD (по умолчанию: 2024-12-31)')

    args = parser.parse_args()

    generate_additional_data(args.flights, args.start_date, args.end_date)
