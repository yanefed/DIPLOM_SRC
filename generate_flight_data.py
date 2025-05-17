import random
import argparse
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

# Подключение к базе данных
db_host = os.environ.get('POSTGRES_HOST', 'localhost')
db_port = os.environ.get('DATABASE_PORT', '5432')
db_user = os.environ.get('POSTGRES_USER', 'postgres')
db_password = os.environ.get('POSTGRES_PASSWORD', '0252')
db_name = os.environ.get('POSTGRES_DB', 'ru_postgres')

connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)
Session = sessionmaker(bind=engine)

def create_tables_if_not_exists():
    """Создать таблицы, если они не существуют"""
    session = Session()
    try:
        # Проверка существования таблиц
        check_tables_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'airlines'
        ) as airlines_exists,
        EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'airports'
        ) as airports_exists,
        EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'flights'
        ) as flights_exists,
        EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'delay'
        ) as delay_exists,
        EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'planes'
        ) as planes_exists,
        EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'flight_airport'
        ) as flight_airport_exists
        """

        result = session.execute(text(check_tables_query)).first()

        # Создаем таблицы, если их нет
        flight_airport_exists = result[5] if len(result) > 5 else False
        if not all(result[:5]) or not flight_airport_exists:  # Check all tables including flight_airport
            print("Создание недостающих таблиц...")

            if not result[0]:  # airlines
                session.execute(text("""
                CREATE TABLE public.airlines (
                    airline_name VARCHAR(255),
                    airline_code VARCHAR(10) PRIMARY KEY
                )
                """))

            if not result[1]:  # airports
                session.execute(text("""
                CREATE TABLE public.airports (
                    id SERIAL,
                    display_airport_name VARCHAR(255),
                    airport_code VARCHAR(10) PRIMARY KEY,
                    airport_city VARCHAR(255),
                    airport_fullname VARCHAR(255),
                    airport_state VARCHAR(50),
                    airport_country VARCHAR(50),
                    latitude FLOAT,
                    longitude FLOAT
                )
                """))

            if not result[4]:  # planes
                session.execute(text("""
                CREATE TABLE public.planes (
                    manufacture_year INTEGER,
                    tail_num VARCHAR(20) PRIMARY KEY,
                    number_of_seats INTEGER,
                    plane_type VARCHAR(50),
                    airline_code VARCHAR(10) REFERENCES airlines(airline_code)
                )
                """))

            if not result[2]:  # flights
                session.execute(text("""
                CREATE TABLE public.flights (
                    id SERIAL PRIMARY KEY,
                    fl_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    airline_code VARCHAR(10) REFERENCES airlines(airline_code),
                    origin_airport VARCHAR(10) REFERENCES airports(airport_code),
                    dest_airport VARCHAR(10) REFERENCES airports(airport_code),
                    distance INTEGER,
                    tail_num VARCHAR(20) REFERENCES planes(tail_num),
                    dep_time VARCHAR(10),
                    arr_time VARCHAR(10),
                    air_time INTEGER
                )
                """))

            if not result[3]:  # delay
                session.execute(text("""
                CREATE TABLE public.delay (
                    id INTEGER PRIMARY KEY REFERENCES flights(id),
                    dep_delay INTEGER,
                    arr_delay INTEGER,
                    cancelled INTEGER DEFAULT 0,
                    cancellation_code VARCHAR(5)
                )
                """))

            # Check for flight_airport table
            if not flight_airport_exists:  # flight_airport
                session.execute(text("""
                CREATE TABLE public.flight_airport (
                    flight_id INTEGER REFERENCES flights(id),
                    airport_id INTEGER,
                    airport_type VARCHAR(20) NOT NULL,
                    PRIMARY KEY (flight_id, airport_id)
                )
                """))

            session.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
            session.commit()
            print("Таблицы успешно созданы.")
        else:
            print("Все необходимые таблицы уже существуют.")

    except Exception as e:
        session.rollback()
        print(f"Ошибка при создании таблиц: {str(e)}")
    finally:
        session.close()

def generate_airlines():
    """Генерирует данные авиакомпаний"""
    # airlines = [
    #     {"airline_code": "G4"},
    #     {"airline_code": "YX"},
    #     {"airline_code": "WN"},
    #     {"airline_code": "UA"},
    #     {"airline_code": "OO"},
    #     {"airline_code": "F9"},
    #     {"airline_code": "9E"},
    #     {"airline_code": "DL"},
    #     {"airline_code": "NK"}
    # ]
    airlines = [
            {"airline_code": "SU", "airline_name": "Аэрофлот"},
            {"airline_code": "S7", "airline_name": "S7 Airlines"},
            {"airline_code": "U6", "airline_name": "Уральские Авиалинии"},
            {"airline_code": "UT", "airline_name": "ЮТэйр"},
            {"airline_code": "N4", "airline_name": "Северный Ветер (Nordwind)"},
            {"airline_code": "5N", "airline_name": "SmartAvia"},
            {"airline_code": "A4", "airline_name": "Азимут"},
            {"airline_code": "R3", "airline_name": "Якутия"},
            {"airline_code": "IK", "airline_name": "Ираэро"},
            {"airline_code": "KL", "airline_name": "Россия"},
            {"airline_code": "D9", "airline_name": "Победа"},
            {"airline_code": "WZ", "airline_name": "Red Wings"},
            {"airline_code": "R2", "airline_name": "РусЛайн"},
            {"airline_code": "L2", "airline_name": "Авиакомпания «ИрАэро»"},
            {"airline_code": "H9", "airline_name": "Хортица"},
            {"airline_code": "N6", "airline_name": "Нордавиа"},
            {"airline_code": "Y7", "airline_name": "Саратовские авиалинии"},
            {"airline_code": "A3", "airline_name": "Авиакомпания «Трансаэро»"},
            {"airline_code": "D8", "airline_name": "Дальавиа"},
            {"airline_code": "F7", "airline_name": "Авиакомпания «Томск Авиа»"},
            {"airline_code": "Q3", "airline_name": "Кубань-авиа"}
    ]

    return airlines

def generate_airports():
    """Генерирует данные аэропортов"""
    # airport_codes = [
    #     {"airport_code": "FLL", "lat": 26.0742, "lon": -80.1506, "name": "Fort Lauderdale-Hollywood International Airport", "city": "Fort Lauderdale", "state": "FL", "country": "USA", "fullname": "Fort Lauderdale-Hollywood International Airport"},
    #     {"airport_code": "CLE", "lat": 41.4117, "lon": -81.8550, "name": "Cleveland Hopkins International Airport", "city": "Cleveland", "state": "OH", "country": "USA", "fullname": "Cleveland Hopkins International Airport"},
    #     {"airport_code": "CLT", "lat": 35.2141, "lon": -80.9431, "name": "Charlotte Douglas International Airport", "city": "Charlotte", "state": "NC", "country": "USA", "fullname": "Charlotte Douglas International Airport"},
    #     {"airport_code": "PBI", "lat": 26.6832, "lon": -80.0956, "name": "Palm Beach International Airport", "city": "West Palm Beach", "state": "FL", "country": "USA", "fullname": "Palm Beach International Airport"},
    #     {"airport_code": "AUS", "lat": 30.1975, "lon": -97.6664, "name": "Austin-Bergstrom International Airport", "city": "Austin", "state": "TX", "country": "USA", "fullname": "Austin-Bergstrom International Airport"},
    #     {"airport_code": "BUR", "lat": 34.2012, "lon": -118.3598, "name": "Hollywood Burbank Airport", "city": "Burbank", "state": "CA", "country": "USA", "fullname": "Hollywood Burbank Airport"},
    #     {"airport_code": "ABY", "lat": 31.5356, "lon": -84.1945, "name": "Southwest Georgia Regional Airport", "city": "Albany", "state": "GA", "country": "USA", "fullname": "Southwest Georgia Regional Airport"},
    #     {"airport_code": "AGS", "lat": 33.3699, "lon": -81.9645, "name": "Augusta Regional Airport", "city": "Augusta", "state": "GA", "country": "USA", "fullname": "Augusta Regional Airport"},
    #     {"airport_code": "ANC", "lat": 61.1743, "lon": -149.9963, "name": "Ted Stevens Anchorage International Airport", "city": "Anchorage", "state": "AK", "country": "USA", "fullname": "Ted Stevens Anchorage International Airport"},
    #     {"airport_code": "BDL", "lat": 41.9389, "lon": -72.6832, "name": "Bradley International Airport", "city": "Windsor Locks", "state": "CT", "country": "USA", "fullname": "Bradley International Airport"},
    #     {"airport_code": "BHM", "lat": 33.5629, "lon": -86.7535, "name": "Birmingham-Shuttlesworth International Airport", "city": "Birmingham", "state": "AL", "country": "USA", "fullname": "Birmingham-Shuttlesworth International Airport"},
    #     {"airport_code": "BOI", "lat": 43.5645, "lon": -116.2224, "name": "Boise Airport", "city": "Boise", "state": "ID", "country": "USA", "fullname": "Boise Airport"},
    #     {"airport_code": "ABQ", "lat": 35.0402, "lon": -106.6091, "name": "Albuquerque International Sunport", "city": "Albuquerque", "state": "NM", "country": "USA", "fullname": "Albuquerque International Sunport"},
    #     {"airport_code": "DSM", "lat": 41.5341, "lon": -93.6631, "name": "Des Moines International Airport", "city": "Des Moines", "state": "IA", "country": "USA", "fullname": "Des Moines International Airport"},
    #     {"airport_code": "ELP", "lat": 31.8072, "lon": -106.3776, "name": "El Paso International Airport", "city": "El Paso", "state": "TX", "country": "USA", "fullname": "El Paso International Airport"},
    #     {"airport_code": "DTW", "lat": 42.2124, "lon": -83.3534, "name": "Detroit Metropolitan Airport", "city": "Detroit", "state": "MI", "country": "USA", "fullname": "Detroit Metropolitan Wayne County Airport"},
    #     {"airport_code": "DAL", "lat": 32.8471, "lon": -96.8518, "name": "Dallas Love Field", "city": "Dallas", "state": "TX", "country": "USA", "fullname": "Dallas Love Field"},
    #     {"airport_code": "DFW", "lat": 32.8972, "lon": -97.0377, "name": "Dallas/Fort Worth International Airport", "city": "Dallas", "state": "TX", "country": "USA", "fullname": "Dallas/Fort Worth International Airport"},
    #     {"airport_code": "CHS", "lat": 32.8986, "lon": -80.0405, "name": "Charleston International Airport", "city": "Charleston", "state": "SC", "country": "USA", "fullname": "Charleston International Airport"},
    #     {"airport_code": "FAT", "lat": 36.7758, "lon": -119.7181, "name": "Fresno Yosemite International Airport", "city": "Fresno", "state": "CA", "country": "USA", "fullname": "Fresno Yosemite International Airport"},
    #     {"airport_code": "CVG", "lat": 39.0489, "lon": -84.6678, "name": "Cincinnati/Northern Kentucky International Airport", "city": "Cincinnati", "state": "OH", "country": "USA", "fullname": "Cincinnati/Northern Kentucky International Airport"},
    #     {"airport_code": "IND", "lat": 39.7169, "lon": -86.2956, "name": "Indianapolis International Airport", "city": "Indianapolis", "state": "IN", "country": "USA", "fullname": "Indianapolis International Airport"},
    #     {"airport_code": "JFK", "lat": 40.6413, "lon": -73.7781, "name": "John F. Kennedy International Airport", "city": "New York", "state": "NY", "country": "USA", "fullname": "John F. Kennedy International Airport"},
    #     {"airport_code": "DAY", "lat": 39.9023, "lon": -84.2194, "name": "Dayton International Airport", "city": "Dayton", "state": "OH", "country": "USA", "fullname": "Dayton International Airport"},
    #     {"airport_code": "HNL", "lat": 21.3187, "lon": -157.9224, "name": "Daniel K. Inouye International Airport", "city": "Honolulu", "state": "HI", "country": "USA", "fullname": "Daniel K. Inouye International Airport"},
    #     {"airport_code": "JAX", "lat": 30.4941, "lon": -81.6879, "name": "Jacksonville International Airport", "city": "Jacksonville", "state": "FL", "country": "USA", "fullname": "Jacksonville International Airport"},
    #     {"airport_code": "PGV", "lat": 35.6353, "lon": -77.3853, "name": "Pitt-Greenville Airport", "city": "Greenville", "state": "NC", "country": "USA", "fullname": "Pitt-Greenville Airport"},
    #     {"airport_code": "MHT", "lat": 42.9326, "lon": -71.4356, "name": "Manchester-Boston Regional Airport", "city": "Manchester", "state": "NH", "country": "USA", "fullname": "Manchester-Boston Regional Airport"},
    #     {"airport_code": "BUF", "lat": 42.9404, "lon": -78.7322, "name": "Buffalo Niagara International Airport", "city": "Buffalo", "state": "NY", "country": "USA", "fullname": "Buffalo Niagara International Airport"},
    #     {"airport_code": "BWI", "lat": 39.1774, "lon": -76.6684, "name": "Baltimore/Washington International Airport", "city": "Baltimore", "state": "MD", "country": "USA", "fullname": "Baltimore/Washington International Thurgood Marshall Airport"},
    #     {"airport_code": "BOS", "lat": 42.3643, "lon": -71.0052, "name": "Boston Logan International Airport", "city": "Boston", "state": "MA", "country": "USA", "fullname": "Boston Logan International Airport"},
    #     {"airport_code": "LAX", "lat": 33.9416, "lon": -118.4085, "name": "Los Angeles International Airport", "city": "Los Angeles", "state": "CA", "country": "USA", "fullname": "Los Angeles International Airport"},
    #     {"airport_code": "KOA", "lat": 19.7388, "lon": -156.0456, "name": "Ellison Onizuka Kona International Airport", "city": "Kona", "state": "HI", "country": "USA", "fullname": "Ellison Onizuka Kona International Airport at Keahole"},
    #     {"airport_code": "LGA", "lat": 40.7769, "lon": -73.8740, "name": "LaGuardia Airport", "city": "New York", "state": "NY", "country": "USA", "fullname": "LaGuardia Airport"},
    #     {"airport_code": "GRR", "lat": 42.8808, "lon": -85.5228, "name": "Gerald R. Ford International Airport", "city": "Grand Rapids", "state": "MI", "country": "USA", "fullname": "Gerald R. Ford International Airport"},
    #     {"airport_code": "EWR", "lat": 40.6895, "lon": -74.1745, "name": "Newark Liberty International Airport", "city": "Newark", "state": "NJ", "country": "USA", "fullname": "Newark Liberty International Airport"},
    #     {"airport_code": "MYR", "lat": 33.6797, "lon": -78.9283, "name": "Myrtle Beach International Airport", "city": "Myrtle Beach", "state": "SC", "country": "USA", "fullname": "Myrtle Beach International Airport"},
    #     {"airport_code": "LAS", "lat": 36.0800, "lon": -115.1522, "name": "Harry Reid International Airport", "city": "Las Vegas", "state": "NV", "country": "USA", "fullname": "Harry Reid International Airport"},
    #     {"airport_code": "GSO", "lat": 36.0978, "lon": -79.9373, "name": "Piedmont Triad International Airport", "city": "Greensboro", "state": "NC", "country": "USA", "fullname": "Piedmont Triad International Airport"},
    #     {"airport_code": "DCA", "lat": 38.8512, "lon": -77.0402, "name": "Ronald Reagan Washington National Airport", "city": "Washington", "state": "DC", "country": "USA", "fullname": "Ronald Reagan Washington National Airport"},
    #     {"airport_code": "CMH", "lat": 39.9999, "lon": -82.8872, "name": "John Glenn Columbus International Airport", "city": "Columbus", "state": "OH", "country": "USA", "fullname": "John Glenn Columbus International Airport"},
    #     {"airport_code": "DEN", "lat": 39.8561, "lon": -104.6737, "name": "Denver International Airport", "city": "Denver", "state": "CO", "country": "USA", "fullname": "Denver International Airport"},
    #     {"airport_code": "TPA", "lat": 27.9756, "lon": -82.5333, "name": "Tampa International Airport", "city": "Tampa", "state": "FL", "country": "USA", "fullname": "Tampa International Airport"},
    #     {"airport_code": "SYR", "lat": 43.1112, "lon": -76.1062, "name": "Syracuse Hancock International Airport", "city": "Syracuse", "state": "NY", "country": "USA", "fullname": "Syracuse Hancock International Airport"},
    #     {"airport_code": "GEG", "lat": 47.6199, "lon": -117.5354, "name": "Spokane International Airport", "city": "Spokane", "state": "WA", "country": "USA", "fullname": "Spokane International Airport"},
    #     {"airport_code": "HOU", "lat": 29.6454, "lon": -95.2789, "name": "William P. Hobby Airport", "city": "Houston", "state": "TX", "country": "USA", "fullname": "William P. Hobby Airport"},
    #     {"airport_code": "IAD", "lat": 38.9445, "lon": -77.4558, "name": "Washington Dulles International Airport", "city": "Washington", "state": "VA", "country": "USA", "fullname": "Washington Dulles International Airport"},
    #     {"airport_code": "LIT", "lat": 34.7294, "lon": -92.2243, "name": "Bill and Hillary Clinton National Airport", "city": "Little Rock", "state": "AR", "country": "USA", "fullname": "Bill and Hillary Clinton National Airport"},
    #     {"airport_code": "MDW", "lat": 41.7868, "lon": -87.7522, "name": "Chicago Midway International Airport", "city": "Chicago", "state": "IL", "country": "USA", "fullname": "Chicago Midway International Airport"},
    #     {"airport_code": "ORD", "lat": 41.9786, "lon": -87.9048, "name": "O'Hare International Airport", "city": "Chicago", "state": "IL", "country": "USA", "fullname": "O'Hare International Airport"},
    #     {"airport_code": "OMA", "lat": 41.3032, "lon": -95.8940, "name": "Eppley Airfield", "city": "Omaha", "state": "NE", "country": "USA", "fullname": "Eppley Airfield"},
    #     {"airport_code": "MKE", "lat": 42.9472, "lon": -87.8966, "name": "Milwaukee Mitchell International Airport", "city": "Milwaukee", "state": "WI", "country": "USA", "fullname": "Milwaukee Mitchell International Airport"},
    #     {"airport_code": "LIH", "lat": 21.9760, "lon": -159.3389, "name": "Lihue Airport", "city": "Lihue", "state": "HI", "country": "USA", "fullname": "Lihue Airport"},
    #     {"airport_code": "STL", "lat": 38.7487, "lon": -90.3700, "name": "St. Louis Lambert International Airport", "city": "St. Louis", "state": "MO", "country": "USA", "fullname": "St. Louis Lambert International Airport"},
    #     {"airport_code": "OGG", "lat": 20.8986, "lon": -156.4305, "name": "Kahului Airport", "city": "Kahului", "state": "HI", "country": "USA", "fullname": "Kahului Airport"},
    #     {"airport_code": "LGB", "lat": 33.8177, "lon": -118.1516, "name": "Long Beach Airport", "city": "Long Beach", "state": "CA", "country": "USA", "fullname": "Long Beach Airport"},
    #     {"airport_code": "MCI", "lat": 39.2976, "lon": -94.7139, "name": "Kansas City International Airport", "city": "Kansas City", "state": "MO", "country": "USA", "fullname": "Kansas City International Airport"},
    #     {"airport_code": "MSY", "lat": 29.9934, "lon": -90.2580, "name": "Louis Armstrong New Orleans International Airport", "city": "New Orleans", "state": "LA", "country": "USA", "fullname": "Louis Armstrong New Orleans International Airport"},
    #     {"airport_code": "MSP", "lat": 44.8820, "lon": -93.2218, "name": "Minneapolis–Saint Paul International Airport", "city": "Minneapolis", "state": "MN", "country": "USA", "fullname": "Minneapolis–Saint Paul International Airport"},
    #     {"airport_code": "ORF", "lat": 36.8946, "lon": -76.2012, "name": "Norfolk International Airport", "city": "Norfolk", "state": "VA", "country": "USA", "fullname": "Norfolk International Airport"},
    #     {"airport_code": "SMF", "lat": 38.6953, "lon": -121.5908, "name": "Sacramento International Airport", "city": "Sacramento", "state": "CA", "country": "USA", "fullname": "Sacramento International Airport"},
    #     {"airport_code": "MAF", "lat": 31.9425, "lon": -102.2019, "name": "Midland International Air and Space Port", "city": "Midland", "state": "TX", "country": "USA", "fullname": "Midland International Air and Space Port"},
    #     {"airport_code": "XNA", "lat": 36.2818, "lon": -94.3069, "name": "Northwest Arkansas National Airport", "city": "Bentonville", "state": "AR", "country": "USA", "fullname": "Northwest Arkansas National Airport"},
    #     {"airport_code": "PHX", "lat": 33.4343, "lon": -112.0097, "name": "Phoenix Sky Harbor International Airport", "city": "Phoenix", "state": "AZ", "country": "USA", "fullname": "Phoenix Sky Harbor International Airport"},
    #     {"airport_code": "SNA", "lat": 33.6762, "lon": -117.8676, "name": "John Wayne Airport", "city": "Santa Ana", "state": "CA", "country": "USA", "fullname": "John Wayne Airport"},
    #     {"airport_code": "PHL", "lat": 39.8719, "lon": -75.2411, "name": "Philadelphia International Airport", "city": "Philadelphia", "state": "PA", "country": "USA", "fullname": "Philadelphia International Airport"},
    #     {"airport_code": "PNS", "lat": 30.4734, "lon": -87.1866, "name": "Pensacola International Airport", "city": "Pensacola", "state": "FL", "country": "USA", "fullname": "Pensacola International Airport"},
    #     {"airport_code": "PSP", "lat": 33.8233, "lon": -116.5062, "name": "Palm Springs International Airport", "city": "Palm Springs", "state": "CA", "country": "USA", "fullname": "Palm Springs International Airport"},
    #     {"airport_code": "MCO", "lat": 28.4312, "lon": -81.3081, "name": "Orlando International Airport", "city": "Orlando", "state": "FL", "country": "USA", "fullname": "Orlando International Airport"},
    #     {"airport_code": "ONT", "lat": 34.0559, "lon": -117.6011, "name": "Ontario International Airport", "city": "Ontario", "state": "CA", "country": "USA", "fullname": "Ontario International Airport"},
    #     {"airport_code": "PIT", "lat": 40.4915, "lon": -80.2329, "name": "Pittsburgh International Airport", "city": "Pittsburgh", "state": "PA", "country": "USA", "fullname": "Pittsburgh International Airport"},
    #     {"airport_code": "PDX", "lat": 45.5887, "lon": -122.5975, "name": "Portland International Airport", "city": "Portland", "state": "OR", "country": "USA", "fullname": "Portland International Airport"},
    #     {"airport_code": "SJU", "lat": 18.4394, "lon": -66.0018, "name": "Luis Muñoz Marín International Airport", "city": "San Juan", "state": "PR", "country": "USA", "fullname": "Luis Muñoz Marín International Airport"},
    #     {"airport_code": "PWM", "lat": 43.6462, "lon": -70.3095, "name": "Portland International Jetport", "city": "Portland", "state": "ME", "country": "USA", "fullname": "Portland International Jetport"},
    #     {"airport_code": "RIC", "lat": 37.5052, "lon": -77.3197, "name": "Richmond International Airport", "city": "Richmond", "state": "VA", "country": "USA", "fullname": "Richmond International Airport"},
    #     {"airport_code": "RST", "lat": 43.9082, "lon": -92.5000, "name": "Rochester International Airport", "city": "Rochester", "state": "MN", "country": "USA", "fullname": "Rochester International Airport"},
    #     {"airport_code": "RNO", "lat": 39.4991, "lon": -119.7682, "name": "Reno-Tahoe International Airport", "city": "Reno", "state": "NV", "country": "USA", "fullname": "Reno-Tahoe International Airport"},
    #     {"airport_code": "RDU", "lat": 35.8801, "lon": -78.7880, "name": "Raleigh-Durham International Airport", "city": "Raleigh", "state": "NC", "country": "USA", "fullname": "Raleigh-Durham International Airport"},
    #     {"airport_code": "SLC", "lat": 40.7899, "lon": -111.9791, "name": "Salt Lake City International Airport", "city": "Salt Lake City", "state": "UT", "country": "USA", "fullname": "Salt Lake City International Airport"},
    #     {"airport_code": "SJC", "lat": 37.3639, "lon": -121.9289, "name": "Norman Y. Mineta San Jose International Airport", "city": "San Jose", "state": "CA", "country": "USA", "fullname": "Norman Y. Mineta San Jose International Airport"},
    #     {"airport_code": "SMF", "lat": 38.6953, "lon": -121.5908, "name": "Sacramento International Airport", "city": "Sacramento", "state": "CA", "country": "USA", "fullname": "Sacramento International Airport"},
    #     {"airport_code": "SAT", "lat": 29.5337, "lon": -98.4698, "name": "San Antonio International Airport", "city": "San Antonio", "state": "TX", "country": "USA", "fullname": "San Antonio International Airport"},
    #     {"airport_code": "SAN", "lat": 32.7336, "lon": -117.1897, "name": "San Diego International Airport", "city": "San Diego", "state": "CA", "country": "USA", "fullname": "San Diego International Airport"},
    #     {"airport_code": "SPN", "lat": 15.1190, "lon": 145.7290, "name": "Saipan International Airport", "city": "Saipan", "state": "MP", "country": "USA", "fullname": "Saipan International Airport"},
    #     {"airport_code": "SDF", "lat": 38.1740, "lon": -85.7360, "name": "Louisville Muhammad Ali International Airport", "city": "Louisville", "state": "KY", "country": "USA", "fullname": "Louisville Muhammad Ali International Airport"},
    #     {"airport_code": "SAV", "lat": 32.1276, "lon": -81.2023, "name": "Savannah/Hilton Head International Airport", "city": "Savannah", "state": "GA", "country": "USA", "fullname": "Savannah/Hilton Head International Airport"},
    #     {"airport_code": "MSN", "lat": 43.1399, "lon": -89.3375, "name": "Dane County Regional Airport", "city": "Madison", "state": "WI", "country": "USA", "fullname": "Dane County Regional Airport"},
    #     {"airport_code": "PVD", "lat": 41.7240, "lon": -71.4283, "name": "Rhode Island T. F. Green International Airport", "city": "Providence", "state": "RI", "country": "USA", "fullname": "Rhode Island T. F. Green International Airport"},
    #     {"airport_code": "RSW", "lat": 26.5362, "lon": -81.7552, "name": "Southwest Florida International Airport", "city": "Fort Myers", "state": "FL", "country": "USA", "fullname": "Southwest Florida International Airport"},
    #     {"airport_code": "TUS", "lat": 32.1161, "lon": -110.9410, "name": "Tucson International Airport", "city": "Tucson", "state": "AZ", "country": "USA", "fullname": "Tucson International Airport"},
    #     {"airport_code": "TLH", "lat": 30.3965, "lon": -84.3503, "name": "Tallahassee International Airport", "city": "Tallahassee", "state": "FL", "country": "USA", "fullname": "Tallahassee International Airport"},
    #     {"airport_code": "SEA", "lat": 47.4502, "lon": -122.3088, "name": "Seattle-Tacoma International Airport", "city": "Seattle", "state": "WA", "country": "USA", "fullname": "Seattle-Tacoma International Airport"},
    #     {"airport_code": "BRW", "lat": 71.2854, "lon": -156.7667, "name": "Wiley Post–Will Rogers Memorial Airport", "city": "Barrow", "state": "AK", "country": "USA", "fullname": "Wiley Post–Will Rogers Memorial Airport"},
    #     {"airport_code": "TUL", "lat": 36.1984, "lon": -95.8881, "name": "Tulsa International Airport", "city": "Tulsa", "state": "OK", "country": "USA", "fullname": "Tulsa International Airport"},
    #     {"airport_code": "MEM", "lat": 35.0424, "lon": -89.9767, "name": "Memphis International Airport", "city": "Memphis", "state": "TN", "country": "USA", "fullname": "Memphis International Airport"},
    #     {"airport_code": "TYS", "lat": 35.8110, "lon": -83.9996, "name": "McGhee Tyson Airport", "city": "Knoxville", "state": "TN", "country": "USA", "fullname": "McGhee Tyson Airport"},
    #     {"airport_code": "MIA", "lat": 25.7932, "lon": -80.2906, "name": "Miami International Airport", "city": "Miami", "state": "FL", "country": "USA", "fullname": "Miami International Airport"}
    # ]
    airport_codes = [
            {"airport_code": "SVO", "lat": 55.9726, "lon": 37.4146, "name": "Шереметьево", "city": "Москва", "state": "МО", "country": "Россия", "fullname": "Международный аэропорт Шереметьево"},
            {"airport_code": "DME", "lat": 55.4103, "lon": 37.9026, "name": "Домодедово", "city": "Москва", "state": "МО", "country": "Россия", "fullname": "Международный аэропорт Домодедово"},
            {"airport_code": "VKO", "lat": 55.5983, "lon": 37.2615, "name": "Внуково", "city": "Москва", "state": "МО", "country": "Россия", "fullname": "Международный аэропорт Внуково"},
            {"airport_code": "LED", "lat": 59.8003, "lon": 30.2625, "name": "Пулково", "city": "Санкт-Петербург", "state": "ЛО", "country": "Россия", "fullname": "Международный аэропорт Пулково"},
            {"airport_code": "AER", "lat": 43.4499, "lon": 39.9566, "name": "Сочи", "city": "Сочи", "state": "КК", "country": "Россия", "fullname": "Международный аэропорт Сочи"},
            {"airport_code": "KZN", "lat": 55.6063, "lon": 49.2787, "name": "Казань", "city": "Казань", "state": "РТ", "country": "Россия", "fullname": "Международный аэропорт Казань"},
            {"airport_code": "OVB", "lat": 55.0126, "lon": 82.6507, "name": "Толмачево", "city": "Новосибирск", "state": "НСО", "country": "Россия", "fullname": "Международный аэропорт Толмачево"},
            {"airport_code": "SVX", "lat": 56.7431, "lon": 60.8027, "name": "Кольцово", "city": "Екатеринбург", "state": "СО", "country": "Россия", "fullname": "Международный аэропорт Кольцово"},
            {"airport_code": "ROV", "lat": 47.4939, "lon": 39.9244, "name": "Платов", "city": "Ростов-на-Дону", "state": "РО", "country": "Россия", "fullname": "Международный аэропорт Платов"},
            {"airport_code": "KRR", "lat": 45.0346, "lon": 39.1705, "name": "Пашковский", "city": "Краснодар", "state": "КК", "country": "Россия", "fullname": "Международный аэропорт Пашковский"},
            {"airport_code": "UFA", "lat": 54.5575, "lon": 55.8742, "name": "Уфа", "city": "Уфа", "state": "РБ", "country": "Россия", "fullname": "Международный аэропорт Уфа"},
            {"airport_code": "KGD", "lat": 54.8898, "lon": 20.5927, "name": "Храброво", "city": "Калининград", "state": "КО", "country": "Россия", "fullname": "Международный аэропорт Храброво"},
            {"airport_code": "MRV", "lat": 44.2251, "lon": 43.0819, "name": "Минеральные Воды", "city": "Минеральные Воды", "state": "СК", "country": "Россия", "fullname": "Международный аэропорт Минеральные Воды"},
            {"airport_code": "IKT", "lat": 52.2669, "lon": 104.3887, "name": "Иркутск", "city": "Иркутск", "state": "ИО", "country": "Россия", "fullname": "Международный аэропорт Иркутск"},
            {"airport_code": "VOG", "lat": 48.7824, "lon": 44.3455, "name": "Гумрак", "city": "Волгоград", "state": "ВО", "country": "Россия", "fullname": "Международный аэропорт Гумрак"},
            {"airport_code": "KHV", "lat": 48.5282, "lon": 135.1889, "name": "Новый", "city": "Хабаровск", "state": "ХК", "country": "Россия", "fullname": "Международный аэропорт Хабаровск-Новый"},
            {"airport_code": "VVO", "lat": 43.3979, "lon": 132.1479, "name": "Владивосток", "city": "Владивосток", "state": "ПК", "country": "Россия", "fullname": "Международный аэропорт Владивосток"},
            {"airport_code": "SGC", "lat": 61.3428, "lon": 73.4018, "name": "Сургут", "city": "Сургут", "state": "ХМАО", "country": "Россия", "fullname": "Международный аэропорт Сургут"},
            {"airport_code": "GOJ", "lat": 56.2301, "lon": 43.7871, "name": "Стригино", "city": "Нижний Новгород", "state": "НО", "country": "Россия", "fullname": "Международный аэропорт Стригино"},
            {"airport_code": "CEK", "lat": 55.3052, "lon": 61.5054, "name": "Баландино", "city": "Челябинск", "state": "ЧО", "country": "Россия", "fullname": "Международный аэропорт Баландино"},
            {"airport_code": "MMK", "lat": 53.3914, "lon": 58.7603, "name": "Магнитогорск", "city": "Магнитогорск", "state": "ЧО", "country": "Россия", "fullname": "Международный аэропорт Магнитогорск"},
            {"airport_code": "BAX", "lat": 53.3639, "lon": 83.5386, "name": "Барнаул", "city": "Барнаул", "state": "АК", "country": "Россия", "fullname": "Международный аэропорт имени Г.С. Титова"},
            {"airport_code": "PKC", "lat": 53.1664, "lon": 158.4535, "name": "Елизово", "city": "Петропавловск-Камчатский", "state": "КК", "country": "Россия", "fullname": "Международный аэропорт Петропавловск-Камчатский"},
            {"airport_code": "TJM", "lat": 57.1896, "lon": 65.3243, "name": "Рощино", "city": "Тюмень", "state": "ТО", "country": "Россия", "fullname": "Международный аэропорт Рощино"},
            {"airport_code": "AAQ", "lat": 44.9000, "lon": 37.3167, "name": "Витязево", "city": "Анапа", "state": "КК", "country": "Россия", "fullname": "Международный аэропорт Анапа-Витязево"},
            {"airport_code": "ARH", "lat": 64.6000, "lon": 40.7167, "name": "Талаги", "city": "Архангельск", "state": "АО", "country": "Россия", "fullname": "Международный аэропорт Архангельск"},
            {"airport_code": "ASF", "lat": 46.2833, "lon": 48.0063, "name": "Нариманово", "city": "Астрахань", "state": "АО", "country": "Россия", "fullname": "Международный аэропорт Астрахань"},
            {"airport_code": "PEE", "lat": 57.9167, "lon": 56.0211, "name": "Большое Савино", "city": "Пермь", "state": "ПК", "country": "Россия", "fullname": "Международный аэропорт Пермь"},
            {"airport_code": "MCX", "lat": 42.8167, "lon": 47.6523, "name": "Уйташ", "city": "Махачкала", "state": "РД", "country": "Россия", "fullname": "Международный аэропорт Махачкала"},
            {"airport_code": "YKS", "lat": 62.0933, "lon": 129.7717, "name": "Якутск", "city": "Якутск", "state": "РС(Я)", "country": "Россия", "fullname": "Международный аэропорт Якутск"},
            {"airport_code": "OMS", "lat": 54.9667, "lon": 73.3167, "name": "Центральный", "city": "Омск", "state": "ОО", "country": "Россия", "fullname": "Международный аэропорт Омск-Центральный"},
            {"airport_code": "GDX", "lat": 59.9100, "lon": 150.7200, "name": "Сокол", "city": "Магадан", "state": "МО", "country": "Россия", "fullname": "Международный аэропорт Магадан-Сокол"},
            {"airport_code": "STW", "lat": 45.1092, "lon": 42.1128, "name": "Шпаковское", "city": "Ставрополь", "state": "СК", "country": "Россия", "fullname": "Международный аэропорт Ставрополь"},
            {"airport_code": "REN", "lat": 51.5833, "lon": 46.0667, "name": "Центральный", "city": "Оренбург", "state": "ОО", "country": "Россия", "fullname": "Международный аэропорт Оренбург"},
            {"airport_code": "TOF", "lat": 56.5000, "lon": 84.9667, "name": "Богашёво", "city": "Томск", "state": "ТО", "country": "Россия", "fullname": "Международный аэропорт Томск"},
            {"airport_code": "RTW", "lat": 51.5667, "lon": 46.0667, "name": "Центральный", "city": "Саратов", "state": "СО", "country": "Россия", "fullname": "Международный аэропорт Гагарин"},
            {"airport_code": "KJA", "lat": 56.1728, "lon": 92.4933, "name": "Емельяново", "city": "Красноярск", "state": "КК", "country": "Россия", "fullname": "Международный аэропорт Красноярск"},
            {"airport_code": "BQS", "lat": 50.4258, "lon": 127.4103, "name": "Игнатьево", "city": "Благовещенск", "state": "АО", "country": "Россия", "fullname": "Международный аэропорт Благовещенск"},
            {"airport_code": "UUD", "lat": 51.8067, "lon": 107.4383, "name": "Байкал", "city": "Улан-Удэ", "state": "РБ", "country": "Россия", "fullname": "Международный аэропорт Байкал"},
            {"airport_code": "DYR", "lat": 64.7350, "lon": 177.7417, "name": "Угольный", "city": "Анадырь", "state": "ЧАО", "country": "Россия", "fullname": "Международный аэропорт Анадырь"},
            {"airport_code": "NOJ", "lat": 63.1833, "lon": 75.2700, "name": "Ноябрьск", "city": "Ноябрьск", "state": "ЯНАО", "country": "Россия", "fullname": "Аэропорт Ноябрьск"},
            {"airport_code": "NUX", "lat": 65.9500, "lon": 78.3667, "name": "Новый Уренгой", "city": "Новый Уренгой", "state": "ЯНАО", "country": "Россия", "fullname": "Международный аэропорт Новый Уренгой"},
            {"airport_code": "CSY", "lat": 56.1500, "lon": 43.0667, "name": "Чебоксары", "city": "Чебоксары", "state": "ЧР", "country": "Россия", "fullname": "Международный аэропорт Чебоксары"},
            {"airport_code": "PEZ", "lat": 53.1167, "lon": 45.0167, "name": "Терновка", "city": "Пенза", "state": "ПО", "country": "Россия", "fullname": "Международный аэропорт Пенза"},
            {"airport_code": "ULV", "lat": 54.2683, "lon": 48.2267, "name": "Баратаевка", "city": "Ульяновск", "state": "УО", "country": "Россия", "fullname": "Международный аэропорт Ульяновск-Баратаевка"},
            {"airport_code": "IJK", "lat": 56.8300, "lon": 53.4500, "name": "Ижевск", "city": "Ижевск", "state": "УР", "country": "Россия", "fullname": "Международный аэропорт Ижевск"},
            {"airport_code": "NYM", "lat": 65.4811, "lon": 72.6989, "name": "Надым", "city": "Надым", "state": "ЯНАО", "country": "Россия", "fullname": "Аэропорт Надым"},
            {"airport_code": "UCT", "lat": 43.3883, "lon": 132.1533, "name": "Кневичи", "city": "Уссурийск", "state": "ПК", "country": "Россия", "fullname": "Аэропорт Кневичи"},
            {"airport_code": "SCW", "lat": 61.6700, "lon": 50.8452, "name": "Сыктывкар", "city": "Сыктывкар", "state": "РК", "country": "Россия", "fullname": "Международный аэропорт Сыктывкар"},
            {"airport_code": "KYZ", "lat": 51.6694, "lon": 94.4006, "name": "Кызыл", "city": "Кызыл", "state": "РТ", "country": "Россия", "fullname": "Аэропорт Кызыл"},
            {"airport_code": "KEJ", "lat": 55.2700, "lon": 86.1072, "name": "Кемерово", "city": "Кемерово", "state": "КО", "country": "Россия", "fullname": "Международный аэропорт имени А.А. Леонова"},
            {"airport_code": "GRV", "lat": 43.3221, "lon": 45.0254, "name": "Грозный", "city": "Грозный", "state": "ЧР", "country": "Россия", "fullname": "Международный аэропорт Грозный"},
            {"airport_code": "NAL", "lat": 43.5129, "lon": 43.6366, "name": "Нальчик", "city": "Нальчик", "state": "КБР", "country": "Россия", "fullname": "Международный аэропорт Нальчик"},
            {"airport_code": "MQF", "lat": 43.8224, "lon": 44.6083, "name": "Магас", "city": "Магас", "state": "РИ", "country": "Россия", "fullname": "Аэропорт Магас"},
            {"airport_code": "VLK", "lat": 67.4886, "lon": 63.9931, "name": "Воркута", "city": "Воркута", "state": "РК", "country": "Россия", "fullname": "Аэропорт Воркута"},
            {"airport_code": "USK", "lat": 65.4456, "lon": 52.4336, "name": "Усинск", "city": "Усинск", "state": "РК", "country": "Россия", "fullname": "Аэропорт Усинск"},
            {"airport_code": "UKX", "lat": 56.8560, "lon": 105.7303, "name": "Усть-Кут", "city": "Усть-Кут", "state": "ИО", "country": "Россия", "fullname": "Аэропорт Усть-Кут"},
            {"airport_code": "HTG", "lat": 71.9781, "lon": 102.4911, "name": "Хатанга", "city": "Хатанга", "state": "КК", "country": "Россия", "fullname": "Аэропорт Хатанга"},
            {"airport_code": "NSK", "lat": 69.3111, "lon": 87.3322, "name": "Норильск", "city": "Норильск", "state": "КК", "country": "Россия", "fullname": "Аэропорт Норильск"},
            {"airport_code": "IGT", "lat": 43.4389, "lon": 41.9308, "name": "Игнатьево", "city": "Майкоп", "state": "РА", "country": "Россия", "fullname": "Аэропорт Майкоп"},
            {"airport_code": "ESL", "lat": 46.3734, "lon": 44.3314, "name": "Элиста", "city": "Элиста", "state": "РК", "country": "Россия", "fullname": "Аэропорт Элиста"},
            {"airport_code": "SLY", "lat": 66.5972, "lon": 66.6110, "name": "Салехард", "city": "Салехард", "state": "ЯНАО", "country": "Россия", "fullname": "Аэропорт Салехард"},
            {"airport_code": "TYD", "lat": 55.2783, "lon": 124.7353, "name": "Тында", "city": "Тында", "state": "АО", "country": "Россия", "fullname": "Аэропорт Тында"},
            {"airport_code": "CYX", "lat": 68.7406, "lon": 161.3378, "name": "Певек", "city": "Певек", "state": "ЧАО", "country": "Россия", "fullname": "Аэропорт Певек"},
            {"airport_code": "CNN", "lat": 67.4691, "lon": 136.6494, "name": "Чокурдах", "city": "Чокурдах", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Чокурдах"},
            {"airport_code": "ULK", "lat": 60.7200, "lon": 114.8250, "name": "Ленск", "city": "Ленск", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Ленск"},
            {"airport_code": "MJZ", "lat": 62.5347, "lon": 114.0395, "name": "Мирный", "city": "Мирный", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Мирный"},
            {"airport_code": "CKH", "lat": 70.6231, "lon": 147.9019, "name": "Черский", "city": "Черский", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Черский"},
            {"airport_code": "KXK", "lat": 50.4094, "lon": 136.9342, "name": "Комсомольск-на-Амуре", "city": "Комсомольск-на-Амуре", "state": "ХК", "country": "Россия", "fullname": "Аэропорт Хурба"},
            {"airport_code": "GDG", "lat": 51.5167, "lon": 103.5667, "name": "Магдагачи", "city": "Магдагачи", "state": "АО", "country": "Россия", "fullname": "Аэропорт Магдагачи"},
            {"airport_code": "DEE", "lat": 68.6953, "lon": 112.8017, "name": "Удачный", "city": "Удачный", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Удачный"},
            {"airport_code": "KEM", "lat": 64.9500, "lon": 34.5833, "name": "Кемь", "city": "Кемь", "state": "РК", "country": "Россия", "fullname": "Аэропорт Кемь"},
            {"airport_code": "KSZ", "lat": 61.2357, "lon": 46.6971, "name": "Котлас", "city": "Котлас", "state": "АО", "country": "Россия", "fullname": "Аэропорт Котлас"},
            {"airport_code": "NFG", "lat": 61.1083, "lon": 72.6500, "name": "Нефтеюганск", "city": "Нефтеюганск", "state": "ХМАО", "country": "Россия", "fullname": "Аэропорт Нефтеюганск"},
            {"airport_code": "OHO", "lat": 59.4100, "lon": 143.0667, "name": "Охотск", "city": "Охотск", "state": "ХК", "country": "Россия", "fullname": "Аэропорт Охотск"},
            {"airport_code": "RYB", "lat": 58.1042, "lon": 38.9294, "name": "Рыбинск", "city": "Рыбинск", "state": "ЯО", "country": "Россия", "fullname": "Аэропорт Староселье"},
            {"airport_code": "URS", "lat": 51.7500, "lon": 36.2958, "name": "Курск", "city": "Курск", "state": "КО", "country": "Россия", "fullname": "Аэропорт Курск-Восточный"},
            {"airport_code": "VUS", "lat": 60.7833, "lon": 46.3000, "name": "Великий Устюг", "city": "Великий Устюг", "state": "ВО", "country": "Россия", "fullname": "Аэропорт Великий Устюг"},
            {"airport_code": "VGD", "lat": 59.2833, "lon": 39.9500, "name": "Вологда", "city": "Вологда", "state": "ВО", "country": "Россия", "fullname": "Аэропорт Вологда"},
            {"airport_code": "OSW", "lat": 54.2333, "lon": 36.6000, "name": "Орск", "city": "Орск", "state": "ОО", "country": "Россия", "fullname": "Аэропорт Орск"},
            {"airport_code": "PEX", "lat": 65.1167, "lon": 57.1333, "name": "Печора", "city": "Печора", "state": "РК", "country": "Россия", "fullname": "Аэропорт Печора"},
            {"airport_code": "EYK", "lat": 63.6833, "lon": 66.6833, "name": "Белоярский", "city": "Белоярский", "state": "ХМАО", "country": "Россия", "fullname": "Аэропорт Белоярский"},
            {"airport_code": "BZK", "lat": 53.2141, "lon": 34.1764, "name": "Брянск", "city": "Брянск", "state": "БО", "country": "Россия", "fullname": "Аэропорт Брянск"},
            {"airport_code": "SWT", "lat": 60.7667, "lon": 77.6667, "name": "Стрежевой", "city": "Стрежевой", "state": "ТО", "country": "Россия", "fullname": "Аэропорт Стрежевой"},
            {"airport_code": "ULY", "lat": 54.4000, "lon": 48.8000, "name": "Ульяновск-Восточный", "city": "Ульяновск", "state": "УО", "country": "Россия", "fullname": "Аэропорт Ульяновск-Восточный"},
            {"airport_code": "HTA", "lat": 52.0267, "lon": 113.3050, "name": "Чита", "city": "Чита", "state": "ЗК", "country": "Россия", "fullname": "Аэропорт Кадала"},
            {"airport_code": "IAA", "lat": 67.4372, "lon": 86.6217, "name": "Игарка", "city": "Игарка", "state": "КК", "country": "Россия", "fullname": "Аэропорт Игарка"},
            {"airport_code": "KVX", "lat": 58.5033, "lon": 49.3483, "name": "Киров", "city": "Киров", "state": "КО", "country": "Россия", "fullname": "Аэропорт Победилово"},
            {"airport_code": "JOK", "lat": 56.4833, "lon": 44.0500, "name": "Йошкар-Ола", "city": "Йошкар-Ола", "state": "РМЭ", "country": "Россия", "fullname": "Аэропорт Йошкар-Ола"},
            {"airport_code": "KLF", "lat": 54.5333, "lon": 36.3667, "name": "Калуга", "city": "Калуга", "state": "КО", "country": "Россия", "fullname": "Международный аэропорт Калуга"},
            {"airport_code": "TBW", "lat": 52.8067, "lon": 41.4825, "name": "Тамбов", "city": "Тамбов", "state": "ТО", "country": "Россия", "fullname": "Аэропорт Тамбов-Донское"},
            {"airport_code": "TLK", "lat": 59.8764, "lon": 111.0444, "name": "Талакан", "city": "Талакан", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Талакан"},
            {"airport_code": "IRM", "lat": 63.1833, "lon": 64.4403, "name": "Игрим", "city": "Игрим", "state": "ХМАО", "country": "Россия", "fullname": "Аэропорт Игрим"},
            {"airport_code": "OVS", "lat": 61.3267, "lon": 63.6017, "name": "Советский", "city": "Советский", "state": "ХМАО", "country": "Россия", "fullname": "Аэропорт Советский"},
            {"airport_code": "URJ", "lat": 60.1167, "lon": 64.8333, "name": "Урай", "city": "Урай", "state": "ХМАО", "country": "Россия", "fullname": "Аэропорт Урай"},
            {"airport_code": "UEN", "lat": 56.9333, "lon": 62.7000, "name": "Туринск", "city": "Туринск", "state": "СО", "country": "Россия", "fullname": "Аэропорт Туринск"},
            {"airport_code": "EKS", "lat": 49.2100, "lon": 142.7000, "name": "Южно-Сахалинск", "city": "Южно-Сахалинск", "state": "СО", "country": "Россия", "fullname": "Аэропорт Хомутово"},
            {"airport_code": "DHG", "lat": 44.5583, "lon": 135.0139, "name": "Дальнегорск", "city": "Дальнегорск", "state": "ПК", "country": "Россия", "fullname": "Аэропорт Дальнегорск"},
            {"airport_code": "GVN", "lat": 45.3167, "lon": 147.6833, "name": "Южно-Курильск", "city": "Южно-Курильск", "state": "СО", "country": "Россия", "fullname": "Аэропорт Менделеево"},
            {"airport_code": "OHH", "lat": 53.5167, "lon": 142.1667, "name": "Оха", "city": "Оха", "state": "СО", "country": "Россия", "fullname": "Аэропорт Оха"},
            {"airport_code": "KBL", "lat": 48.4167, "lon": 135.1667, "name": "Комсомольск-на-Амуре", "city": "Комсомольск-на-Амуре", "state": "ХК", "country": "Россия", "fullname": "Аэропорт Комсомольск-на-Амуре"},
            {"airport_code": "GDZ", "lat": 44.5825, "lon": 38.0158, "name": "Геленджик", "city": "Геленджик", "state": "КК", "country": "Россия", "fullname": "Аэропорт Геленджик"},
            {"airport_code": "BCX", "lat": 43.2983, "lon": 132.1522, "name": "Белая Гора", "city": "Белая Гора", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Белая Гора"},
            {"airport_code": "TII", "lat": 71.6981, "lon": 128.9031, "name": "Тикси", "city": "Тикси", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Тикси"},
            {"airport_code": "CKL", "lat": 70.2833, "lon": 163.9833, "name": "Чокурдах", "city": "Чокурдах", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Чокурдах"},
            {"airport_code": "ZKP", "lat": 65.7167, "lon": 150.7000, "name": "Зырянка", "city": "Зырянка", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Зырянка"},
            {"airport_code": "SUY", "lat": 62.1833, "lon": 117.6333, "name": "Сунтар", "city": "Сунтар", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Сунтар"},
            {"airport_code": "UKG", "lat": 70.0108, "lon": 135.6456, "name": "Усть-Куйга", "city": "Усть-Куйга", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Усть-Куйга"},
            {"airport_code": "VRI", "lat": 67.5481, "lon": 133.3900, "name": "Верхневилюйск", "city": "Верхневилюйск", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Верхневилюйск"},
            {"airport_code": "SEK", "lat": 67.4703, "lon": 153.7364, "name": "Среднеколымск", "city": "Среднеколымск", "state": "РС(Я)", "country": "Россия", "fullname": "Аэропорт Среднеколымск"},
    ]
    return airport_codes

def generate_planes(airlines):
    """Генерирует данные о самолетах для каждой авиакомпании"""
    planes = []

    for airline in airlines:
        airline_code = airline["airline_code"]
        num_planes = random.randint(10, 30)  # Случайное количество самолетов для авиакомпании

        for i in range(num_planes):
            # Все самолеты в данных - Boeing 737-700 с 143 местами
            plane_type = "BOEING 737-700"
            seats = 143

            # Годы производства из реальных данных (2000-2011)
            manufacture_year = random.randint(2000, 2011)

            # Генерация бортового номера в формате из данных
            # Форматы из данных: NxxxWN, NxxxxA, NxxxxB и т.д.
            prefix = "N"
            middle = random.randint(100, 9999)
            suffix_options = ["WN", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", ""]
            suffix = random.choice(suffix_options)

            # Для некоторых авиакомпаний специальные суффиксы
            if airline_code == "OO":
                suffix = random.choice(["A", "B", "C", "D", "E", "F", "G", "H"])
            elif airline_code == "YX":
                suffix = "WN"

            tail_num = f"{prefix}{middle}{suffix}"

            planes.append({
                "manufacture_year": manufacture_year,
                "tail_num": tail_num,
                "number_of_seats": seats,
                "plane_type": plane_type,
                "airline_code": airline_code
            })

    return planes

def calculate_distance(lat1, lon1, lat2, lon2):
    """Рассчитывает приблизительное расстояние между двумя точками в км"""
    from math import radians, cos, sin, asin, sqrt

    # Преобразование градусов в радианы
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Формула гаверсинуса
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Радиус Земли в км

    return int(c * r)

def generate_flights(airlines, airports, planes, num_flights, start_date, end_date):
    """Генерирует данные о полетах"""
    flights = []
    delays = []

    # Создаем словарь самолетов по авиакомпаниям для быстрого доступа
    airline_planes = {}
    for plane in planes:
        if plane["airline_code"] not in airline_planes:
            airline_planes[plane["airline_code"]] = []
        airline_planes[plane["airline_code"]].append(plane["tail_num"])

    # Создание словарь расстояний между аэропортами
    distances = {}
    for ap1 in airports:
        for ap2 in airports:
            if ap1["airport_code"] != ap2["airport_code"]:
                key = f"{ap1['airport_code']}-{ap2['airport_code']}"
                if "lat" in ap1 and "lon" in ap1 and "lat" in ap2 and "lon" in ap2:
                    dist = calculate_distance(ap1["lat"], ap1["lon"], ap2["lat"], ap2["lon"])
                    distances[key] = dist
                else:
                    # Если нет координат, используем приблизительное расстояние
                    distances[key] = random.randint(500, 3000)

    # Создаем список дат
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = [date.to_pydatetime() for date in date_range]

    flight_id = 1
    for _ in range(num_flights):
        # Выбираем случайную авиакомпанию
        airline = random.choice(airlines)
        airline_code = airline["airline_code"]

        if airline_code not in airline_planes or not airline_planes[airline_code]:
            continue  # Пропускаем, если у авиакомпании нет самолетов

        # Выбираем случайный самолет этой авиакомпании
        tail_num = random.choice(airline_planes[airline_code])

        # Выбираем аэропорты вылета и прилета
        origin = random.choice(airports)
        dest = random.choice([ap for ap in airports if ap["airport_code"] != origin["airport_code"]])

        # Получаем расстояние
        distance = distances.get(f"{origin['airport_code']}-{dest['airport_code']}", 1000)  # Значение по умолчанию 1000 км
        # distance = 0

        # Выбираем случайную дату и время
        flight_date = random.choice(dates)

        # Генерируем время вылета (чаще в утренние и вечерние часы)
        hour_weights = [1, 0.5, 0.5, 0.5, 1, 2, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 4, 3, 2, 1, 0.5]
        dep_hour = random.choices(range(24), weights=hour_weights)[0]
        dep_minute = random.choice(range(0, 60, 5))  # С шагом в 5 минут
        dep_time = f"{dep_hour:02d}:{dep_minute:02d}"

        # Оценка времени полета на основе расстояния (приблизительно 800 км/ч)
        flight_time_hours = distance / 800
        flight_time_minutes = int(flight_time_hours * 60)

        # Расчет времени прибытия
        arr_hour = (dep_hour + flight_time_hours) % 24
        arr_minute = (dep_minute + flight_time_minutes % 60) % 60
        arr_time = f"{int(arr_hour):02d}:{arr_minute:02d}"

        air_time = flight_time_minutes

        # Добавляем полет
        flights.append({
            "id": flight_id,
            "fl_date": flight_date,
            "airline_code": airline_code,
            "origin_airport": origin["airport_code"],
            "dest_airport": dest["airport_code"],
            "distance": distance,
            "tail_num": tail_num,
            "dep_time": dep_time,
            "arr_time": arr_time,
            "air_time": air_time
        })

        # Генерируем данные о задержке
        # Вероятность задержки зависит от авиакомпании, времени года и дня недели
        delay_prob = 0.2  # Базовая вероятность

        # Зимние месяцы - выше вероятность задержек
        if flight_date.month in [12, 1, 2]:
            delay_prob += 0.1

        # Летние месяцы - больше пассажиров
        if flight_date.month in [6, 7, 8]:
            delay_prob += 0.05

        # Выходные дни - больше рейсов
        if flight_date.weekday() >= 5:  # Суббота и воскресенье
            delay_prob += 0.05

        # Вероятность отмены рейса
        cancel_prob = 0.01

        # Выше вероятность для определенных авиакомпаний (для разнообразия данных)
        if airline_code in ["UT", "WZ", "A4"]:
            delay_prob += 0.1
            cancel_prob += 0.01

        # Утренние рейсы более пунктуальны
        if 5 <= dep_hour <= 10:
            delay_prob -= 0.05

        # Генерируем задержку или отмену
        is_cancelled = random.random() < cancel_prob

        if is_cancelled:
            delays.append({
                "id": flight_id,
                "dep_delay": None,
                "arr_delay": None,
                "cancelled": 1,
                "cancellation_code": random.choice(["A", "B", "C", "D"])  # A-авиакомпания, B-погода, C-нац.службы, D-техника
            })
        else:
            has_delay = random.random() < delay_prob

            if has_delay:
                # Генерируем задержку вылета (экспоненциальное распределение)
                # Большинство задержек небольшие, но некоторые могут быть значительными
                dep_delay = int(random.expovariate(1.0/20))  # Среднее значение ~20 минут

                # Но не слишком большие задержки
                dep_delay = min(dep_delay, 180)

                # Задержка прилета обычно коррелирует с задержкой вылета
                correlation = random.uniform(0.8, 1.2)  # Небольшое случайное отклонение
                arr_delay = int(dep_delay * correlation)

                # Учитываем возможность "нагнать время" в полете
                if random.random() < 0.3:  # 30% шанс сократить время
                    arr_delay = max(0, arr_delay - random.randint(5, 15))
            else:
                dep_delay = 0
                arr_delay = 0

            delays.append({
                "id": flight_id,
                "dep_delay": dep_delay,
                "arr_delay": arr_delay,
                "cancelled": 0,
                "cancellation_code": None
            })

        flight_id += 1

    return flights, delays

def insert_data(data, table_name):
    """Вставляет данные в указанную таблицу"""
    session = Session()
    try:
        if not data:
            print(f"Нет данных для вставки в таблицу {table_name}")
            return

        # Подготовка запроса
        if table_name == "airlines":
            for item in data:
                session.execute(
                    text("INSERT INTO public.airlines (airline_name, airline_code) "
                         "VALUES (:name, :code)"),
                    {"name": item.get("airline_name", ""), "code": item["airline_code"]}
                )
        elif table_name == "airports":
            for item in data:
                session.execute(
                    text("""
                    INSERT INTO public.airports (display_airport_name, airport_code, airport_city, airport_fullname,
                    airport_state, airport_country, latitude, longitude)
                    VALUES (:name, :code, :city, :fullname, :state, :country, :lat, :lon)
                    """),
                    {
                        "name": item.get("name", item.get("display_airport_name", "")),
                        "code": item["airport_code"],
                        "city": item.get("city", ""),
                        "fullname": item.get("fullname", ""),
                        "state": item.get("state", ""),
                        "country": item.get("country", ""),
                        "lat": item.get("lat", 0),
                        "lon": item.get("lon", 0)
                    }
                )
        elif table_name == "planes":
            for item in data:
                session.execute(
                    text("""
                    INSERT INTO public.planes (manufacture_year, tail_num, number_of_seats, plane_type, airline_code)
                    VALUES (:year, :tail_num, :seats, :type, :airline_code)
                    """),
                    {
                        "year": item["manufacture_year"], "tail_num": item["tail_num"],
                        "seats": item["number_of_seats"], "type": item["plane_type"],
                        "airline_code": item["airline_code"]
                    }
                )
        elif table_name == "flights":
            for item in data:
                session.execute(
                    text("""
                    INSERT INTO public.flights (id, fl_date, airline_code, origin_airport, dest_airport,
                    distance, tail_num, dep_time, arr_time, air_time)
                    VALUES (:id, :fl_date, :airline_code, :origin_airport, :dest_airport,
                    :distance, :tail_num, :dep_time, :arr_time, :air_time)
                    """),
                    {
                        "id": item["id"], "fl_date": item["fl_date"], "airline_code": item["airline_code"],
                        "origin_airport": item["origin_airport"], "dest_airport": item["dest_airport"],
                        "distance": item["distance"], "tail_num": item["tail_num"],
                        "dep_time": item["dep_time"], "arr_time": item["arr_time"], "air_time": item["air_time"]
                    }
                )
        elif table_name == "delay":
            for item in data:
                session.execute(
                    text("""
                    INSERT INTO public.delay (id, dep_delay, arr_delay, cancelled, cancellation_code)
                    VALUES (:id, :dep_delay, :arr_delay, :cancelled, :cancellation_code)
                    """),
                    {
                        "id": item["id"], "dep_delay": item["dep_delay"], "arr_delay": item["arr_delay"],
                        "cancelled": item["cancelled"], "cancellation_code": item["cancellation_code"]
                    }
                )
        elif table_name == "flight_airport":
            for item in data:
                session.execute(
                    text("""
                    INSERT INTO public.flight_airport (flight_id, airport_id, airport_type)
                    VALUES (:flight_id, :airport_id, :airport_type)

                    """),
                    {
                        "flight_id": item["flight_id"], "airport_id": item["airport_id"],
                        "airport_type": item["airport_type"]
                    }
                )

        session.commit()
        print(f"Данные успешно добавлены в таблицу {table_name}")
    except Exception as e:
        session.rollback()
        print(f"Ошибка при вставке данных в таблицу {table_name}: {str(e)}")
    finally:
        session.close()

def generate_flight_airports(flights, airports):
    """Генерирует данные для таблицы flight_airport"""
    flight_airports = []

    # Получаем идентификаторы аэропортов из базы данных
    session = Session()
    try:
        # Создаем словарь для быстрого поиска id аэропорта по коду
        airport_code_to_id = {}

        # Убедимся, что таблица аэропортов существует
        check_table = session.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'airports'
            )
        """)).scalar()

        if not check_table:
            print("Таблица аэропортов не существует, сначала создайте таблицу аэропортов")
            return []

        airport_query = session.execute(text("SELECT id, airport_code FROM public.airports"))
        for row in airport_query:
            airport_code_to_id[row[1]] = row[0]

        print(f"Найдено {len(airport_code_to_id)} аэропортов в базе данных")

        if not airport_code_to_id:
            print("Предупреждение: В базе данных не найдены аэропорты")

            # Если в базе нет аэропортов, вставим их и получим id
            print("Вставляем аэропорты в базу данных...")
            for airport in airports:
                result = session.execute(
                    text("""
                    INSERT INTO public.airports (id, display_airport_name, airport_code, airport_city, airport_fullname,
                    airport_state, airport_country, latitude, longitude)
                    VALUES (:id, :name, :code, :city, :fullname, :state, :country, :lat, :lon)
                    ON CONFLICT (airport_code) DO NOTHING
                    """),
                    {
                        "id": airport["id"],
                        "name": airport.get("name", ""),
                        "code": airport["airport_code"],
                        "city": airport.get("city", ""),
                        "fullname": airport.get("fullname", ""),
                        "state": airport.get("state", ""),
                        "country": airport.get("country", ""),
                        "lat": airport.get("lat", 0),
                        "lon": airport.get("lon", 0)
                    }
                )

                # Получаем id вставленного аэропорта
                id_result = result.first()
                if id_result:
                    airport_code_to_id[airport["airport_code"]] = id_result[0]

            session.commit()
            print(f"Вставлено {len(airport_code_to_id)} аэропортов")

            # Если все еще нет аэропортов, повторно запросим их из базы
            if not airport_code_to_id:
                airport_query = session.execute(text("SELECT id, airport_code FROM public.airports"))
                for row in airport_query:
                    airport_code_to_id[row[1]] = row[0]
                print(f"После вставки найдено {len(airport_code_to_id)} аэропортов")

        # Для каждого рейса создаем две записи: для аэропорта вылета и прилета
        for flight in flights:
            origin_code = flight["origin_airport"]
            dest_code = flight["dest_airport"]

            # Получаем id аэропортов по коду
            origin_id = airport_code_to_id.get(origin_code)
            dest_id = airport_code_to_id.get(dest_code)

            if origin_id is not None:
                flight_airports.append({
                    "flight_id": flight["id"],
                    "airport_id": origin_id,
                    "airport_type": "departure"
                })
            else:
                print(f"Не найден ID для аэропорта с кодом {origin_code}")

            if dest_id is not None:
                flight_airports.append({
                    "flight_id": flight["id"],
                    "airport_id": dest_id,
                    "airport_type": "arrival"
                })
            else:
                print(f"Не найден ID для аэропорта с кодом {dest_code}")
    except Exception as e:
        print(f"Ошибка при получении идентификаторов аэропортов: {str(e)}")
        session.rollback()
    finally:
        session.close()

    print(f"Создано {len(flight_airports)} записей для flight_airport")
    return flight_airports

def main():
    parser = argparse.ArgumentParser(description='Генерация тестовых данных для системы прогнозирования задержек авиарейсов')
    parser.add_argument('--flights', type=int, default=10000, help='Количество рейсов для генерации (по умолчанию: 1000)')
    parser.add_argument('--start_date', type=str, default='2024-01-01', help='Начальная дата в формате YYYY-MM-DD (по умолчанию: 2023-01-01)')
    parser.add_argument('--end_date', type=str, default='2024-12-31', help='Конечная дата в формате YYYY-MM-DD (по умолчанию: 2023-12-31)')

    args = parser.parse_args()

    print(f"Генерация {args.flights} рейсов с {args.start_date} по {args.end_date}")

    # Создание таблиц если они не существуют
    create_tables_if_not_exists()

    # Генерация и вставка данных
    airlines = generate_airlines()
    airports = generate_airports()
    planes = generate_planes(airlines)

    # Вставка базовых данных
    insert_data(airlines, "airlines")
    insert_data(airports, "airports")
    insert_data(planes, "planes")

    # Генерация и вставка данных о рейсах
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    flights, delays = generate_flights(airlines, airports, planes, args.flights, start_date, end_date)

    insert_data(flights, "flights")
    insert_data(delays, "delay")

    # Генерация и вставка данных для flight_airport
    flight_airports = generate_flight_airports(flights, airports)
    insert_data(flight_airports, "flight_airport")

    print(f"Генерация данных завершена. Создано {len(flights)} рейсов.")

if __name__ == "__main__":
    main()
