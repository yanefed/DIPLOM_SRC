import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

db_host = os.environ.get('POSTGRES_HOST', 'localhost')
db_port = os.environ.get('DATABASE_PORT', '5432')
db_user = os.environ.get('POSTGRES_USER', 'postgres')
db_password = os.environ.get('POSTGRES_PASSWORD', '0252')
db_name = os.environ.get('POSTGRES_DB', 'postgres')

connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)
Session = sessionmaker(bind=engine)
session = Session()


def measure_performance(plane, origin, destination, date_limit, num_measurements=3):
    execution_times = []
    rows_counts = []

    for i in range(1, 21):  # Iterate from 1 to 20 (increments of 4000)
        num_rows = i * 4000
        # Insert code to clear cache, if needed
        session.execute("DISCARD PLANS")

        measurement_times = []
        for _ in range(num_measurements):
            start_time = datetime.now()
            session.execute(text("CALL probability(:plane, :origin, :destination, :date_limit)"),
                            {'plane'      : plane, 'origin': origin,
                             'destination': destination, 'date_limit': date_limit})
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
            measurement_times.append(execution_time)

        avg_execution_time = sum(measurement_times) / num_measurements
        execution_times.append(avg_execution_time)
        rows_counts.append(num_rows)

        print(f"Number of Rows: {num_rows}, Avg Execution Time: {avg_execution_time} ms")

    return execution_times, rows_counts


if __name__ == "__main__":
    plane = 'N208WN'
    origin = 'SJC'
    destination = 'LAX'
    date_limit = '2024-06-01'
    num_measurements = 3  # Number of measurements to take for each row count

    execution_times, rows_counts = measure_performance(plane, origin, destination, date_limit, num_measurements)

    plt.figure(figsize=(10, 6))
    plt.plot(rows_counts, execution_times, marker='o')
    plt.xlabel('Number of Rows')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time vs. Number of Rows')
    plt.grid(True)

    # Adding x-axis labels for number of rows
    x_ticks = [i * 4000 for i in range(1, 21)]  # Label every 4000 rows
    plt.xticks(x_ticks, [f"{num_rows:,}" for num_rows in x_ticks])

    plt.show()

# import matplotlib.pyplot as plt
# from sqlalchemy import create_engine, text
# from datetime import datetime
# import os
#
# # Конфигурация подключения к базе данных
# db_host = os.environ.get('POSTGRES_HOST', 'vidicode.ru')
# db_port = os.environ.get('DATABASE_PORT', '5433')
# db_user = os.environ.get('POSTGRES_USER', 'dev')
# db_password = os.environ.get('POSTGRES_PASSWORD', 'welcome')
# db_name = os.environ.get('POSTGRES_DB', 'flight_delay')
#
# connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
# engine = create_engine(connection_string)
#
#
# def measure_performance(num_measurements=3):
#     execution_times = []
#
#     # Выполнить запрос для получения всех записей из таблицы flights
#     with engine.connect() as connection:
#         result = connection.execute("SELECT id, tail_num, origin_airport, dest_airport FROM flights")
#         flights = result.fetchall()
#
#     # Провести замер времени выполнения запроса для каждой записи
#     for flight in flights:
#         measurement_times = []
#         for _ in range(num_measurements):
#             start_time = datetime.now()
#             with engine.connect() as connection:
#                 connection.execute(text("CALL probability(:plane, :origin, :destination, :date_limit)"),
#                                    {'plane'      : flight[1], 'origin': flight[2],
#                                     'destination': flight[3], 'date_limit': '2024-06-01'})
#             end_time = datetime.now()
#             execution_time = (end_time - start_time).total_seconds() * 1000  # Преобразовать в миллисекунды
#             measurement_times.append(execution_time)
#
#         avg_execution_time = sum(measurement_times) / num_measurements
#         execution_times.append(avg_execution_time)
#
#         print(f"Flight ID: {flight[0]}, Avg Execution Time: {avg_execution_time} ms")
#
#     return execution_times
#
#
# if __name__ == "__main__":
#     num_measurements = 3  # Количество измерений для каждой записи
#
#     execution_times = measure_performance(num_measurements)
#
#     # Построить гистограмму времени выполнения запросов
#     plt.hist(execution_times, bins=20)
#     plt.xlabel('Execution Time (ms)')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Query Execution Times')
#     plt.grid(True)
#     plt.show()
