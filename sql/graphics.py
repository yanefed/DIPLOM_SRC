# -*- coding: utf-8 -*-
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import matplotlib as mpl
import matplotlib.font_manager as font_manager

# Настройка шрифтов для корректного отображения кириллицы
mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False  # Корректное отображение минуса

# Увеличиваем размер шрифта для лучшей читаемости
plt.rcParams['font.size'] = 12

# Подключение к базе данных
db_host = os.environ.get('POSTGRES_HOST', 'localhost')
db_port = os.environ.get('DATABASE_PORT', '5432')
db_user = os.environ.get('POSTGRES_USER', 'postgres')
db_password = os.environ.get('POSTGRES_PASSWORD', '0252')
db_name = os.environ.get('POSTGRES_DB', 'ru_postgres')

connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)
Session = sessionmaker(bind=engine)

# Подавление предупреждений
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

# Настройка стилей для графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')


def load_data(table_name='flight_data_for_visualization'):
    """
    Загружает данные о рейсах из базы данных PostgreSQL

    Args:
        table_name (str): Имя таблицы в базе данных (по умолчанию 'flight_data_for_visualization')

    Returns:
        list: Список словарей, где каждый словарь представляет строку данных
    """
    print(f"Загрузка данных из таблицы {table_name}...")

    session = Session()
    query = f"SELECT * FROM public.{table_name} LIMIT 100"

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


# Функция для базовой предобработки данных
def preprocess_data(df):
    """
    Выполняет базовую предобработку данных

    Args:
        df (DataFrame): Исходный DataFrame

    Returns:
        DataFrame: Обработанный DataFrame
    """
    print("Выполняется предобработка данных...")

    # Преобразование текстовых дат в datetime
    if 'FlightDate' in df.columns:
        df['FlightDate'] = pd.to_datetime(df['FlightDate'])

    # Заполнение пропусков в числовых колонках с задержками
    delay_columns = ['DepDelay', 'ArrDelay', 'CarrierDelay', 'WeatherDelay',
                     'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

    for col in delay_columns:
        if col in df.columns:
            # Заполняем пропуски нулями для нечисловых значений
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Создаем категориальные переменные для месяца и дня недели
    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype('category')

    if 'DayOfWeek' in df.columns:
        df['DayOfWeek'] = df['DayOfWeek'].astype('category')

    print("Предобработка данных завершена.")
    return df


def configure_plot_for_cyrillic(title=None, xlabel=None, ylabel=None, legend_loc='best'):
    """
    Настраивает график для корректного отображения кириллицы и оптимального размещения элементов

    Args:
        title (str): Заголовок графика
        xlabel (str): Подпись оси X
        ylabel (str): Подпись оси Y
        legend_loc (str): Расположение легенды
    """
    if title: plt.title(title, fontsize=14)
    if xlabel: plt.xlabel(xlabel, fontsize=12)
    if ylabel: plt.ylabel(ylabel, fontsize=12)
    if plt.gca().get_legend() is not None:
        plt.legend(loc=legend_loc, fontsize=11)


# Функция для анализа задержек по авиакомпаниям
def analyze_airline_delays(df):
    """
    Создает графики для анализа задержек по авиакомпаниям

    Args:
        df (DataFrame): DataFrame с данными о рейсах
    """
    print("Анализ задержек по авиакомпаниям...")

    # 1. Количество рейсов по авиакомпаниям
    plt.figure(figsize=(12, 6))
    airline_counts = df["Reporting_Airline"].value_counts()
    airline_counts.plot(kind='bar')
    configure_plot_for_cyrillic(
        title="Количество рейсов по авиакомпаниям",
        xlabel="Авиакомпания",
        ylabel="Количество рейсов"
    )
    plt.tight_layout(pad=2.0)  # Увеличиваем отступы для предотвращения обрезания текста
    plt.savefig('./img/airline_flight_counts.png', dpi=300,
                bbox_inches='tight')  # Увеличиваем разрешение и добавляем bbox_inches
    plt.close()

    # 2. Средние задержки по авиакомпаниям

    plt.figure(figsize=(15, 8))
    # Создаем DataFrame с задержками по авиакомпаниям
    airline_delays = df.groupby('Reporting_Airline')[['DepDelay', 'ArrDelay']].mean()

    # Переименовываем колонки на русский язык для легенды
    airline_delays = airline_delays.rename(columns={
        'DepDelay': 'Задержка вылета',
        'ArrDelay': 'Задержка прибытия'
    })

    # Сортируем по задержке вылета
    airline_delays.sort_values('Задержка вылета', ascending=False).plot.bar()

    configure_plot_for_cyrillic(
        title="Распределение задержек по разным авиакомпаниям",
        xlabel="Авиакомпании",  # Поменял местами, так как это ось X в графике
        ylabel="Задержка (мин)"  # Поменял местами, так как это ось Y в графике
    )
    plt.tight_layout(pad=2.0)
    plt.savefig('./img/airline_delay_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Анализ задержек по авиакомпаниям завершен.")


# Функция для анализа временных трендов задержек
def analyze_delay_trends(df):
    """
    Создает графики для анализа трендов задержек по времени

    Args:
        df (DataFrame): DataFrame с данными о рейсах
    """
    print("Анализ временных трендов задержек...")

    # Создаем поле Year-Month для группировки
    if 'Year' in df.columns and 'Month' in df.columns:
        df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
    elif 'FlightDate' in df.columns:
        df['YearMonth'] = df['FlightDate'].dt.strftime('%Y-%m')

    # 1. Средние задержки вылета и прибытия по месяцам
    monthly_delays = df.groupby('YearMonth')[['DepDelay', 'ArrDelay']].mean().reset_index()

    plt.figure(figsize=(15, 6))
    sns.lineplot(data=monthly_delays, x='YearMonth', y='DepDelay', label='Задержка вылета')
    sns.lineplot(data=monthly_delays, x='YearMonth', y='ArrDelay', label='Задержка прибытия')
    configure_plot_for_cyrillic(
        title="Средние задержки вылета и прибытия по времени",
        xlabel="Год-Месяц",
        ylabel="Средняя задержка (минуты)"
    )
    plt.xticks(rotation=45)
    plt.tight_layout(pad=2.0)
    plt.savefig('./img/monthly_delays.png')
    plt.close()

    # 2. Доля отмененных рейсов по месяцам
    if 'Cancelled' in df.columns:
        monthly_cancellations = df.groupby('YearMonth')['Cancelled'].mean().reset_index()

        plt.figure(figsize=(15, 6))
        sns.lineplot(data=monthly_cancellations, x='YearMonth', y='Cancelled')
        configure_plot_for_cyrillic(
            title="Доля отмененных рейсов по времени",
            xlabel="Год-Месяц",
            ylabel="Доля отмененных рейсов"
        )
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.savefig('./img/monthly_cancellations.png')
        plt.close()

    # 3. Распределение задержек по дням недели
    if 'DayOfWeek' in df.columns:
        day_mapping = {1: 'Понедельник', 2: 'Вторник', 3: 'Среда',
                       4: 'Четверг', 5: 'Пятница', 6: 'Суббота', 7: 'Воскресенье'}

        df['DayName'] = df['DayOfWeek'].map(day_mapping)
        day_delays = df.groupby('DayName')[['DepDelay', 'ArrDelay']].mean().reindex(
            ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
        )

        day_delays = day_delays.rename(columns={'DepDelay': 'Задержка вылета', 'ArrDelay': 'Задержка прибытия'})

        plt.figure(figsize=(12, 6))
        day_delays.plot.bar()
        configure_plot_for_cyrillic(
            title="Средние задержки по дням недели",
            xlabel="День недели",
            ylabel="Средняя задержка (минуты)"
        )
        plt.tight_layout(pad=2.0)
        plt.savefig('./img/delays_by_day_of_week.png')
        plt.close()

    print("Анализ временных трендов задержек завершен.")


# Функция для анализа задержек по аэропортам
def analyze_airport_delays(df):
    """
    Создает графики для анализа задержек по аэропортам

    Args:
        df (DataFrame): DataFrame с данными о рейсах
    """
    print("Анализ задержек по аэропортам...")

    # 1. Топ аэропортов по задержкам вылета
    if 'Origin' in df.columns and 'Dest' in df.columns:
        airport_delays = df.groupby(['Origin', 'Dest'])[['DepDelay', 'ArrDelay']].mean().reset_index()

        # Топ по задержкам вылета
        dep_delays = airport_delays.sort_values('DepDelay', ascending=False).head(20)
        plt.figure(figsize=(15, 6))
        sns.barplot(data=dep_delays, x='Origin', y='DepDelay', hue='Dest', dodge=False)
        configure_plot_for_cyrillic(
            title="Наибольшие средние задержки вылета (топ 20 аэропортов)",
            xlabel="Средняя задержка вылета (минуты)",
            ylabel="Аэропорт (отправление)"
        )
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.savefig('./img/top_departure_delays.png')
        plt.close()

        # Топ по задержкам прибытия
        arr_delays = airport_delays.sort_values('ArrDelay', ascending=False).head(20)
        plt.figure(figsize=(15, 6))
        sns.barplot(data=arr_delays, x='Origin', y='ArrDelay', hue='Dest', dodge=False)
        configure_plot_for_cyrillic(
            title="Наибольшие средние задержки прибытия (топ 20 аэропортов)",
            xlabel="Средняя задержка прибытия (минуты)",
            ylabel="Аэропорт (отправление)"
        )
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.savefig('./img/top_arrival_delays.png')
        plt.close()

    print("Анализ задержек по аэропортам завершен.")


# Функция для анализа причин задержек
def analyze_delay_causes(df):
    """
    Создает графики для анализа причин задержек

    Args:
        df (DataFrame): DataFrame с данными о рейсах
    """
    print("Анализ причин задержек...")

    # 1. Средние задержки по причинам (столбчатая диаграмма)
    delay_types = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

    # Проверяем, есть ли нужные столбцы в данных
    available_delay_types = [col for col in delay_types if col in df.columns]

    if available_delay_types:
        delay_averages = df[available_delay_types].mean().reset_index()
        delay_averages.columns = ['DelayType', 'AverageDelay']

        # Переименовываем типы задержек для более понятных подписей
        delay_mapping = {
            'CarrierDelay': 'Задержка авиакомпании',
            'WeatherDelay': 'Погодные условия',
            'NASDelay': 'Нац. система воздушного движения',
            'SecurityDelay': 'Безопасность',
            'LateAircraftDelay': 'Позднее прибытие ВС'
        }

        delay_averages['DelayType'] = delay_averages['DelayType'].map(
            lambda x: delay_mapping.get(x, x)
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(data=delay_averages, x='DelayType', y='AverageDelay', palette='tab10')
        configure_plot_for_cyrillic(
            title="Средняя задержка по причинам",
            xlabel="Тип задержки",
            ylabel="Средняя задержка (минуты)"
        )
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.savefig('./img/delay_causes_bar.png')
        plt.close()

        # 2. Круговая диаграмма причин задержек
        plt.figure(figsize=(10, 6))
        plt.pie(delay_averages['AverageDelay'], labels=delay_averages['DelayType'],
                autopct='%.1f%%', startangle=90)
        plt.title('Распределение задержек по причинам')
        plt.axis('equal')  # Обеспечивает круглую форму диаграммы
        plt.tight_layout(pad=2.0)
        plt.savefig('./img/delay_causes_pie.png')
        plt.close()

    # 3. Причины отмены рейсов
    if 'Cancelled' in df.columns and 'CancellationCode' in df.columns:
        cancelled_flights = df[df['Cancelled'] == 1]

        if not cancelled_flights.empty:
            cancellation_counts = cancelled_flights['CancellationCode'].value_counts()

            # Преобразование кодов отмены в удобочитаемые метки
            code_mapping = {
                'A': 'Авиакомпания',
                'B': 'Погода',
                'C': 'Нац. система возд. движения',
                'D': 'Безопасность'
            }

            # Создаем новый индекс с преобразованными метками
            new_index = [code_mapping.get(code, code) for code in cancellation_counts.index]
            cancellation_counts.index = new_index

            plt.figure(figsize=(10, 6))
            sns.barplot(x=cancellation_counts.index, y=cancellation_counts.values, palette='tab10')
            configure_plot_for_cyrillic(
                title="Наиболее распространенные причины отмены рейсов",
                xlabel="Причина отмены",
                ylabel="Количество (ед.)"
            )
            plt.tight_layout(pad=2.0)
            plt.savefig('./img/cancellation_reasons.png')
            plt.close()

    print("Анализ причин задержек завершен.")


# Функция для анализа корреляций
def analyze_correlations(df):
    """
    Создает тепловую карту корреляций между различными факторами

    Args:
        df (DataFrame): DataFrame с данными о рейсах
    """
    print("Анализ корреляций между факторами...")

    # Список возможных числовых колонок для анализа
    possible_columns = ['Distance', 'DepDelay', 'ArrDelay', 'TaxiIn', 'TaxiOut',
                        'AirTime', 'ActualElapsedTime', 'CRSElapsedTime']

    # Выбираем те, которые фактически присутствуют в данных
    numeric_columns = [col for col in possible_columns if col in df.columns]

    if len(numeric_columns) > 1:  # Нужно минимум 2 колонки для корреляции
        # Словарь для перевода названий колонок на русский
        column_names_ru = {
            'Distance': 'Расстояние',
            'DepDelay': 'Задержка вылета',
            'ArrDelay': 'Задержка прибытия',
            'TaxiIn': 'Руление на прилете',
            'TaxiOut': 'Руление на вылете',
            'AirTime': 'Время в воздухе',
            'ActualElapsedTime': 'Фактическое время',
            'CRSElapsedTime': 'Запланированное время'
        }

        # Вычисляем корреляционную матрицу
        corr_matrix = df[numeric_columns].corr()

        # Переименовываем индексы и колонки
        corr_matrix_ru = corr_matrix.rename(
            index=column_names_ru,
            columns=column_names_ru
        )

        # Создаем тепловую карту
        plt.figure(figsize=(12, 10))  # Увеличиваем размер для лучшей читаемости русского текста
        sns.heatmap(corr_matrix_ru, annot=True, cmap="YlOrBr", linewidths=0.6, fmt='.2f')
        configure_plot_for_cyrillic(
            title="Корреляционная матрица факторов полета и задержек"
        )
        plt.tight_layout(pad=2.0)
        plt.savefig('./img/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Анализ взаимосвязи между временем руления и задержками
        if all(col in df.columns for col in ['TaxiOut', 'DepDelay']):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='TaxiOut', y='DepDelay', data=df.sample(min(5000, len(df))), alpha=0.5)
            configure_plot_for_cyrillic(
                title="Взаимосвязь между временем руления на вылет и задержкой вылета",
                xlabel="Время руления на вылет (минуты)",
                ylabel="Задержка вылета (минуты)"
            )
            plt.tight_layout(pad=2.0)
            plt.savefig('./img/taxiout_depdelay_scatter.png')
            plt.close()

        if all(col in df.columns for col in ['TaxiIn', 'ArrDelay']):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='TaxiIn', y='ArrDelay', data=df.sample(min(5000, len(df))), alpha=0.5)
            configure_plot_for_cyrillic(
                title="Взаимосвязь между временем руления на прилет и задержкой прибытия",
                xlabel="Время руления на прилет (минуты)",
                ylabel="Задержка прибытия (минуты)"
            )
            plt.tight_layout(pad=2.0)
            plt.savefig('./img/taxiin_arrdelay_scatter.png')
            plt.close()

    print("Анализ корреляций завершен.")


# Функция для построения моделей предсказания задержек
def build_delay_prediction_models(df, target='ArrDelay'):
    """
    Создает модели предсказания задержек и визуализирует их результаты

    Args:
        df (DataFrame): DataFrame с данными о рейсах
        target (str): Целевая переменная ('ArrDelay' или 'DepDelay')
    """
    print(f"Построение моделей предсказания для {target}...")

    # Проверяем наличие целевой переменной
    if target not in df.columns:
        print(f"Колонка {target} не найдена в данных. Пропускаем построение моделей.")
        return

    # Список возможных признаков для модели
    possible_features = ['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime',
                         'Distance', 'CRSElapsedTime']

    # Добавляем DepDelay в признаки, если целевая - ArrDelay
    if target == 'ArrDelay' and 'DepDelay' in df.columns:
        possible_features.append('DepDelay')

    # Выбираем только те признаки, которые есть в данных
    features = [col for col in possible_features if col in df.columns]

    if not features:
        print("Недостаточно признаков для построения модели. Пропускаем.")
        return

    # Подготовка данных для моделирования
    data_subset = df[features + [target]].dropna()

    # Проверяем и преобразуем все нечисловые столбцы в числовые
    for column in data_subset.columns:
        # Если столбец нечисловой
        if data_subset[column].dtype == object:
            print(f"Преобразование нечислового столбца: {column}")

            # Особая обработка для временных столбцов
            if column in ['CRSDepTime', 'CRSArrTime'] and data_subset[column].astype(str).str.contains(':').any():
                # Конвертация времени в числовой формат (часы + минуты/60)
                data_subset[column] = data_subset[column].apply(
                    lambda x: float(str(x).split(':')[0]) + float(str(x).split(':')[1]) / 60
                    if isinstance(x, str) and ':' in x else x)
            else:
                # Обычное преобразование в числовой формат
                data_subset[column] = pd.to_numeric(data_subset[column], errors='coerce')

            # Заполнение NaN значений медианой
            if data_subset[column].isna().any():
                data_subset[column] = data_subset[column].fillna(data_subset[column].median())

    # Преобразование категориальных переменных
    data_model = pd.get_dummies(data_subset, columns=['Month', 'DayOfWeek'], drop_first=True)

    # Разделение на признаки и целевую переменную
    X = data_model.drop(target, axis=1)
    y = data_model[target]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Список моделей
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }

    # Обучение моделей и визуализация результатов
    for name, model in models.items():
        print(f"Обучение модели: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Оценка модели
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.2f}")
        print(f"MAE: {mae:.2f}")

        # Визуализация распределения фактических и предсказанных значений
        plt.figure(figsize=(15, 7))
        ax1 = sns.histplot(y_test, kde=True, stat="density", color='r', label='Фактические значения')
        sns.histplot(y_pred, kde=True, stat="density", color='b', label='Предсказанные значения', ax=ax1)
        plt.legend()
        configure_plot_for_cyrillic(
            title=f"Распределение фактических и предсказанных значений {target} - {name}",
            xlabel=f"Фактические и предсказанные значения {target} (минуты)",
            ylabel="Плотность"
        )
        plt.tight_layout()
        plt.savefig(f'./img/{target.lower()}_{name.lower().replace(" ", "_")}_distribution.png')
        plt.close()

        # Визуализация сравнения фактических и предсказанных значений
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred, alpha=0.3)

        # Добавление диагональной линии
        max_value = max(max(y_test), max(y_pred))
        min_value = min(min(y_test), min(y_pred))
        plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', lw=2)
        configure_plot_for_cyrillic(
            title=f"Сравнение фактических и предсказанных значений {target} - {name}",
            xlabel=f"Фактические значения {target} (минуты)",
            ylabel=f"Предсказанные значения {target} (минуты)"
        )
        plt.tight_layout(pad=2.0)
        plt.savefig(f'./img/{target.lower()}_{name.lower().replace(" ", "_")}_comparison.png')
        plt.close()

    print(f"Построение моделей для {target} завершено.")


def main():
    """
    Основная функция, которая запускает весь процесс анализа
    """
    # Проверяем наличие таблицы в базе данных
    session = Session()
    try:
        check_tables_query = """
                             SELECT EXISTS (SELECT \
                                            FROM information_schema.tables \
                                            WHERE table_schema = 'public' \
                                              AND table_name = 'flight_data_for_visualization'); \
                             """
        table_exists = session.execute(text(check_tables_query)).scalar()

        if not table_exists:
            print("Таблица flight_data_for_visualization не найдена в базе данных.")
            return
    finally:
        session.close()

    # Загрузка данных из базы данных
    df = load_data()

    if df is None:
        print("Не удалось загрузить данные. Выход из программы.")
        return

    # Предобработка данных
    df = pd.DataFrame(df)
    df = preprocess_data(df)

    # Вывод основной информации о данных
    print("\nОсновная информация о данных:")
    print(f"Количество строк: {df.shape[0]}")
    print(f"Количество столбцов: {df.shape[1]}")
    print(f"Временной период: с {df['FlightDate'].min()} по {df['FlightDate'].max()}")

    # Анализ задержек по авиакомпаниям
    analyze_airline_delays(df)

    # Анализ временных трендов задержек
    analyze_delay_trends(df)

    # Анализ задержек по аэропортам
    analyze_airport_delays(df)

    # Анализ причин задержек
    analyze_delay_causes(df)

    # Анализ корреляций
    analyze_correlations(df)

    # Построение моделей предсказания задержек
    # Модели для задержек прибытия
    build_delay_prediction_models(df, target='ArrDelay')

    # Модели для задержек вылета
    build_delay_prediction_models(df, target='DepDelay')

    print("\nАнализ завершен. Все графики сохранены в текущем каталоге.")


# Запуск скрипта
if __name__ == "__main__":
    main()