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

# Стандартные размеры графиков
FIGURE_SIZE_STANDARD = (12, 8)  # Стандартный размер для большинства графиков
FIGURE_SIZE_WIDE = (15, 8)  # Широкий формат для временных рядов
FIGURE_SIZE_SQUARE = (10, 10)  # Квадратный формат для корреляций и scatter-plots

# Черно-белая палитра и стили для маркировки
GRAYSCALE_PALETTE = ["#000000", "#333333", "#666666", "#999999", "#CCCCCC"]
HATCH_PATTERNS = ['', '////', '\\\\\\\\', 'xxxx', '....', 'oooo', '****']
LINE_STYLES = ['-', '--', ':', '-.', '-', '--', ':']
MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>']


# Настройка общего стиля для графиков, оптимизированных для черно-белой печати
def set_plotting_style():
    """Устанавливает единый стиль для всех графиков, оптимизированный для черно-белой печати"""
    plt.style.use('grayscale')  # Использование встроенной черно-белой темы matplotlib

    # Увеличиваем толщину линий для лучшей видимости при печати
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['patch.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1.0

    # Настройка фона графиков
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

    # Настройка сетки
    plt.rcParams['grid.color'] = '#aaaaaa'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.7

    # Настройка подписей
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    # Черно-белые маркеры
    plt.rcParams['scatter.marker'] = 'o'
    plt.rcParams['lines.markeredgewidth'] = 1.5


def configure_plot_for_cyrillic(title=None, xlabel=None, ylabel=None, legend_loc='best'):
    """
    Настраивает график для корректного отображения кириллицы и оптимального размещения элементов

    Args:
        title (str): Заголовок графика
        xlabel (str): Подпись оси X
        ylabel (str): Подпись оси Y
        legend_loc (str): Расположение легенды
    """
    if title: plt.title(title, fontsize=16, pad=15, fontweight='bold')
    if xlabel: plt.xlabel(xlabel, fontsize=14, labelpad=10, fontweight='bold')
    if ylabel: plt.ylabel(ylabel, fontsize=14, labelpad=10, fontweight='bold')
    if plt.gca().get_legend() is not None:
        plt.legend(loc=legend_loc, fontsize=12, frameon=True, framealpha=0.9,
                   edgecolor='black', facecolor='white')

    # Добавляем сетку, которая хорошо видна на черно-белой печати
    plt.grid(True, linestyle=':', alpha=0.7, linewidth=1.0)

    # Делаем ободок графика более заметным
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)


# Установка стиля при импорте модуля
set_plotting_style()

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


def load_data(table_name='flight_data_for_visualization', sample_size=100):
    """
    Загружает данные о рейсах из базы данных PostgreSQL

    Args:
        table_name (str): Имя таблицы в базе данных (по умолчанию 'flight_data_for_visualization')

    Returns:
        list: Список словарей, где каждый словарь представляет строку данных
    """
    print(f"Загрузка данных из таблицы {table_name}...")

    session = Session()
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


# Функция для анализа задержек по авиакомпаниям
def analyze_airline_delays(df):
    """
    Создает графики для анализа задержек по авиакомпаниям

    Args:
        df (DataFrame): DataFrame с данными о рейсах
    """
    print("Анализ задержек по авиакомпаниям...")

    # 1. Количество рейсов по авиакомпаниям
    plt.figure(figsize=FIGURE_SIZE_STANDARD)
    airline_counts = df["Reporting_Airline"].value_counts()

    # Создаем черно-белый барплот с шаблонной заливкой для лучшего различения
    ax = plt.gca()
    bars = ax.bar(range(len(airline_counts)), airline_counts, color='white',
                  edgecolor='black', linewidth=1.5)

    # Добавляем различные штриховки для каждого столбца
    for i, bar in enumerate(bars):
        bar.set_hatch(HATCH_PATTERNS[i % len(HATCH_PATTERNS)])

    # Добавляем числовые значения над столбцами
    for i, v in enumerate(airline_counts):
        ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

    # Настраиваем оси
    plt.xticks(range(len(airline_counts)), airline_counts.index, rotation=45)

    configure_plot_for_cyrillic(
        title="Количество рейсов по авиакомпаниям",
        xlabel="Авиакомпания",
        ylabel="Количество рейсов"
    )
    plt.tight_layout(pad=2.0)
    plt.savefig('./img_black/airline_flight_counts.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Средние задержки по авиакомпаниям
    plt.figure(figsize=FIGURE_SIZE_STANDARD)

    # Создаем DataFrame с задержками по авиакомпаниям
    airline_delays = df.groupby('Reporting_Airline')[['DepDelay', 'ArrDelay']].mean()

    # Переименовываем колонки на русский язык для легенды
    airline_delays = airline_delays.rename(columns={
        'DepDelay': 'Задержка вылета',
        'ArrDelay': 'Задержка прибытия'
    })

    # Сортируем по задержке вылета
    airline_delays_sorted = airline_delays.sort_values('Задержка вылета', ascending=False)

    # Создаем индексы для позиций столбцов
    x = np.arange(len(airline_delays_sorted.index))
    width = 0.35  # ширина столбцов

    # Создаем бар-график с разными штриховками для разных типов задержек
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

    # Столбцы для задержки вылета с одним типом штриховки
    bars1 = ax.bar(x - width / 2, airline_delays_sorted['Задержка вылета'], width,
                   color='white', edgecolor='black', linewidth=1.5, hatch='////')

    # Столбцы для задержки прибытия с другим типом штриховки
    bars2 = ax.bar(x + width / 2, airline_delays_sorted['Задержка прибытия'], width,
                   color='white', edgecolor='black', linewidth=1.5, hatch='\\\\\\\\')

    configure_plot_for_cyrillic(
        title="Распределение задержек по разным авиакомпаниям",
        xlabel="Авиакомпания",
        ylabel="Задержка (мин)",
        legend_loc='upper right'
    )

    # Устанавливаем метки на оси X
    ax.set_xticks(x)
    ax.set_xticklabels(airline_delays_sorted.index, rotation=45, ha='right')

    # Добавляем легенду с пояснениями штриховок
    ax.legend([bars1[0], bars2[0]], ['Задержка вылета', 'Задержка прибытия'])

    plt.tight_layout(pad=2.0)
    plt.savefig('./img_black/airline_delay_distribution.png', dpi=300, bbox_inches='tight')
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

    # Переименовываем колонки для русских подписей в легенде
    monthly_delays = monthly_delays.rename(columns={
        'DepDelay': 'Задержка вылета',
        'ArrDelay': 'Задержка прибытия'
    })

    plt.figure(figsize=FIGURE_SIZE_WIDE)

    # Используем разные стили линий и маркеры для черно-белой печати
    plt.plot(monthly_delays['YearMonth'], monthly_delays['Задержка вылета'],
             linestyle='-', linewidth=2, marker='o', markersize=8,
             color='black', label='Задержка вылета')

    plt.plot(monthly_delays['YearMonth'], monthly_delays['Задержка прибытия'],
             linestyle='--', linewidth=2, marker='s', markersize=8,
             color='black', label='Задержка прибытия')

    configure_plot_for_cyrillic(
        title="Средние задержки вылета и прибытия по времени",
        xlabel="Год-Месяц",
        ylabel="Средняя задержка (минуты)",
        legend_loc='upper right'
    )
    plt.xticks(rotation=45)
    plt.tight_layout(pad=2.0)
    plt.savefig('./img_black/monthly_delays.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Доля отмененных рейсов по месяцам
    if 'Cancelled' in df.columns:
        monthly_cancellations = df.groupby('YearMonth')['Cancelled'].mean().reset_index()

        plt.figure(figsize=FIGURE_SIZE_WIDE)
        plt.plot(monthly_cancellations['YearMonth'], monthly_cancellations['Cancelled'],
                 linestyle='-', linewidth=2, marker='D', markersize=8,
                 color='black')

        configure_plot_for_cyrillic(
            title="Доля отмененных рейсов по времени",
            xlabel="Год-Месяц",
            ylabel="Доля отмененных рейсов"
        )
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.savefig('./img_black/monthly_cancellations.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Распределение задержек по дням недели
    if 'DayOfWeek' in df.columns:
        day_mapping = {1: 'Понедельник', 2: 'Вторник', 3: 'Среда',
                       4: 'Четверг', 5: 'Пятница', 6: 'Суббота', 7: 'Воскресенье'}

        df['DayName'] = df['DayOfWeek'].map(day_mapping)
        day_delays = df.groupby('DayName')[['DepDelay', 'ArrDelay']].mean().reindex(
            ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
        )

        # Переименовываем колонки для русской легенды
        day_delays = day_delays.rename(columns={'DepDelay': 'Задержка вылета', 'ArrDelay': 'Задержка прибытия'})

        # Создаем график с группированными столбцами и разными штриховками
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

        # Создаем индексы для позиций столбцов
        x = np.arange(len(day_delays.index))
        width = 0.35  # ширина столбцов

        # Столбцы для задержки вылета с одним типом штриховки
        bars1 = ax.bar(x - width / 2, day_delays['Задержка вылета'], width,
                       color='white', edgecolor='black', linewidth=1.5, hatch='////')

        # Столбцы для задержки прибытия с другим типом штриховки
        bars2 = ax.bar(x + width / 2, day_delays['Задержка прибытия'], width,
                       color='white', edgecolor='black', linewidth=1.5, hatch='\\\\\\\\')

        configure_plot_for_cyrillic(
            title="Средние задержки по дням недели",
            xlabel="День недели",
            ylabel="Средняя задержка (минуты)"
        )

        # Устанавливаем метки на оси X
        ax.set_xticks(x)
        ax.set_xticklabels(day_delays.index, rotation=45, ha='right')

        # Добавляем легенду с пояснениями штриховок
        ax.legend([bars1[0], bars2[0]], ['Задержка вылета', 'Задержка прибытия'])

        plt.tight_layout(pad=2.0)
        plt.savefig('./img_black/delays_by_day_of_week.png', dpi=300, bbox_inches='tight')
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

        # Переименовываем колонки для русской легенды
        airport_delays = airport_delays.rename(columns={
            'DepDelay': 'Задержка вылета',
            'ArrDelay': 'Задержка прибытия'
        })

        # Топ по задержкам вылета
        dep_delays = airport_delays.sort_values('Задержка вылета', ascending=False).head(
            10)  # Ограничим до 10 для читаемости

        # Создаем график с разными штриховками для разных аэропортов назначения
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

        # Создаем индексы для позиций столбцов
        x = np.arange(len(dep_delays))

        # Создаем столбчатый график с черно-белыми столбцами и разной штриховкой
        bars = ax.bar(x, dep_delays['Задержка вылета'], color='white',
                      edgecolor='black', linewidth=1.5)

        # Добавляем разные штриховки для различения аэропортов
        for i, bar in enumerate(bars):
            bar.set_hatch(HATCH_PATTERNS[i % len(HATCH_PATTERNS)])

        configure_plot_for_cyrillic(
            title='Наибольшие средние задержки вылета (топ 10 аэропортов)',
            xlabel='Аэропорт вылета',
            ylabel='Средняя задержка вылета (минуты)'
        )

        # Устанавливаем метки на оси X
        ax.set_xticks(x)
        ax.set_xticklabels(dep_delays['Origin'], rotation=45, ha='right')

        # Добавляем информацию об аэропортах назначения в виде аннотаций
        for i, (_, row) in enumerate(dep_delays.iterrows()):
            ax.annotate(f"→ {row['Dest']}",
                        (i, row['Задержка вылета']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontweight='bold')

        plt.tight_layout(pad=2.0)
        plt.savefig('./img_black/top_departure_delays_by_airport.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Топ по задержкам прибытия
        arr_delays = airport_delays.sort_values('Задержка прибытия', ascending=False).head(
            10)  # Ограничим до 10 для читаемости

        # Создаем график с разными штриховками для разных аэропортов отправления
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

        # Создаем индексы для позиций столбцов
        x = np.arange(len(arr_delays))

        # Создаем столбчатый график с черно-белыми столбцами и разной штриховкой
        bars = ax.bar(x, arr_delays['Задержка прибытия'], color='white',
                      edgecolor='black', linewidth=1.5)

        # Добавляем разные штриховки для различения аэропортов
        for i, bar in enumerate(bars):
            bar.set_hatch(HATCH_PATTERNS[i % len(HATCH_PATTERNS)])

        configure_plot_for_cyrillic(
            title='Наибольшие средние задержки прибытия (топ 10 аэропортов)',
            xlabel='Аэропорт прибытия',
            ylabel='Средняя задержка прибытия (минуты)'
        )

        # Устанавливаем метки на оси X
        ax.set_xticks(x)
        ax.set_xticklabels(arr_delays['Dest'], rotation=45, ha='right')

        # Добавляем информацию об аэропортах отправления в виде аннотаций
        for i, (_, row) in enumerate(arr_delays.iterrows()):
            ax.annotate(f"← {row['Origin']}",
                        (i, row['Задержка прибытия']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontweight='bold')

        plt.tight_layout(pad=2.0)
        plt.savefig('./img_black/top_arrival_delays_by_airport.png', dpi=300, bbox_inches='tight')
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

    # Проверка наличия колонок с причинами задержек
    delay_causes = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
    if not all(cause in df.columns for cause in delay_causes):
        print("Данные о причинах задержек отсутствуют. Пропускаем анализ.")
        return

    # 1. Средние значения разных типов задержек
    avg_delays = pd.DataFrame({
        'Причина': ['Авиакомпания', 'Погода', 'Авиадиспетчеры', 'Безопасность', 'Позднее прибытие самолета'],
        'Средняя задержка': [
            df['CarrierDelay'].mean(),
            df['WeatherDelay'].mean(),
            df['NASDelay'].mean(),
            df['SecurityDelay'].mean(),
            df['LateAircraftDelay'].mean()
        ]
    })

    # Создаем график со штриховками
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

    # Создаем столбчатый график с черно-белыми столбцами и разной штриховкой
    bars = ax.bar(avg_delays['Причина'], avg_delays['Средняя задержка'],
                  color='white', edgecolor='black', linewidth=1.5)

    # Добавляем разные штриховки для различения причин
    for i, bar in enumerate(bars):
        bar.set_hatch(HATCH_PATTERNS[i % len(HATCH_PATTERNS)])

    # Добавляем значения над столбцами
    for i, v in enumerate(avg_delays['Средняя задержка']):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')

    configure_plot_for_cyrillic(
        title='Средние задержки по разным причинам',
        xlabel='Причина задержки',
        ylabel='Средняя задержка (минуты)'
    )

    plt.tight_layout(pad=2.0)
    plt.savefig('./img_black/avg_delay_by_cause.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Доля каждой причины в общей задержке
    # Создаем новый DataFrame с процентным распределением причин
    total_delay = df[delay_causes].sum().sum()
    delay_pct = pd.DataFrame({
        'Причина': ['Авиакомпания', 'Погода', 'Авиадиспетчеры', 'Безопасность', 'Позднее прибытие самолета'],
        'Процент': [
            df['CarrierDelay'].sum() / total_delay * 100,
            df['WeatherDelay'].sum() / total_delay * 100,
            df['NASDelay'].sum() / total_delay * 100,
            df['SecurityDelay'].sum() / total_delay * 100,
            df['LateAircraftDelay'].sum() / total_delay * 100
        ]
    })

    # Создаем черно-белую круговую диаграмму с разными штриховками
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)

    # Создаем круговую диаграмму
    wedges, texts, autotexts = ax.pie(
        delay_pct['Процент'],
        labels=delay_pct['Причина'],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
        textprops={'fontweight': 'bold'}
    )

    # Добавляем разные штриховки для сегментов
    for i, wedge in enumerate(wedges):
        wedge.set_facecolor('white')
        wedge.set_hatch(HATCH_PATTERNS[i % len(HATCH_PATTERNS)])

    # Настраиваем автотексты на контрастный цвет
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    configure_plot_for_cyrillic(
        title='Процентное распределение причин задержек'
    )

    ax.axis('equal')  # Обеспечивает круглую форму пирога
    plt.tight_layout(pad=2.0)
    plt.savefig('./img_black/delay_causes_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Соотношение причин задержек по месяцам (если доступны)
    if 'YearMonth' in df.columns:
        # Создаем сводку причин задержек по месяцам
        monthly_causes = df.groupby('YearMonth')[delay_causes].mean().reset_index()

        # Переименовываем колонки
        monthly_causes = monthly_causes.rename(columns={
            'CarrierDelay': 'Авиакомпания',
            'WeatherDelay': 'Погода',
            'NASDelay': 'Авиадиспетчеры',
            'SecurityDelay': 'Безопасность',
            'LateAircraftDelay': 'Позднее прибытие самолета'
        })

        # Переводим данные в формат long для построения графика
        monthly_causes_long = pd.melt(
            monthly_causes,
            id_vars=['YearMonth'],
            value_vars=['Авиакомпания', 'Погода', 'Авиадиспетчеры', 'Безопасность', 'Позднее прибытие самолета'],
            var_name='Причина',
            value_name='Средняя задержка'
        )

        # Создаем график для черно-белой печати
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

        # Получаем уникальные причины
        causes = monthly_causes_long['Причина'].unique()

        # Для каждой причины рисуем линию с уникальным стилем
        for i, cause in enumerate(causes):
            cause_data = monthly_causes_long[monthly_causes_long['Причина'] == cause]
            ax.plot(cause_data['YearMonth'], cause_data['Средняя задержка'],
                    label=cause,
                    color='black',
                    linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                    marker=MARKER_STYLES[i % len(MARKER_STYLES)],
                    linewidth=2,
                    markersize=8)

        configure_plot_for_cyrillic(
            title='Динамика причин задержек по месяцам',
            xlabel='Год-Месяц',
            ylabel='Средняя задержка (минуты)',
            legend_loc='upper right'
        )

        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.savefig('./img_black/monthly_delay_causes.png', dpi=300, bbox_inches='tight')
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

        # Создаем черно-белую тепловую карту с улучшенной читаемостью
        plt.figure(figsize=FIGURE_SIZE_SQUARE)
        cmap = plt.cm.gray  # Используем черно-белую карту

        # Настройка для лучшей читаемости в черно-белом формате
        # Сильные корреляции будут темными, слабые - светлыми
        mask = np.zeros_like(corr_matrix_ru)
        mask[np.triu_indices_from(mask, 1)] = True  # Маскируем верхний треугольник (дублирующие значения)

        # Добавляем больше контраста
        sns.heatmap(corr_matrix_ru,
                    annot=True,
                    cmap=cmap,
                    vmin=-1,
                    vmax=1,
                    center=0,
                    linewidths=1.0,
                    linecolor='black',
                    fmt='.2f',
                    annot_kws={"size": 10, "weight": "bold"},
                    mask=mask,
                    cbar_kws={"shrink": 0.8})

        # Добавляем сетку для улучшения читаемости
        plt.grid(False)  # Отключаем обычную сетку, т.к. heatmap уже имеет разделители

        configure_plot_for_cyrillic(
            title="Корреляционная матрица факторов полета и задержек"
        )

        plt.tight_layout(pad=2.0)
        plt.savefig('./img_black/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Анализ корреляций завершен.")
    else:
        print("Недостаточно числовых колонок для анализа корреляций.")


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

    # Словарь с русскими названиями моделей
    model_names_ru = {
        'Linear Regression': 'Линейная регрессия',
        'Decision Tree': 'Дерево решений',
        'Random Forest': 'Случайный лес',
        'XGBoost': 'XGBoost'
    }

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

        # Русское название целевой переменной
        target_ru = "задержки прибытия" if target == "ArrDelay" else "задержки вылета"

        # Визуализация распределения фактических и предсказанных значений (оптимизировано для черно-белой печати)
        plt.figure(figsize=FIGURE_SIZE_STANDARD)

        # Рисуем гистограммы с разной штриховкой
        # Фактические значения - сплошной черный
        ax = plt.gca()
        _, bins, _ = ax.hist(y_test, bins=15, alpha=0.6, color='white', edgecolor='black',
                             linewidth=1.5, density=True, label='Фактические значения')

        # Предсказанные значения - с штриховкой
        ax.hist(y_pred, bins=bins, alpha=0.6, color='white', edgecolor='black',
                linewidth=1.5, density=True, label='Предсказанные значения', hatch='////')

        # Добавляем кривые плотности вместо KDE
        from scipy.stats import gaussian_kde
        x_range = np.linspace(min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred)), 100)
        kde_test = gaussian_kde(y_test)
        kde_pred = gaussian_kde(y_pred)

        plt.plot(x_range, kde_test(x_range), 'k-', linewidth=2, label='_nolegend_')
        plt.plot(x_range, kde_pred(x_range), 'k--', linewidth=2, label='_nolegend_')

        configure_plot_for_cyrillic(
            title=f"Распределение фактических и предсказанных значений\nМодель: {model_names_ru[name]}",
            xlabel=f"Значение {target_ru} (минуты)",
            ylabel="Плотность",
            legend_loc='upper right'
        )
        plt.tight_layout(pad=2.0)
        plt.savefig(f'./img_black/{target.lower()}_{name.lower().replace(" ", "_")}_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Визуализация сравнения фактических и предсказанных значений
        plt.figure(figsize=FIGURE_SIZE_SQUARE)

        # Используем контрастные черно-белые маркеры с обводкой
        plt.scatter(y_test, y_pred, s=60, marker='o', facecolors='white',
                    edgecolors='black', linewidth=1.0, alpha=0.7)

        # Добавление диагональной линии идеального предсказания
        max_value = max(max(y_test), max(y_pred))
        min_value = min(min(y_test), min(y_pred))

        # Линия идеального предсказания - толстая черная пунктирная линия
        plt.plot([min_value, max_value], [min_value, max_value],
                 'k--', linewidth=2)

        # Добавляем границы графика рассеяния
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

        plt.tight_layout(pad=2.0)
        plt.savefig(f'./img_black/{target.lower()}_{name.lower().replace(" ", "_")}_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Построение моделей для {target} завершено.")


def main():
    """
    Основная функция, которая запускает весь процесс анализа
    """
    # Создаем директорию для изображений, если её нет
    if not os.path.exists('./img_black'):
        os.makedirs('./img_black')

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

    # Проверка данных перед анализом
    print("\nПроверка данных:")
    print(f"Типы данных в колонках:")
    print(df.dtypes)

    # Проверка на наличие пропущенных значений
    print("\nКоличество пропущенных значений по колонкам:")
    print(df.isnull().sum())

    # Базовая статистика по числовым колонкам
    print("\nСтатистика по числовым колонкам:")
    print(df.describe())

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

    print("\nАнализ завершен. Все графики сохранены в каталоге ./img_black")


if __name__ == "__main__":
    main()
