import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import geopandas as gpd
from matplotlib.colors import ListedColormap
from risk import calculate_risk, calculate_risk_vectorized
import earthaccess
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import griddata
import os
from glob import glob

# Определяем даты конца каждого месяца
END_DATES = [
    "20240131", "20240229", "20240331", "20240430",
    "20240515", "20240930", "20241031", "20241130", "20241231"
]

# Словарь соответствия дат и номеров месяцев
MONTH_MAPPING = {
    "20240131": 1,  # Январь
    "20240229": 2,  # Февраль
    "20240331": 3,  # Март
    "20240430": 4,  # Апрель
    "20240515": 5,  # Май
    "20240930": 9,  # Сентябрь
    "20241031": 10, # Октябрь
    "20241130": 11, # Ноябрь
    "20241231": 12  # Декабрь
}

# Загрузка береговой линии
land = gpd.read_file('ne_10m_land.shp')

# Функция для выбора месяца
def select_month():
    print("\nДоступные месяцы:")
    for i, date in enumerate(END_DATES):
        month_num = MONTH_MAPPING[date]
        month_name = datetime(2024, month_num, 1).strftime('%B')
        print(f"{i+1}. {month_name} 2024 ({date})")
    
    while True:
        try:
            choice = int(input("\nВыберите номер месяца (1-9): ")) - 1
            if 0 <= choice < len(END_DATES):
                return choice
            else:
                print("Неверный номер. Попробуйте снова.")
        except ValueError:
            print("Введите число от 1 до 9.")

# Функция для поиска или загрузки данных
def get_dataset(end_date_str):
    # Сначала ищем локальный файл
    local_pattern = f"data/*/RDEFT4_{end_date_str}.nc"
    existing_files = glob(local_pattern)

    if existing_files:
        print(f"Найден локальный файл для {end_date_str}")
        return xr.open_dataset(existing_files[0])
    
    # Если локальный файл не найден, пытаемся загрузить
    print(f"Локальный файл не найден для {end_date_str}. Попытка загрузки...")
    try:
        # Аутентификация только при необходимости загрузки
        auth = earthaccess.login()
        end_date = datetime.strptime(end_date_str, "%Y%m%d")
        
        results = earthaccess.search_data(
            short_name="RDEFT4",
            bounding_box=(30, 66, 180, 82),  # Расширенная область для всего СМП
            temporal=(end_date, end_date + timedelta(days=1))
        )

        if not results:
            print(f"Файл RDEFT4_{end_date_str}.nc не найден на сервере")
            return None

        local_files = earthaccess.download(results)
        return xr.open_dataset(local_files[0])
    
    except Exception as e:
        print(f"Ошибка при попытке загрузки данных: {e}")
        return None

# Выбор месяца пользователем
selected_month_index = select_month()

# Перебираем все доступные даты
for end_date_str in END_DATES[selected_month_index:selected_month_index+1]:
    # Создаем новое окно для каждого месяца с увеличенным размером
    fig = plt.figure(figsize=(15, 10))
    
    # Получаем dataset
    ds = get_dataset(end_date_str)
    if ds is None:
        print(f"Пропуск {end_date_str} из-за отсутствия данных")
        plt.close(fig)
        continue

    # Получение и обработка данных
    ice_thickness_data = ds['sea_ice_thickness'].values.copy()
    ice_thickness_data[ice_thickness_data == -9999.0] = 0
    ice_thickness_data = np.nan_to_num(ice_thickness_data, nan=0.0)
    
    ice_concentration_data = ds['ice_con'].values.copy()
    ice_concentration_data[ice_concentration_data == -9999.0] = 0
    ice_concentration_data = np.nan_to_num(ice_concentration_data, nan=0.0)

    # Вычисление рисков
    # Старый код с двойным циклом заменён на векторное вычисление
    risk_values = calculate_risk_vectorized(ice_concentration_data, ice_thickness_data)

    # Создание графика для текущего месяца с полярной проекцией
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=90))
    ax.set_extent([20, 190, 66, 82], crs=ccrs.PlateCarree())  # Весь СМП

    # Добавление береговой линии
    ax.coastlines(resolution='50m')
    ax.add_geometries(land.geometry, crs=ccrs.PlateCarree(),
                     facecolor='whitesmoke', edgecolor='black', linewidth=0.5)

    # Отрисовка данных с улучшенными настройками
    img = ax.pcolormesh(ds.lon.values, ds.lat.values, risk_values,
                        transform=ccrs.PlateCarree(),
                        cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1, alpha=0.7)

    # Добавление сетки с улучшенными параметрами для большей области
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                      xlocs=np.arange(20, 200, 20), ylocs=np.arange(66, 84, 3))
    gl.top_labels = False
    gl.right_labels = False

    # Добавление названия месяца
    month_num = MONTH_MAPPING[end_date_str]
    month_name = datetime(2024, month_num, 1).strftime('%B')
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    
    plt.title(f"Карта рисков - {month_name} 2024\n(30-day average ending {end_date.strftime('%d.%m')})",
              fontsize=12, pad=20)

    # Добавление цветовой шкалы с увеличенным отступом
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.08)
    cbar.set_label('Риск (0 - низкий, 1 - высокий)')

    plt.tight_layout()

plt.show()
