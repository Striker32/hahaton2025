import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import geopandas as gpd
from risk import calculate_risk_vectorized
import earthaccess
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from glob import glob
import plotly.graph_objects as go
import matplotlib.cm as cm

# Загрузка береговой линии
land = gpd.read_file('ne_10m_land.shp')

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

def select_month():
    """Функция для выбора месяца"""
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

def get_ice_dataset(end_date_str):
    """Получение данных о льде"""
    local_pattern = f"data/*/RDEFT4_{end_date_str}.nc"
    existing_files = glob(local_pattern)

    if existing_files:
        print(f"Найден локальный файл льда для {end_date_str}")
        return xr.open_dataset(existing_files[0])
    
    print(f"Локальный файл не найден для {end_date_str}. Попытка загрузки...")
    try:
        auth = earthaccess.login()
        end_date = datetime.strptime(end_date_str, "%Y%m%d")
        
        results = earthaccess.search_data(
            short_name="RDEFT4",
            bounding_box=(30, 66, 180, 82),
            temporal=(end_date, end_date + timedelta(days=1))
        )

        if not results:
            print(f"Файл RDEFT4_{end_date_str}.nc не найден на сервере")
            return None

        local_files = earthaccess.download(results)
        return xr.open_dataset(local_files[0])
    
    except Exception as e:
        print(f"Ошибка при попытке загрузки данных о льде: {e}")
        return None

def get_wind_dataset():
    """Получение данных о ветре с GFS"""
    now = datetime.now()
    dates_to_try = [
        (now - timedelta(days=1)).strftime("%Y%m%d"),  # Вчера
        now.strftime("%Y%m%d")  # Сегодня
    ]

    urls = []
    for date_str in dates_to_try:
        for h in [0, 6, 12, 18]:
            url = f"https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date_str}/gfs_0p25_{h:02d}z"
            urls.append((url, date_str, h))

    # Пробуем URL-ы по очереди
    for url, date_str, hour in reversed(urls):
        try:
            print(f"Попытка загрузить данные о ветре: {date_str} {hour:02d}:00 UTC")
            ds = xr.open_dataset(url, engine='pydap')
            print(f"✅ Успешно загружены данные о ветре за {date_str} {hour:02d}:00 UTC")
            return ds
        except Exception as e:
            continue
    
    print("❌ Не удалось загрузить данные о ветре")
    return None

def calculate_wind_risk(wind_speed):
    """
    Расчёт риска на основе скорости ветра (сбалансированная чувствительность)
    
    Для судоходства в Арктике критические диапазоны:
    - 0-5 м/с: Низкий риск (0.0-0.2)
    - 5-10 м/с: Умеренный риск (0.2-0.4)
    - 10-15 м/с: Повышенный риск (0.4-0.6)
    - 15-20 м/с: Высокий риск (0.6-0.8)
    - >20 м/с: Критический риск (0.8-1.0)
    """
    wind_risk = np.zeros_like(wind_speed, dtype=float)
    
    # Низкий риск: 0-5 м/с
    mask1 = (wind_speed >= 0) & (wind_speed < 5)
    wind_risk[mask1] = wind_speed[mask1] / 25.0  # 0.0-0.2
    
    # Умеренный риск: 5-10 м/с
    mask2 = (wind_speed >= 5) & (wind_speed < 10)
    wind_risk[mask2] = 0.2 + (wind_speed[mask2] - 5) / 25.0  # 0.2-0.4
    
    # Повышенный риск: 10-15 м/с
    mask3 = (wind_speed >= 10) & (wind_speed < 15)
    wind_risk[mask3] = 0.4 + (wind_speed[mask3] - 10) / 25.0  # 0.4-0.6
    
    # Высокий риск: 15-20 м/с
    mask4 = (wind_speed >= 15) & (wind_speed < 20)
    wind_risk[mask4] = 0.6 + (wind_speed[mask4] - 15) / 25.0  # 0.6-0.8
    
    # Критический риск: >20 м/с
    mask5 = wind_speed >= 20
    wind_risk[mask5] = 0.8 + np.minimum((wind_speed[mask5] - 20) / 50.0, 0.2)  # 0.8-1.0
    
    return np.clip(wind_risk, 0.0, 1.0)

def combine_risks(ice_risk, wind_risk, ice_weight=0.6, wind_weight=0.4):
    """
    Объединение рисков льда и ветра (улучшенная формула)
    
    Используем комбинацию:
    1. Берём максимум из двух рисков как базу
    2. Добавляем взвешенную сумму для учёта обоих факторов
    3. Усиливаем риск, когда оба фактора значимы
    
    - ice_weight: вес ледового риска (0.6 - лёд критичнее для навигации)
    - wind_weight: вес ветрового риска (0.4 - ветер усиливает опасность)
    """
    # Базовый риск - максимум из двух источников
    base_risk = np.maximum(ice_risk, wind_risk)
    
    # Взвешенная сумма для учёта обоих факторов
    weighted_sum = ice_risk * ice_weight + wind_risk * wind_weight
    
    # Комбинируем: берём среднее между максимумом и взвешенной суммой
    combined = (base_risk * 0.6 + weighted_sum * 0.4)
    
    # Синергетический эффект: если оба риска значимые (>0.3), усиливаем
    synergy_mask = (ice_risk > 0.3) & (wind_risk > 0.3)
    if np.any(synergy_mask):
        # Добавляем бонус, зависящий от произведения рисков
        synergy_boost = ice_risk[synergy_mask] * wind_risk[synergy_mask] * 0.3
        combined[synergy_mask] += synergy_boost
    
    return np.clip(combined, 0.0, 1.0)

def interpolate_to_grid(source_lon, source_lat, source_data, target_lon, target_lat):
    """Интерполяция данных на целевую сетку (оптимизированная версия)"""
    from scipy.interpolate import griddata
    
    # Создаём сетку источника
    if len(source_lon.shape) == 1:
        source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat)
    else:
        source_lon_grid = source_lon
        source_lat_grid = source_lat
    
    # Создаём сетку назначения  
    if len(target_lon.shape) == 1:
        target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)
    else:
        target_lon_grid = target_lon
        target_lat_grid = target_lat
    
    # Подготавливаем данные для интерполяции
    source_points = np.column_stack([
        source_lon_grid.ravel(),
        source_lat_grid.ravel()
    ])
    source_values = source_data.ravel()
    
    # Удаляем NaN и бесконечные значения
    valid_mask = np.isfinite(source_values)
    source_points = source_points[valid_mask]
    source_values = source_values[valid_mask]
    
    # Целевые точки
    target_points = np.column_stack([
        target_lon_grid.ravel(),
        target_lat_grid.ravel()
    ])
    
    # Интерполяция методом ближайшего соседа для граничных точек
    # и линейной для внутренних
    interpolated = griddata(
        source_points,
        source_values,
        target_points,
        method='linear',
        fill_value=0.0
    )
    
    # Заполняем NaN ближайшими значениями
    if np.any(np.isnan(interpolated)):
        interpolated_nearest = griddata(
            source_points,
            source_values,
            target_points,
            method='nearest'
        )
        nan_mask = np.isnan(interpolated)
        interpolated[nan_mask] = interpolated_nearest[nan_mask]
    
    return interpolated.reshape(target_lon_grid.shape)

# Основной код
print("=" * 60)
print("КОМПЛЕКСНАЯ КАРТА РИСКОВ")
print("Ледовая обстановка + Скорость ветра")
print("=" * 60)

# Выбор месяца для ледовых данных
selected_month_index = select_month()
end_date_str = END_DATES[selected_month_index]

# Загрузка данных о льде
print("\n📊 Загрузка данных о льде...")
ice_ds = get_ice_dataset(end_date_str)

if ice_ds is None:
    print("❌ Не удалось загрузить данные о льде. Выход.")
    exit(1)

# Загрузка данных о ветре
print("\n🌬️ Загрузка данных о ветре...")
wind_ds = get_wind_dataset()

if wind_ds is None:
    print("❌ Не удалось загрузить данные о ветре. Выход.")
    exit(1)

# Обработка данных о льде
print("\n🧊 Обработка данных о льде...")
ice_thickness_data = ice_ds['sea_ice_thickness'].values.copy()
ice_thickness_data[ice_thickness_data == -9999.0] = 0
ice_thickness_data = np.nan_to_num(ice_thickness_data, nan=0.0)

ice_concentration_data = ice_ds['ice_con'].values.copy()
ice_concentration_data[ice_concentration_data == -9999.0] = 0
ice_concentration_data = np.nan_to_num(ice_concentration_data, nan=0.0)

# Вычисление ледового риска
print("⚙️ Расчёт ледового риска...")
ice_risk = calculate_risk_vectorized(ice_concentration_data, ice_thickness_data)

# Обработка данных о ветре
print("\n🌬️ Обработка данных о ветре...")
time_idx = 0
lon_min, lon_max = 20, 190
lat_min, lat_max = 66, 82

# Получаем компоненты ветра
lon_indices = (wind_ds.lon >= lon_min) & (wind_ds.lon <= lon_max)
lat_indices = (wind_ds.lat >= lat_min) & (wind_ds.lat <= lat_max)

lon = wind_ds.lon[lon_indices].values
lat = wind_ds.lat[lat_indices].values

u_wind = wind_ds['ugrd10m'][time_idx, lat_indices, lon_indices].values
v_wind = wind_ds['vgrd10m'][time_idx, lat_indices, lon_indices].values

# Обработка маскированных массивов и NaN значений
if hasattr(u_wind, 'filled'):
    u_wind = u_wind.filled(0.0)
if hasattr(v_wind, 'filled'):
    v_wind = v_wind.filled(0.0)

u_wind = np.nan_to_num(u_wind, nan=0.0)
v_wind = np.nan_to_num(v_wind, nan=0.0)

# Вычисление скорости ветра
wind_speed = np.sqrt(u_wind**2 + v_wind**2)

print(f"📊 Статистика скорости ветра:")
print(f"   Мин: {np.min(wind_speed):.2f} м/с")
print(f"   Макс: {np.max(wind_speed):.2f} м/с")
print(f"   Средн: {np.mean(wind_speed):.2f} м/с")

print("⚙️ Расчёт ветрового риска...")
wind_risk_data = calculate_wind_risk(wind_speed)

print(f"📊 Статистика ветрового риска:")
print(f"   Мин: {np.min(wind_risk_data):.3f}")
print(f"   Макс: {np.max(wind_risk_data):.3f}")
print(f"   Средн: {np.mean(wind_risk_data):.3f}")

# Интерполяция ветрового риска на сетку ледовых данных
print("🔄 Согласование сеток данных...")
wind_risk_interpolated = interpolate_to_grid(
    lon, lat, wind_risk_data,
    ice_ds.lon.values, ice_ds.lat.values
)

print(f"📊 Статистика интерполированного ветрового риска:")
print(f"   Мин: {np.min(wind_risk_interpolated):.3f}")
print(f"   Макс: {np.max(wind_risk_interpolated):.3f}")
print(f"   Средн: {np.mean(wind_risk_interpolated):.3f}")

# Объединение рисков
print("🔗 Объединение рисков...")
combined_risk = combine_risks(ice_risk, wind_risk_interpolated)

# Получаем название месяца для веб-карты
month_num = MONTH_MAPPING[end_date_str]
month_name = datetime(2024, month_num, 1).strftime('%B')

# ========================================
# СОЗДАНИЕ ИНТЕРАКТИВНОЙ ВЕБ-КАРТЫ
# ========================================

def create_interactive_map(ice_risk, wind_risk, combined_risk, 
                          ice_ds, ice_date, wind_date):
    """
    Создаёт интерактивную веб-карту с комплексным риском (сплошная заливка)
    """
    print("   Конвертация данных для веб-отображения...")
    print(f"   Исходные данные: форма {combined_risk.shape}")
    
    # Получаем координаты из датасета
    lon_2d = ice_ds.lon.values
    lat_2d = ice_ds.lat.values
    
    # Получаем маску льда для более точной фильтрации суши
    ice_con = ice_ds['ice_con'].values
    ice_thickness = ice_ds['sea_ice_thickness'].values
    
    # Создаём маску океана (где есть данные о льде)
    ocean_mask = ((ice_con != -9999.0) | (ice_thickness != -9999.0)) & (lat_2d >= 60)
    
    # Применяем маску к данным риска (заменяем сушу на NaN для прозрачности)
    combined_masked = combined_risk.copy()
    combined_masked[~ocean_mask] = np.nan
    
    print(f"   Океанских ячеек: {np.sum(ocean_mask)} из {ocean_mask.size}")
    
    # Создаём фигуру
    fig = go.Figure()
    
    # Создаём сплошную заливку используя многоугольники (более плотное покрытие)
    print("   Создание сплошной заливки с адаптивным размером...")
    
    # Используем Scattergeo с sizemode='area' для адаптивного размера при зуме
    # Создаём очень плотную сетку точек
    step = 1
    lon_plot = lon_2d[::step, ::step]
    lat_plot = lat_2d[::step, ::step]
    risk_plot = combined_masked[::step, ::step]
    
    # Преобразуем в 1D массивы, убирая NaN
    lon_flat = lon_plot.ravel()
    lat_flat = lat_plot.ravel()
    risk_flat = risk_plot.ravel()
    
    # Фильтруем NaN значения (суша)
    valid = ~np.isnan(risk_flat)
    lon_valid = lon_flat[valid]
    lat_valid = lat_flat[valid]
    risk_valid = risk_flat[valid]
    
    print(f"   Точек для отрисовки: {len(risk_valid)}")
    
    # Используем Scattergeo с большим размером маркеров для сплошного покрытия
    # Размер подобран так, чтобы маркеры перекрывались даже при максимальном зуме
    marker_size = 8  # Увеличенный размер для покрытия при любом масштабе
    
    print(f"   Размер маркера: {marker_size} (оптимизирован для сплошного покрытия)")
    
    fig.add_trace(
        go.Scattergeo(
            lon=lon_valid,
            lat=lat_valid,
            mode='markers',
            marker=dict(
                size=marker_size,
                color=risk_valid,
                colorscale=[
                    [0.0, '#2166ac'],   # Синий - низкий риск
                    [0.3, '#4393c3'],   # Голубой
                    [0.5, '#fee090'],   # Жёлтый - средний
                    [0.7, '#f4a582'],   # Оранжевый
                    [0.9, '#d6604d'],   # Красный
                    [1.0, '#b2182b']    # Тёмно-красный - критический
                ],
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(
                        text="Уровень<br>риска",
                        side="right"
                    ),
                    tickmode="array",
                    tickvals=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                    ticktext=['0.0<br>Низкий', '0.3<br>Умеренный', '0.5<br>Повышенный', 
                             '0.7<br>Высокий', '0.9<br>Критич.', '1.0'],
                    len=0.7,
                    thickness=20
                ),
                opacity=0.9,
                line=dict(width=0),
                symbol='square'
            ),
            hovertemplate='<b>Комплексный риск навигации</b><br>' +
                         'Риск: %{marker.color:.3f}<br>' +
                         'Широта: %{lat:.2f}°<br>' +
                         'Долгота: %{lon:.2f}°' +
                         '<extra></extra>',
            showlegend=False
        )
    )
    
    # Настройки географии
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=105, lat=75, roll=0),
        showcountries=True,
        countrycolor='darkgray',
        showcoastlines=True,
        coastlinecolor='black',
        coastlinewidth=1.5,
        showland=True,
        landcolor='#f5f5f5',
        showocean=True,
        oceancolor='#e8f4f8',
        showlakes=True,
        lakecolor='#e8f4f8',
        bgcolor='white',
        lataxis=dict(range=[66, 82], showgrid=True, gridwidth=0.5, gridcolor='lightgray'),
        lonaxis=dict(range=[20, 190], showgrid=True, gridwidth=0.5, gridcolor='lightgray')
    )
    
    # Общие настройки layout
    fig.update_layout(
        title=dict(
            text=f'⚓ КОМПЛЕКСНАЯ КАРТА РИСКОВ НАВИГАЦИИ В АРКТИКЕ<br>' +
                 f'<sub>Северный морской путь | Лёд: {ice_date} | Ветер: {wind_date}</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#333', family='Arial, sans-serif')
        ),
        height=850,
        showlegend=False,
        margin=dict(l=20, r=150, t=100, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest'
    )
    
    # Добавляем информацию
    annotations_text = (
        "<b>📊 Интерпретация уровней риска:</b><br><br>" +
        "<span style='color:#2166ac; font-size:20px'>●</span> <b>0.0-0.3: Низкий</b><br>" +
        "  Безопасная навигация<br><br>" +
        "<span style='color:#4393c3; font-size:20px'>●</span> <b>0.3-0.5: Умеренный</b><br>" +
        "  Навигация с осторожностью<br><br>" +
        "<span style='color:#fee090; font-size:20px'>●</span> <b>0.5-0.7: Повышенный</b><br>" +
        "  Требуется ледокол<br><br>" +
        "<span style='color:#f4a582; font-size:20px'>●</span> <b>0.7-0.9: Высокий</b><br>" +
        "  Навигация затруднена<br><br>" +
        "<span style='color:#b2182b; font-size:20px'>●</span> <b>0.9-1.0: Критический</b><br>" +
        "  Крайне опасно<br><br>" +
        "<i>💡 Вращайте карту мышью для просмотра</i>"
    )
    
    fig.add_annotation(
        text=annotations_text,
        xref="paper", yref="paper",
        x=0.01, y=0.02,
        xanchor='left', yanchor='bottom',
        showarrow=False,
        font=dict(size=11, color='#444'),
        bgcolor='rgba(255,255,255,0.97)',
        bordercolor='#333',
        borderwidth=2,
        borderpad=15,
        align='left'
    )
    
    return fig

# Создание интерактивной веб-карты
print("\n🌐 Создание интерактивной веб-карты...")

web_map = create_interactive_map(
    ice_risk=ice_risk,
    wind_risk=wind_risk_interpolated,
    combined_risk=combined_risk,
    ice_ds=ice_ds,
    ice_date=f"{month_name} 2024",
    wind_date=str(wind_ds.time.values[time_idx])[:19] + " UTC"
)

# Сохранение карты в HTML
output_file = f'arctic_risk_map_{end_date_str}.html'
# Оптимизация: сжатие HTML и отключение лишних опций
config = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}
web_map.write_html(
    output_file, 
    config=config,
    include_plotlyjs='cdn'  # Загрузка Plotly с CDN для уменьшения размера файла
)
print(f"\n✅ Интерактивная веб-карта сохранена!")
print(f"   📁 Файл: {output_file}")
print(f"   🌐 Откройте файл в браузере для просмотра")
print(f"   💡 Функции: вращение карты, масштабирование, детальная информация при наведении")

print("\n✅ Карты созданы успешно!")
print(f"\n📊 Статистика комплексного риска:")
print(f"   Минимальный риск: {np.nanmin(combined_risk):.3f}")
print(f"   Средний риск: {np.nanmean(combined_risk):.3f}")
print(f"   Максимальный риск: {np.nanmax(combined_risk):.3f}")
print(f"   Площадь с риском >0.7: {np.sum(combined_risk > 0.7) / combined_risk.size * 100:.1f}%")
