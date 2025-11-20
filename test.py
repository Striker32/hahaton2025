import xarray as xr
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd

# Загрузка береговой линии (как в main.py)
land = gpd.read_file('ne_10m_land.shp')

# Пробуем сегодняшнюю и вчерашнюю даты
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

for url_info in urls:
    print(url_info[0])

if not urls:
    print("Нет доступных данных")
    exit(1)

# Пробуем URL-ы по очереди, пока не найдём рабочий
ds = None
for url, date_str, hour in reversed(urls):  # Начинаем с самых свежих
    try:
        print(f"\nПопытка загрузить данные: {date_str} {hour:02d}:00 UTC")
        ds = xr.open_dataset(url, engine='pydap')
        print(f"✅ Успешно загружены данные за {date_str} {hour:02d}:00 UTC")
        break
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        continue

if ds is None:
    print("\n❌ Не удалось загрузить данные ни с одного URL")
    exit(1)

time_idx = 0

# границы области - полный Северный морской путь (как в карте рисков)
lon_min, lon_max = 20, 190
lat_min, lat_max = 66, 82

# GFS использует долготы 0..360, наши границы уже в этом диапазоне
# сделаем подвыборку по области (slice работает, если lon монотонен)
ds_sub = ds.sel(
    time=ds.time[time_idx],
    lon=slice(lon_min, lon_max),
    lat=slice(lat_min, lat_max)
)

# извлекаем массивы u/v (величины в м/с)
u = ds_sub['ugrd10m'].values  # shape: (lat, lon) после выборки по time
v = ds_sub['vgrd10m'].values

lon = ds_sub['lon'].values
lat = ds_sub['lat'].values

# вычисляем скорость и направление (метеорологическая: откуда дует)
speed = np.sqrt(u**2 + v**2)

# направление (градусы, откуда дует, 0 = север, 90 = восток)
# стандартное: dir = atan2(-u, -v) в радианах -> градусы -> 0..360
direction_deg = (np.degrees(np.arctan2(-u, -v)) + 360) % 360

# Визуализация
fig = plt.figure(figsize=(15, 10))  # Такой же размер как в main.py
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=90))
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Добавление береговой линии (как в main.py)
ax.coastlines(resolution='50m')
ax.add_geometries(land.geometry, crs=ccrs.PlateCarree(),
                 facecolor='whitesmoke', edgecolor='black', linewidth=0.5)

# pcolormesh для скорости (сетка lon/lat -> нужно сделать meshgrid)
Lon, Lat = np.meshgrid(lon, lat)
pcm = ax.pcolormesh(Lon, Lat, speed, transform=ccrs.PlateCarree(),
                    cmap='viridis', shading='auto', alpha=0.7)
cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.08)
cbar.set_label('Скорость ветра (м/с)')

# стрелки: декриментируем плотность, чтобы не было «шеренг»
step = 6  # подберите под плотность вашей сетки (6 -> каждые 6 точек)
u_dec = u[::step, ::step]
v_dec = v[::step, ::step]
lon_dec = lon[::step]
lat_dec = lat[::step]
Lon2, Lat2 = np.meshgrid(lon_dec, lat_dec)

# рисуем стрелки (quiver)
Q = ax.quiver(Lon2, Lat2, u_dec, v_dec, transform=ccrs.PlateCarree(),
              scale=300, width=0.0025, pivot='middle')
# легенда стрелок: примерная подпись (не автоматическая)
ax.quiverkey(Q, 0.9, -0.05, 10, '10 м/с', labelpos='E')

# Добавление сетки с улучшенными параметрами (как в main.py)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  xlocs=np.arange(20, 200, 20), ylocs=np.arange(66, 84, 3))
gl.top_labels = False
gl.right_labels = False

plt.title(f"Карта ветра - Северный морской путь\nСкорость (цвет) и направление (стрелки)\n{str(ds.time.values[time_idx])}",
          fontsize=12, pad=20)
plt.tight_layout()
plt.show()