from flask import Flask, render_template, request, jsonify, send_file
import matplotlib
matplotlib.use('Agg')  # Бэкенд без GUI для серверной генерации
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import geopandas as gpd
from risk import calculate_risk_vectorized
import earthaccess
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from glob import glob
import plotly.graph_objects as go
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Загрузка береговой линии
land = gpd.read_file('ne_10m_land.shp')

# Определяем даты конца каждого месяца
END_DATES = [
    "20240131", "20240229", "20240331", "20240430",
    "20240515", "20240930", "20241031", "20241130", "20241231"
]

MONTH_MAPPING = {
    "20240131": 1,
    "20240229": 2,
    "20240331": 3,
    "20240430": 4,
    "20240515": 5,
    "20240930": 9,
    "20241031": 10,
    "20241130": 11,
    "20241231": 12
}

MONTH_NAMES_RU = {
    1: "Январь",
    2: "Февраль",
    3: "Март",
    4: "Апрель",
    5: "Май",
    9: "Сентябрь",
    10: "Октябрь",
    11: "Ноябрь",
    12: "Декабрь"
}

def get_ice_dataset(end_date_str):
    local_pattern = f"data/*/RDEFT4_{end_date_str}.nc"
    existing_files = glob(local_pattern)

    if existing_files:
        return xr.open_dataset(existing_files[0])
    
    try:
        auth = earthaccess.login()
        end_date = datetime.strptime(end_date_str, "%Y%m%d")
        
        results = earthaccess.search_data(
            short_name="RDEFT4",
            bounding_box=(30, 66, 180, 82),
            temporal=(end_date, end_date + timedelta(days=1))
        )

        if not results:
            return None

        local_files = earthaccess.download(results)
        return xr.open_dataset(local_files[0])
    
    except Exception as e:
        return None

def get_wind_dataset():
    now = datetime.now()
    dates_to_try = [
        (now - timedelta(days=1)).strftime("%Y%m%d"),
        now.strftime("%Y%m%d")
    ]

    urls = []
    for date_str in dates_to_try:
        for h in [0, 6, 12, 18]:
            url = f"https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date_str}/gfs_0p25_{h:02d}z"
            urls.append((url, date_str, h))

    for url, date_str, hour in reversed(urls):
        try:
            ds = xr.open_dataset(url, engine='pydap')
            return ds
        except Exception as e:
            continue
    
    return None

def calculate_wind_risk(wind_speed):
    wind_risk = np.zeros_like(wind_speed, dtype=float)
    
    mask1 = (wind_speed >= 0) & (wind_speed < 5)
    wind_risk[mask1] = wind_speed[mask1] / 25.0
    
    mask2 = (wind_speed >= 5) & (wind_speed < 10)
    wind_risk[mask2] = 0.2 + (wind_speed[mask2] - 5) / 25.0
    
    mask3 = (wind_speed >= 10) & (wind_speed < 15)
    wind_risk[mask3] = 0.4 + (wind_speed[mask3] - 10) / 25.0
    
    mask4 = (wind_speed >= 15) & (wind_speed < 20)
    wind_risk[mask4] = 0.6 + (wind_speed[mask4] - 15) / 25.0
    
    mask5 = wind_speed >= 20
    wind_risk[mask5] = 0.8 + np.minimum((wind_speed[mask5] - 20) / 50.0, 0.2)
    
    return np.clip(wind_risk, 0.0, 1.0)

def combine_risks(ice_risk, wind_risk, ice_weight=0.6, wind_weight=0.4):
    base_risk = np.maximum(ice_risk, wind_risk)
    weighted_sum = ice_risk * ice_weight + wind_risk * wind_weight
    combined = (base_risk * 0.6 + weighted_sum * 0.4)
    
    synergy_mask = (ice_risk > 0.3) & (wind_risk > 0.3)
    if np.any(synergy_mask):
        synergy_boost = ice_risk[synergy_mask] * wind_risk[synergy_mask] * 0.3
        combined[synergy_mask] += synergy_boost
    
    return np.clip(combined, 0.0, 1.0)

def interpolate_to_grid(source_lon, source_lat, source_data, target_lon, target_lat):
    if len(source_lon.shape) == 1:
        source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat)
    else:
        source_lon_grid = source_lon
        source_lat_grid = source_lat
    
    if len(target_lon.shape) == 1:
        target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)
    else:
        target_lon_grid = target_lon
        target_lat_grid = target_lat
    
    source_points = np.column_stack([
        source_lon_grid.ravel(),
        source_lat_grid.ravel()
    ])
    source_values = source_data.ravel()
    
    valid_mask = np.isfinite(source_values)
    source_points = source_points[valid_mask]
    source_values = source_values[valid_mask]
    
    target_points = np.column_stack([
        target_lon_grid.ravel(),
        target_lat_grid.ravel()
    ])
    
    interpolated = griddata(
        source_points,
        source_values,
        target_points,
        method='linear',
        fill_value=0.0
    )
    
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

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def calculate_route_length(lats, lons):
    total_distance = 0.0
    for i in range(len(lats) - 1):
        total_distance += haversine_distance(lats[i], lons[i], lats[i+1], lons[i+1])
    return total_distance

def get_risk_along_route(route_lats, route_lons, risk_grid, grid_lats, grid_lons):
    from scipy.interpolate import griddata
    
    if len(grid_lats.shape) == 2:
        lat_flat = grid_lats.ravel()
        lon_flat = grid_lons.ravel()
    else:
        lon_2d, lat_2d = np.meshgrid(grid_lons, grid_lats)
        lat_flat = lat_2d.ravel()
        lon_flat = lon_2d.ravel()
    
    risk_flat = risk_grid.ravel()
    
    valid_mask = np.isfinite(risk_flat)
    lat_valid = lat_flat[valid_mask]
    lon_valid = lon_flat[valid_mask]
    risk_valid = risk_flat[valid_mask]
    
    points = np.column_stack([lat_valid, lon_valid])
    route_points = np.column_stack([route_lats, route_lons])
    
    risks = griddata(points, risk_valid, route_points, method='linear', fill_value=0.0)
    risks = np.nan_to_num(risks, nan=0.0)
    
    return risks

def analyze_routes(routes_data, combined_risk, grid_lats, grid_lons):
    route_stats = []
    
    for route_name, route_info in routes_data.items():
        lats = route_info['lat']
        lons = route_info['lon']
        
        route_length = calculate_route_length(lats, lons)
        
        num_sample_points = max(100, int(route_length / 10))
        
        sample_lats = np.interp(
            np.linspace(0, len(lats) - 1, num_sample_points),
            np.arange(len(lats)),
            lats
        )
        sample_lons = np.interp(
            np.linspace(0, len(lons) - 1, num_sample_points),
            np.arange(len(lons)),
            lons
        )
        
        risks = get_risk_along_route(sample_lats, sample_lons, combined_risk, grid_lats, grid_lons)
        
        avg_risk = np.mean(risks)
        max_risk = np.max(risks)
        high_risk_ratio = np.sum(risks > 0.5) / len(risks)
        critical_risk_ratio = np.sum(risks > 0.7) / len(risks)
        
        risk_integral = np.trapz(risks) * route_length / len(risks)
        
        high_risk_distance = high_risk_ratio * route_length
        critical_risk_distance = critical_risk_ratio * route_length
        
        route_stats.append({
            'name': route_name,
            'length': route_length,
            'avg_risk': avg_risk,
            'max_risk': max_risk,
            'high_risk_points': int(high_risk_ratio * 100),
            'critical_risk_points': int(critical_risk_ratio * 100),
            'risks': risks,
            'risk_integral': risk_integral,
            'high_risk_distance': high_risk_distance,
            'critical_risk_distance': critical_risk_distance
        })
    
    all_avg_risks = [r['avg_risk'] for r in route_stats]
    max_avg_risk = max(all_avg_risks)
    
    if max_avg_risk < 0.3:
        best_route = min(route_stats, key=lambda r: r['length'])
        reason = "низкий общий риск, выбран кратчайший маршрут"
    else:
        for route in route_stats:
            risk_score = route['avg_risk'] * 100 + route['max_risk'] * 30
            
            length_penalty = (route['length'] / 1000) * 0.5
            
            critical_penalty = route['critical_risk_distance'] * 10
            high_penalty = route['high_risk_distance'] * 2
            
            route['score'] = risk_score + length_penalty + critical_penalty + high_penalty
        
        best_route = min(route_stats, key=lambda r: r['score'])
        
        if best_route['critical_risk_points'] > 0:
            reason = f"минимум критических участков ({best_route['critical_risk_points']}%)"
        elif best_route['high_risk_points'] > 30:
            reason = f"оптимальное соотношение длины и риска"
        else:
            reason = f"низкий средний риск ({best_route['avg_risk']:.3f})"
    
    return best_route, route_stats, reason

def create_ice_risk_map(ice_ds, ice_risk, month_name, end_date_str):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=90))
    ax.set_extent([20, 190, 66, 82], crs=ccrs.PlateCarree())
    
    ax.coastlines(resolution='50m')
    ax.add_geometries(land.geometry, crs=ccrs.PlateCarree(),
                     facecolor='whitesmoke', edgecolor='black', linewidth=0.5)
    
    img = ax.pcolormesh(ice_ds.lon.values, ice_ds.lat.values, ice_risk,
                        transform=ccrs.PlateCarree(),
                        cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1, alpha=0.7)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                      xlocs=np.arange(20, 200, 20), ylocs=np.arange(66, 84, 3))
    gl.top_labels = False
    gl.right_labels = False
    
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    plt.title(f"Карта ледовых рисков - {month_name} 2024\n(30-day average ending {end_date.strftime('%d.%m')})",
              fontsize=12, pad=20)
    
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.08)
    cbar.set_label('Риск (0 - низкий, 1 - высокий)')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return image_base64

def create_wind_map(wind_ds, u_wind, v_wind, wind_speed, lon, lat):
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=90))
    ax.set_extent([20, 190, 66, 82], crs=ccrs.PlateCarree())
    
    ax.coastlines(resolution='50m')
    ax.add_geometries(land.geometry, crs=ccrs.PlateCarree(),
                     facecolor='whitesmoke', edgecolor='black', linewidth=0.5)
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    pcm = ax.pcolormesh(lon_grid, lat_grid, wind_speed,
                       transform=ccrs.PlateCarree(),
                       cmap='viridis', shading='auto', alpha=0.7)
    
    cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.08)
    cbar.set_label('Скорость ветра (м/с)')
    
    step = 10
    u_dec = u_wind[::step, ::step]
    v_dec = v_wind[::step, ::step]
    lon_dec = lon[::step]
    lat_dec = lat[::step]
    lon_grid_dec, lat_grid_dec = np.meshgrid(lon_dec, lat_dec)
    
    Q = ax.quiver(lon_grid_dec, lat_grid_dec, u_dec, v_dec, 
                  transform=ccrs.PlateCarree(),
                  scale=300, width=0.0025, pivot='middle')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                     xlocs=np.arange(20, 200, 20), ylocs=np.arange(66, 84, 3))
    gl.top_labels = False
    gl.right_labels = False
    
    time_idx = 0
    plt.title(f"Карта ветра - Северный морской путь\nСкорость (цвет) и направление (стрелки)\n{str(wind_ds.time.values[time_idx])[:19]} UTC",
             fontsize=12, pad=20)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return image_base64

def create_combined_map(ice_ds, combined_risk, month_name, end_date_str):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=90))
    ax.set_extent([20, 190, 66, 82], crs=ccrs.PlateCarree())
    
    ax.coastlines(resolution='50m')
    ax.add_geometries(land.geometry, crs=ccrs.PlateCarree(),
                     facecolor='whitesmoke', edgecolor='black', linewidth=0.5)
    
    img = ax.pcolormesh(ice_ds.lon.values, ice_ds.lat.values, combined_risk,
                        transform=ccrs.PlateCarree(),
                        cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1, alpha=0.7)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                      xlocs=np.arange(20, 200, 20), ylocs=np.arange(66, 84, 3))
    gl.top_labels = False
    gl.right_labels = False
    
    plt.title(f"Комплексная карта рисков (Лёд + Ветер) - {month_name} 2024",
              fontsize=12, pad=20)
    
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.08)
    cbar.set_label('Комплексный риск (0 - низкий, 1 - высокий)')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return image_base64

def create_interactive_map(ice_risk, wind_risk, combined_risk, ice_ds, ice_date, wind_date, end_date_str):
    lon_2d = ice_ds.lon.values
    lat_2d = ice_ds.lat.values
    
    ice_con = ice_ds['ice_con'].values
    ice_thickness = ice_ds['sea_ice_thickness'].values
    
    ocean_mask = ((ice_con != -9999.0) | (ice_thickness != -9999.0)) & (lat_2d >= 60)
    
    combined_masked = combined_risk.copy()
    combined_masked[~ocean_mask] = np.nan
    
    lon_flat = lon_2d.ravel()
    lat_flat = lat_2d.ravel()
    risk_flat = combined_masked.ravel()
    
    valid = ~np.isnan(risk_flat)
    lon_valid = lon_flat[valid]
    lat_valid = lat_flat[valid]
    risk_valid = risk_flat[valid]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scattergeo(
            lon=lon_valid,
            lat=lat_valid,
            mode='markers',
            marker=dict(
                size=8,
                color=risk_valid,
                colorscale=[
                    [0.0, '#2166ac'],
                    [0.3, '#4393c3'],
                    [0.5, '#fee090'],
                    [0.7, '#f4a582'],
                    [0.9, '#d6604d'],
                    [1.0, '#b2182b']
                ],
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(text="Уровень<br>риска", side="right"),
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
    
    smp_routes = {
        'Маршрут 1': {
            'lat': [66.28136, 66.97840, 67.34440, 68.86238, 69.48712, 69.93301, 70.22509, 71.14641,
                   75.89244, 77.06427, 77.04507, 77.78273, 77.84217, 77.76934, 77.95960, 77.37989,
                   76.38259, 69.59466, 69.44223],
            'lon': [-169.16316+360, -171.44848+360, -173.89058+360, -178.73195+360, 179.04017, 176.34099,
                   170.30000, 168.71477, 152.56261, 141.55171, 126.08505, 104.70592, 102.92685, 97.92964,
                   92.20008, 68.16177, 59.07201, 35.09516, 35.56900],
            'color': 'red'
        },
        'Маршрут 2': {
            'lat': [66.28136, 66.97840, 67.34440, 68.86238, 69.48712, 69.93301, 70.22509, 71.14641,
                   75.89244, 76.69456, 76.42558, 77.78273, 77.84217, 77.12069, 76.41703, 76.32531,
                   73.90052, 73.68322, 70.29943, 69.44223],
            'lon': [-169.16316+360, -171.44848+360, -173.89058+360, -178.73195+360, 179.04017, 176.34099,
                   170.30000, 168.71477, 152.56261, 140.95190, 119.57277, 104.70592, 102.92685, 92.91948,
                   87.70580, 82.21977, 72.57057, 68.83539, 57.51315, 35.56900],
            'color': 'blue'
        },
        'Маршрут 3': {
            'lat': [66.28136, 66.97840, 67.34440, 68.86238, 69.48712, 69.93301, 70.22509, 70.85337,
                   74.30679, 74.55985, 74.39928, 77.78273, 77.25878, 76.00890, 74.34771, 73.90052,
                   73.68322, 70.29943, 69.44223],
            'lon': [-169.16316+360, -171.44848+360, -173.89058+360, -178.73195+360, 179.04017, 176.34099,
                   170.30000, 162.97369, 146.56445, 141.12327, 139.23814, 104.70592, 96.42121, 87.27781,
                   80.43000, 72.57057, 68.83539, 57.51315, 35.56900],
            'color': 'lime'
        },
        'Маршрут 4': {
            'lat': [66.28136, 66.97840, 67.34440, 68.86238, 69.48712, 69.93301, 70.22509, 69.81512,
                   70.06556, 69.70144, 69.81512, 69.75204, 69.96574, 70.77507, 71.30875, 71.08518,
                   71.66086, 72.32247, 72.82536, 72.95425, 71.80535, 72.15543, 73.07157, 73.64653,
                   74.10278, 73.60536, 73.80001, 73.68759, 73.54342, 74.02258, 74.11277, 74.11277,
                   74.63392, 74.95942, 75.35222, 75.92267, 76.62420, 76.87522, 76.97433, 77.51431,
                   77.78273, 77.63106, 76.55348, 76.48997, 76.31611, 75.34456, 73.83562, 73.90052,
                   73.68322, 70.29943, 69.44223],
            'lon': [-169.16316+360, -171.44848+360, -173.89058+360, -178.73195+360, 179.04017, 176.34099,
                   170.30000, 170.46525, 168.60419, 166.81613, 164.15277, 162.36420, 160.57613, 160.06525,
                   157.65683, 152.91298, 151.30737, 150.10315, 144.66597, 140.36000, 137.22176, 130.10598,
                   130.10598, 128.46387, 123.39160, 119.59651, 116.34880, 114.08634, 113.53897, 113.10108,
                   111.05757, 109.96284, 111.96985, 113.17406, 114.01336, 114.34178, 113.21055, 111.60494,
                   109.41547, 107.59091, 104.70592, 102.60759, 97.58845, 96.38230, 94.59253, 86.38293,
                   80.81908, 72.57057, 68.83539, 57.51315, 35.56900],
            'color': 'yellow'
        }
    }
    
    best_route, all_routes_stats, reason = analyze_routes(smp_routes, combined_risk, lat_2d, lon_2d)
    
    for route_name, route_data in smp_routes.items():
        is_best = (route_name == best_route['name'])
        
        fig.add_trace(
            go.Scattergeo(
                lon=route_data['lon'],
                lat=route_data['lat'],
                mode='lines+markers',
                line=dict(
                    width=4 if is_best else 2, 
                    color=route_data['color']
                ),
                marker=dict(
                    size=6 if is_best else 4, 
                    color=route_data['color']
                ),
                name=route_name + (' ⭐ РЕКОМЕНДУЕТСЯ' if is_best else ''),
                hovertemplate='<b>' + route_name + '</b><br>' +
                             'Широта: %{lat:.2f}°<br>' +
                             'Долгота: %{lon:.2f}°' +
                             '<extra></extra>',
                showlegend=True
            )
        )
    
    cities = {
        'Мурманск': {'lat': 68.9585, 'lon': 33.0827},
        'Архангельск': {'lat': 64.5401, 'lon': 40.5433},
        'Нарьян-Мар': {'lat': 67.6381, 'lon': 53.0574},
        'Сабетта': {'lat': 71.2819, 'lon': 72.0508},
        'Диксон': {'lat': 73.5069, 'lon': 80.5464},
        'Дудинка': {'lat': 69.4058, 'lon': 86.1778},
        'Тикси': {'lat': 71.6372, 'lon': 128.8719},
        'Певек': {'lat': 69.7011, 'lon': 170.3133}
    }
    
    city_lats = [city['lat'] for city in cities.values()]
    city_lons = [city['lon'] for city in cities.values()]
    city_names = list(cities.keys())
    
    fig.add_trace(
        go.Scattergeo(
            lon=city_lons,
            lat=city_lats,
            mode='markers+text',
            marker=dict(
                size=10,
                color='black',
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            text=city_names,
            textposition='top center',
            textfont=dict(size=10, color='black', family='Arial Black'),
            name='Города',
            hovertemplate='<b>%{text}</b><br>' +
                         'Широта: %{lat:.4f}°<br>' +
                         'Долгота: %{lon:.4f}°' +
                         '<extra></extra>',
            showlegend=True
        )
    )
    
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
    
    fig.update_layout(
        title=dict(
            text=f'⚓ КОМПЛЕКСНАЯ КАРТА РИСКОВ НАВИГАЦИИ В АРКТИКЕ<br>' +
                 f'<sub>Северный морской путь | Лёд: {ice_date} | Ветер: {wind_date}</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#333', family='Arial, sans-serif')
        ),
        height=850,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#333',
            borderwidth=1
        ),
        margin=dict(l=20, r=150, t=100, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest'
    )
    
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
    
    output_file = f'static/arctic_risk_map_{end_date_str}.html'
    os.makedirs('static', exist_ok=True)
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
    fig.write_html(output_file, config=config, include_plotlyjs='cdn')
    
    return output_file, best_route, all_routes_stats, reason

@app.route('/check_map/<month_index>')
def check_map(month_index):
    try:
        idx = int(month_index)
        end_date_str = END_DATES[idx]
        map_file = f'static/arctic_risk_map_{end_date_str}.html'
        exists = os.path.exists(map_file)
        return jsonify({'exists': exists})
    except:
        return jsonify({'exists': False})

@app.route('/')
def index():
    months = []
    for i, date in enumerate(END_DATES):
        month_num = MONTH_MAPPING[date]
        month_name = MONTH_NAMES_RU[month_num]
        months.append({
            'index': i,
            'name': month_name,
            'date': date
        })
    return render_template('index.html', months=months)

@app.route('/generate', methods=['POST'])
def generate_maps():
    try:
        month_index = int(request.json['month_index'])
        end_date_str = END_DATES[month_index]
        month_num = MONTH_MAPPING[end_date_str]
        month_name = MONTH_NAMES_RU[month_num]
        
        ice_ds = get_ice_dataset(end_date_str)
        if ice_ds is None:
            return jsonify({'error': 'Не удалось загрузить данные о льде'}), 500
        
        # Загрузка данных о ветре
        wind_ds = get_wind_dataset()
        if wind_ds is None:
            return jsonify({'error': 'Не удалось загрузить данные о ветре'}), 500
        
        ice_thickness_data = ice_ds['sea_ice_thickness'].values.copy()
        ice_thickness_data[ice_thickness_data == -9999.0] = 0
        ice_thickness_data = np.nan_to_num(ice_thickness_data, nan=0.0)
        
        ice_concentration_data = ice_ds['ice_con'].values.copy()
        ice_concentration_data[ice_concentration_data == -9999.0] = 0
        ice_concentration_data = np.nan_to_num(ice_concentration_data, nan=0.0)
        
        ice_risk = calculate_risk_vectorized(ice_concentration_data, ice_thickness_data)
        
        time_idx = 0
        lon_min, lon_max = 20, 190
        lat_min, lat_max = 66, 82
        
        lon_indices = (wind_ds.lon >= lon_min) & (wind_ds.lon <= lon_max)
        lat_indices = (wind_ds.lat >= lat_min) & (wind_ds.lat <= lat_max)
        
        lon = wind_ds.lon[lon_indices].values
        lat = wind_ds.lat[lat_indices].values
        
        u_wind = wind_ds['ugrd10m'][time_idx, lat_indices, lon_indices].values
        v_wind = wind_ds['vgrd10m'][time_idx, lat_indices, lon_indices].values
        
        if hasattr(u_wind, 'filled'):
            u_wind = u_wind.filled(0.0)
        if hasattr(v_wind, 'filled'):
            v_wind = v_wind.filled(0.0)
        
        u_wind = np.nan_to_num(u_wind, nan=0.0)
        v_wind = np.nan_to_num(v_wind, nan=0.0)
        
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
        
        wind_risk_data = calculate_wind_risk(wind_speed)
        
        wind_risk_interpolated = interpolate_to_grid(
            lon, lat, wind_risk_data,
            ice_ds.lon.values, ice_ds.lat.values
        )
        
        combined_risk = combine_risks(ice_risk, wind_risk_interpolated)
        
        ice_map_base64 = create_ice_risk_map(ice_ds, ice_risk, month_name, end_date_str)
        wind_map_base64 = create_wind_map(wind_ds, u_wind, v_wind, wind_speed, lon, lat)
        combined_map_base64 = create_combined_map(ice_ds, combined_risk, month_name, end_date_str)
        
        interactive_map_path, best_route, all_routes_stats, reason = create_interactive_map(
            ice_risk, wind_risk_interpolated, combined_risk,
            ice_ds, f"{month_name} 2024",
            str(wind_ds.time.values[time_idx])[:19] + " UTC",
            end_date_str
        )
        
        routes_info = []
        for route_stat in all_routes_stats:
            routes_info.append({
                'name': route_stat['name'],
                'length': round(route_stat['length'], 1),
                'avg_risk': round(route_stat['avg_risk'], 3),
                'max_risk': round(route_stat['max_risk'], 3),
                'high_risk_points': int(route_stat['high_risk_points']),
                'critical_risk_points': int(route_stat['critical_risk_points'])
            })
        
        return jsonify({
            'success': True,
            'ice_map': ice_map_base64,
            'wind_map': wind_map_base64,
            'combined_map': combined_map_base64,
            'interactive_map_url': f'/{interactive_map_path}',
            'month_name': month_name,
            'best_route': best_route['name'],
            'best_route_reason': reason,
            'routes_stats': routes_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
