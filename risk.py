import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------
# Шаг 1: Определение нечетких переменных
# ---------------------------
# Входные переменные:
# Допустим:
# ice_concentration в диапазоне [0, 100]%
# ice_thickness в диапазоне [0, 3] метров

ice_concentration = ctrl.Antecedent(np.arange(0, 101, 1), 'ice_concentration')
ice_thickness = ctrl.Antecedent(np.arange(1, 3.1, 0.1), 'ice_thickness')

# Выходная переменная: риск в диапазоне [0, 1] (0 низкий риск, 1 высокий риск)
risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'risk')

# ---------------------------
# Шаг 2: Определение функций принадлежности
# ---------------------------
# Для концентрации льда определяем нечеткие множества: Низкая, Средняя, Высокая
ice_concentration['low'] = fuzz.trapmf(ice_concentration.universe, [0, 0, 20, 40])
ice_concentration['medium'] = fuzz.trapmf(ice_concentration.universe, [30, 50, 60, 80])
ice_concentration['high'] = fuzz.trapmf(ice_concentration.universe, [70, 90, 100, 100])

# Для толщины льда определяем нечеткие множества: Тонкий, Умеренный, Толстый
ice_thickness['thin'] = fuzz.trapmf(ice_thickness.universe, [0, 0, 0.5, 1.0])
ice_thickness['moderate'] = fuzz.trimf(ice_thickness.universe, [0.8, 1.5, 2.2])
ice_thickness['thick'] = fuzz.trapmf(ice_thickness.universe, [1.8, 2.4, 3.0, 3.0])

# Для риска определяем нечеткие множества: Низкий, Средний, Высокий
# Эти значения могут быть скорректированы в зависимости от желаемого распределения риска
risk['low'] = fuzz.trapmf(risk.universe, [0, 0, 0.3, 0.5])
risk['medium'] = fuzz.trimf(risk.universe, [0.4, 0.5, 0.6])
risk['high'] = fuzz.trapmf(risk.universe, [0.5, 0.7, 1.0, 1.0])

# ---------------------------
# Шаг 3: Определение нечетких правил
# ---------------------------
# Примеры правил (адаптируйте их на основе экспертных знаний):
# Если концентрация льда низкая и толщина тонкая -> риск низкий
rule1 = ctrl.Rule(ice_concentration['low'] & ice_thickness['thin'], risk['low'])

# Если концентрация льда средняя и толщина умеренная -> риск средний
rule2 = ctrl.Rule(ice_concentration['medium'] & ice_thickness['moderate'], risk['medium'])

# Если концентрация льда высокая и толщина большая -> риск высокий
rule3 = ctrl.Rule(ice_concentration['high'] & ice_thickness['thick'], risk['high'])

# Если концентрация льда высокая и толщина умеренная -> риск высокий (так как толстый лед и высокая концентрация оба вносят вклад)
rule4 = ctrl.Rule(ice_concentration['high'] & ice_thickness['moderate'], risk['high'])

# Если концентрация льда средняя и толщина большая -> риск стремится к средне-высокому
rule5 = ctrl.Rule(ice_concentration['medium'] & ice_thickness['thick'], risk['high'])

# Можно добавить более детальные правила для отражения реальных условий

# ---------------------------
# Шаг 4: Построение и моделирование нечеткой системы управления
# ---------------------------
risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])

def calculate_risk(ice_concentration_value, ice_thickness_value):
    # Input validation
    if np.isnan(ice_concentration_value) or np.isnan(ice_thickness_value):
        return 0.0  # return minimum risk for missing data
    
    # Для концентрации льда < 70%, используем линейную шкалу риска
    if ice_concentration_value < 70:
        # Преобразуем концентрацию льда в риск (0-70% -> 0-0.7)
        return (ice_concentration_value / 100)
    
    # Для концентрации льда >= 70% используем нечеткую логику
    # Clip values to valid ranges
    ice_concentration_value = np.clip(ice_concentration_value, 0, 100)
    ice_thickness_value = np.clip(ice_thickness_value, 1.0, 3.0)
    
    try:
        risk_calc = ctrl.ControlSystemSimulation(risk_ctrl)
        risk_calc.input['ice_concentration'] = ice_concentration_value
        risk_calc.input['ice_thickness'] = ice_thickness_value
        risk_calc.compute()
        return risk_calc.output['risk']
    except:
        return 0.0  # return minimum risk if calculation fails

# Добавляем векторизованную функцию после calculate_risk
def calculate_risk_vectorized(ice_concentration_array, ice_thickness_array):
    """
    Векторизованная версия расчёта рисков для массивов данных
    
    Parameters:
    -----------
    ice_concentration_array : np.ndarray
        Массив значений концентрации льда (0-100%)
    ice_thickness_array : np.ndarray
        Массив значений толщины льда (метры)
    
    Returns:
    --------
    np.ndarray
        Массив значений рисков (0-1)
    """
    # Создаём выходной массив
    risk_array = np.zeros_like(ice_concentration_array, dtype=float)
    
    # Обрабатываем NaN значения
    valid_mask = ~(np.isnan(ice_concentration_array) | np.isnan(ice_thickness_array))
    
    # Для концентрации < 70% используем линейную шкалу
    low_concentration_mask = valid_mask & (ice_concentration_array < 70)
    risk_array[low_concentration_mask] = ice_concentration_array[low_concentration_mask] / 100
    
    # Для концентрации >= 70% используем нечёткую логику
    high_concentration_mask = valid_mask & (ice_concentration_array >= 70) & (ice_thickness_array > 0)
    
    if np.any(high_concentration_mask):
        # Получаем индексы для обработки
        high_conc_indices = np.where(high_concentration_mask)
        
        # Клиппируем значения
        conc_values = np.clip(ice_concentration_array[high_conc_indices], 0, 100)
        thick_values = np.clip(ice_thickness_array[high_conc_indices], 1.0, 3.0)
        
        # Вычисляем риски для каждого элемента
        for idx in range(len(conc_values)):
            i, j = high_conc_indices[0][idx], high_conc_indices[1][idx]
            try:
                risk_calc = ctrl.ControlSystemSimulation(risk_ctrl)
                risk_calc.input['ice_concentration'] = float(conc_values[idx])
                risk_calc.input['ice_thickness'] = float(thick_values[idx])
                risk_calc.compute()
                risk_array[i, j] = risk_calc.output['risk']
            except:
                risk_array[i, j] = 0.0
    
    return risk_array

# Убираем тестовый код и оставляем только если запускаем файл напрямую
if __name__ == '__main__':
    # Тестовый пример
    test_risk = calculate_risk(65, 1.8)
    print("Test risk:", test_risk)

    # Визуализация
    import matplotlib.pyplot as plt
    ice_concentration.view()
    ice_thickness.view()
    risk.view()
    plt.show()