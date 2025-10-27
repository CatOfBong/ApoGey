import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime

# ====================== ФИЗИЧЕСКИЕ КОНСТАНТЫ ======================
R = 287.058  # Газовая постоянная для воздуха (Дж/(кг·K))
gamma = 1.4   # Показатель адиабаты для воздуха
R_earth = 6371000.0  # Радиус Земли (м)
g0 = 9.80665  # Гравитация на уровне моря (м/с²)
KGF_TO_NEWTON = 9.80665  # Коэффициент перевода кгс в Н

# ====================== МОДЕЛЬ АТМОСФЕРЫ ======================
# Слои атмосферы: (h_min, h_max, L, T_base, p_base)
ATMOSPHERE_LAYERS = [
    (0, 11000, -0.0065, 288.15, 101325.0),    # Тропосфера
    (11000, 20000, 0, 216.65, 22632.040),      # Стратосфера (нижняя)
    (20000, 32000, 0.001, 216.65, 5474.88),    # Стратосфера (средняя)
    (32000, 47000, 0.0028, 228.65, 868.02),    # Стратосфера (верхняя)
    (47000, 100000, 0, 270.65, 110.91)         # Мезосфера
]

def gravity(y):
    """Учет изменения гравитации с высотой"""
    return g0 * (R_earth / (R_earth + y))**2

def atmosphere(y):
    """Рассчитывает параметры атмосферы на заданной высоте (0-100 км)"""
    # Для отрицательных высот используем параметры уровня моря
    if y < 0:
        p = ATMOSPHERE_LAYERS[0][4]
        T = ATMOSPHERE_LAYERS[0][3]
        rho = p / (R * T)
        c = np.sqrt(gamma * R * T)
        return rho, c, T, p
    
    # Для высот выше 100 км - экстраполяция
    if y > 100000:
        T = 270.65  # Температура мезопаузы
        p = 110.91 * np.exp(-g0 * (y - 47000) / (R * T))
        rho = p / (R * T)
        c = np.sqrt(gamma * R * T)
        return rho, c, T, p
    
    # Поиск соответствующего слоя атмосферы
    for layer in ATMOSPHERE_LAYERS:
        h_min, h_max, L, T_base, p_base = layer
        if h_min <= y <= h_max:
            dy = y - h_min
            if L == 0:  # Изотермический слой
                T = T_base
                p = p_base * np.exp(-g0 * dy / (R * T_base))
            else:
                T = T_base + L * dy
                p = p_base * (T / T_base) ** (-g0/(R*L))
            
            rho = p / (R * T)
            c = np.sqrt(gamma * R * T)
            return rho, c, T, p
    
    return 0.0, 300.0, 270.65, 0.0

def calculate_dynamic_viscosity(T):
    """Рассчитывает динамическую вязкость воздуха по температуре (формула Сазерленда)"""
    T0 = 273.15  # Референсная температура (0°C в К)
    mu0 = 1.716e-5  # Вязкость при T0 (Па·с)
    S = 110.4  # Константа Сазерленда для воздуха
    
    mu = mu0 * (T / T0)**1.5 * (T0 + S) / (T + S)
    return mu

def calculate_reynolds(rho, V, L, mu):
    """Рассчитывает число Рейнольдса"""
    return rho * V * L / mu

# ====================== АЭРОДИНАМИЧЕСКИЕ РАСЧЕТЫ ======================
def calculate_lift_coefficient(AR, sweep_angle=0, mach=0, reynolds=1e6, alpha=0):
    """
    Рассчитывает коэффициент подъемной силы для крыла
    
    Parameters:
    AR - удлинение крыла
    sweep_angle - угол стреловидности (град)
    mach - число Маха
    reynolds - число Рейнольдса
    alpha - угол атаки (град)
    """
    # Базовый коэффициент подъемной силы (линейная теория)
    alpha_rad = np.radians(alpha)
    CL_alpha = 2 * np.pi * AR / (2 + np.sqrt(4 + (AR / 0.95)**2 * (1 + np.tan(np.radians(sweep_angle))**2)))
    
    # Поправка на число Маха
    beta = np.sqrt(1 - mach**2) if mach < 1 else np.sqrt(mach**2 - 1)
    CL_alpha_mach = CL_alpha / beta if mach < 1 else CL_alpha / beta
    
    CL = CL_alpha_mach * alpha_rad
    
    # Поправка на число Рейнольдса (упрощенно)
    Re_correction = 1.0 - 0.1 * np.log10(reynolds / 1e6)
    CL *= Re_correction
    
    return CL

def calculate_drag_coefficient(CD0, AR, CL, e=0.85, mach=0, sweep_angle=0):
    """
    Рассчитывает полный коэффициент сопротивления
    
    Parameters:
    CD0 - коэффициент сопротивления при нулевой подъемной силе
    AR - удлинение крыла
    CL - коэффициент подъемной силы
    e - коэффициент Освальда (эффективность крыла)
    mach - число Маха
    sweep_angle - угол стреловидности (град)
    """
    # Индуктивное сопротивление
    CDi = CL**2 / (np.pi * AR * e)
    
    # Волновое сопротивление (упрощенная модель)
    CD_wave = 0
    if mach > 0.8:
        # Эмпирическая формула для волнового сопротивления
        CD_wave = 0.02 * (mach - 0.8)**2 * (1 + 0.1 * np.tan(np.radians(sweep_angle)))
    
    return CD0 + CDi + CD_wave

def calculate_lift_force(rho, V, S, CL):
    """Рассчитывает подъемную силу"""
    return 0.5 * rho * V**2 * S * CL

def calculate_drag_force(rho, V, S, CD):
    """Рассчитывает силу сопротивления"""
    return 0.5 * rho * V**2 * S * CD

def calculate_thrust_required(drag, efficiency=0.8):
    """Рассчитывает требуемую тягу с учетом эффективности двигательной установки"""
    return drag / efficiency

def calculate_power_required(thrust, V, efficiency=0.8):
    """Рассчитывает требуемую мощность"""
    return thrust * V / efficiency

def kgf_to_newton(kgf):
    """Переводит килограмм-силы в Ньютоны"""
    return kgf * KGF_TO_NEWTON

def newton_to_kgf(newton):
    """Переводит Ньютоны в килограмм-силы"""
    return newton / KGF_TO_NEWTON

# ====================== ОСНОВНЫЕ РАСЧЕТЫ ======================
def analyze_wing_performance(altitudes, velocities, CD0_range, mass, wingspan, chord, 
                           thrust_available_kgf=None, power_available=None,
                           sweep_angle=0, alpha=5, e=0.85, propulsion_efficiency=0.8):
    """
    Основная функция анализа характеристик крыла
    
    Parameters:
    altitudes - диапазон высот (м)
    velocities - диапазон скоростей (м/с)
    CD0_range - диапазон коэффициентов сопротивления
    mass - масса объекта (кг)
    wingspan - размах крыла (м)
    chord - средняя аэродинамическая хорда (м)
    thrust_available_kgf - доступная тяга двигателя (кгс)
    power_available - доступная мощность двигателя (Вт)
    sweep_angle - угол стреловидности (град)
    alpha - угол атаки (град)
    e - коэффициент Освальда
    propulsion_efficiency - КПД двигательной установки
    """
    
    results = []
    S = wingspan * chord  # Площадь крыла
    AR = wingspan**2 / S  # Удлинение крыла
    
    # Конвертируем тягу из кгс в Ньютоны если задана
    thrust_available_N = kgf_to_newton(thrust_available_kgf) if thrust_available_kgf is not None else None
    
    for altitude in altitudes:
        # Атмосферные условия на высоте
        rho, c, T, p = atmosphere(altitude)
        mu = calculate_dynamic_viscosity(T)
        g = gravity(altitude)
        W = mass * g  # Вес
        
        for V in velocities:
            mach = V / c if c > 0 else 0
            L_ref = chord  # Характерная длина для числа Рейнольдса
            Re = calculate_reynolds(rho, V, L_ref, mu)
            
            for CD0 in CD0_range:
                # Расчет коэффициента подъемной силы
                CL = calculate_lift_coefficient(AR, sweep_angle, mach, Re, alpha)
                
                # Расчет коэффициента сопротивления
                CD = calculate_drag_coefficient(CD0, AR, CL, e, mach, sweep_angle)
                
                # Расчет аэродинамических сил
                L = calculate_lift_force(rho, V, S, CL)  # Подъемная сила
                D = calculate_drag_force(rho, V, S, CD)   # Сопротивление
                
                # Расчет требуемых тяги и мощности
                thrust_req = calculate_thrust_required(D, propulsion_efficiency)
                power_req = calculate_power_required(thrust_req, V, propulsion_efficiency)
                
                # Проверка возможности полета
                thrust_check = thrust_available_N is None or thrust_req <= thrust_available_N
                power_check = power_available is None or power_req <= power_available
                lift_check = L >= W
                
                can_fly = lift_check and thrust_check and power_check
                
                # Аэродинамическое качество
                K = CL / CD if CD > 0 else 0
                
                result = {
                    'Высота_м': altitude,
                    'Скорость_мс': V,
                    'Число_Маха': mach,
                    'Коэф_подъемной_силы_CL': CL,
                    'Коэф_сопротивления_CD': CD,
                    'CD0': CD0,
                    'Подъемная_сила_Н': L,
                    'Сопротивление_Н': D,
                    'Требуемая_тяга_Н': thrust_req,
                    'Требуемая_тяга_кгс': newton_to_kgf(thrust_req),
                    'Требуемая_мощность_Вт': power_req,
                    'Доступная_тяга_кгс': thrust_available_kgf,
                    'Доступная_мощность_Вт': power_available,
                    'Аэродинамическое_качество': K,
                    'Число_Рейнольдса': Re,
                    'Плотность_воздуха_кгм3': rho,
                    'Скорость_звука_мс': c,
                    'Температура_K': T,
                    'Давление_Па': p,
                    'Вес_Н': W,
                    'Удлинение_крыла': AR,
                    'Площадь_крыла_м2': S,
                    'Возможность_полета': can_fly
                }
                
                results.append(result)
    
    return results

def save_to_csv(filename, results, parameters):
    """Сохраняет результаты в CSV файл"""
    # Создаем директорию если нужно
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Записываем заголовок с параметрами
        writer.writerow(["Анализ характеристик крылатого объекта"])
        writer.writerow([f"Время расчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow([])
        
        # Параметры расчета
        writer.writerow(["Параметры расчета:"])
        for key, value in parameters.items():
            writer.writerow([key, str(value)])
        writer.writerow([])
        
        # Основные данные
        writer.writerow(["Результаты расчетов:"])
        
        if results:
            # Заголовки таблицы
            headers = list(results[0].keys())
            writer.writerow(headers)
            
            # Записываем данные
            for row in results:
                writer.writerow([row[key] for key in headers])
    
    print(f"Данные сохранены в файл: {filename}")

# ====================== ПАРАМЕТРЫ РАСЧЕТА ======================
if __name__ == "__main__":
    # Диапазоны параметров
    altitudes = np.arange(30, 151, 10)  # Высоты от 30 до 150 м с шагом 10 м
    velocities = np.arange(190, 221, 25)  # Скорости от 190 до 220 м/с с шагом 25 м/с
    CD0_range = np.arange(0.02, 0.41, 0.02)  # Коэффициенты сопротивления от 0.02 до 0.4
    
    # Параметры объекта
    mass = 2400  # Масса, кг
    wingspan = 3.0  # Размах крыла, м
    chord = 0.6  # Хорда крыла, м
    
    # Параметры двигательной установки (можно задать либо тягу, либо мощность)
    thrust_available_kgf = 450  # Доступная тяга, кгс
    power_available = None  # Доступная мощность, Вт (если None - используется тяга)
    
    # Дополнительные параметры
    sweep_angle = 8  # Угол стреловидности, град
    alpha = 10  # Угол атаки, град
    e = 0.85  # Коэффициент Освальда
    propulsion_efficiency = 0.8  # КПД двигательной установки
    
    parameters = {
        "Масса_кг": mass,
        "Размах_крыла_м": wingspan,
        "Хорда_м": chord,
        "Площадь_крыла_м2": wingspan * chord,
        "Удлинение_крыла": wingspan**2 / (wingspan * chord),
        "Доступная_тяга_кгс": thrust_available_kgf,
        "Доступная_тяга_Н": kgf_to_newton(thrust_available_kgf) if thrust_available_kgf else "Не задана",
        "Доступная_мощность_Вт": power_available if power_available else "Не задана",
        "Угол_стреловидности_град": sweep_angle,
        "Угол_атаки_град": alpha,
        "Коэффициент_Освальда": e,
        "КПД_двигательной_установки": propulsion_efficiency,
        "Диапазон_высот_м": f"{altitudes[0]} - {altitudes[-1]}",
        "Диапазон_скоростей_мс": f"{velocities[0]} - {velocities[-1]}",
        "Диапазон_CD0": f"{CD0_range[0]:.3f} - {CD0_range[-1]:.3f}"
    }
    
    # Выполнение расчетов
    print("Выполнение аэродинамических расчетов...")
    results = analyze_wing_performance(
        altitudes=altitudes,
        velocities=velocities,
        CD0_range=CD0_range,
        mass=mass,
        wingspan=wingspan,
        chord=chord,
        thrust_available_kgf=thrust_available_kgf,
        power_available=power_available,
        sweep_angle=sweep_angle,
        alpha=alpha,
        e=e,
        propulsion_efficiency=propulsion_efficiency
    )
    
    # Сохранение результатов
    csv_filename = os.path.join(os.getcwd(), "wing_analysis_results.csv")
    save_to_csv(csv_filename, results, parameters)
    
    print(f"\nРасчет завершен. Обработано {len(results)} комбинаций параметров.")
    
    # Статистика результатов
    flyable_count = sum(1 for r in results if r['Возможность_полета'])
    print(f"Возможных режимов полета: {flyable_count} из {len(results)}")
    
    if flyable_count > 0:
        # Находим режим с максимальным аэродинамическим качеством
        best_K = max(results, key=lambda x: x['Аэродинамическое_качество'] if x['Возможность_полета'] else 0)
        print(f"Лучшее аэродинамическое качество: {best_K['Аэродинамическое_качество']:.2f} "
              f"(высота: {best_K['Высота_м']} м, скорость: {best_K['Скорость_мс']} м/с)")