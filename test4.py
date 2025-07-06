import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import csv
from datetime import datetime

# Настройка стиля графиков
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'

# ====================== ПАРАМЕТРЫ СИМУЛЯЦИИ ======================
# Константы стандартной атмосферы
T0 = 288.15      # Температура на уровне моря (K)
p0 = 101325.0    # Давление на уровне моря (Па)
L = 0.0065       # Температурный градиент (K/м)
g0 = 9.80665     # Ускорение свободного падения (м/с²)
R = 287.058      # Газовая постоянная (Дж/(кг·K))
gamma = 1.4      # Показатель адиабаты
T11 = 216.65     # Температура на 11 км (K)
p11 = p0 * (T11/T0) ** (g0/(R*L))  # Давление на 11 км (Па)

# Параметры ракеты
ROCKET_PARAMS = {
    'diameter': 0.06,       # Диаметр ракеты (м)
    'm0': 4.0,              # Стартовая масса (кг)
    'm_final': 3.0,         # Масса после выгорания топлива (кг)
    'thrust': 600.0,        # Тяга (Н)
    'burn_time': 1.97,      # Время работы двигателя (с)
    'g': 9.81               # Гравитация (м/с²)
}

# Рассчитываем площадь поперечного сечения
A = np.pi * (ROCKET_PARAMS['diameter']/2)**2

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def atmosphere(y):
    """Рассчитывает параметры атмосферы на заданной высоте"""
    if y <= 11000:
        T = T0 - L * y
        p = p0 * (T / T0) ** (g0/(R*L))
    else:
        T = T11
        p = p11 * np.exp(-g0 * (y - 11000) / (R * T11))
    
    rho = p / (R * T)
    c = np.sqrt(gamma * R * T)
    return rho, c

def calculate_Cd(M):
    """Аппроксимация коэффициента сопротивления в зависимости от числа Маха"""
    if M < 0.8:
        return 0.1
    elif M < 1.0:
        return 0.1 + 0.9 * (M - 0.8) / 0.2
    elif M < 1.2:
        return 1.0 - 0.8 * (M - 1.0) / 0.2
    else:
        return max(0.1, 0.4 / (M**0.6))

def rocket_motion(t, state, m0, m_final, thrust, burn_time, g):
    """Дифференциальные уравнения движения ракеты"""
    y, v = state
    mass_flow = (m0 - m_final) / burn_time
    
    # Атмосферные параметры (кэшируем для текущей высоты)
    rho, c = atmosphere_cache.get(y, (None, None))
    if rho is None:
        rho, c = atmosphere(max(y, 0))
        atmosphere_cache[y] = (rho, c)
    
    M = abs(v) / c if c > 0 else 0
    Cd = calculate_Cd(M)
    
    F_drag = 0.5 * rho * v**2 * Cd * A
    drag_accel = -np.copysign(F_drag, v)
    
    if t <= burn_time:
        m = m0 - mass_flow * t
        dvdt = thrust/m - g + drag_accel/m
    else:
        m = m_final
        dvdt = -g + drag_accel/m
    
    return [v, dvdt]

def save_to_csv(filename, summary, time_points, data_columns, column_names):
    """Сохраняет данные в CSV файл с заголовком"""
    # Создаем директорию если нужно
    dirname = os.path.dirname(filename)
    if dirname:  # Проверяем что путь не пустой
        os.makedirs(dirname, exist_ok=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Записываем заголовок с параметрами
        writer.writerow(["Расчет траектории ракеты"])
        writer.writerow([f"Время расчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow([])
        
        # Параметры симуляции
        writer.writerow(["Параметры ракеты:"])
        for key, value in ROCKET_PARAMS.items():
            writer.writerow([key, str(value)])
        writer.writerow(["Площадь поперечного сечения (A):", f"{A:.6f} м²"])
        writer.writerow([])
        
        # Ключевые результаты
        writer.writerow(["Ключевые результаты:"])
        for key, value in summary.items():
            writer.writerow([key, str(value)])
        writer.writerow([])
        
        # Основные данные
        writer.writerow(["Детальные данные полета:"])
        writer.writerow(column_names)
        
        # Записываем данные
        for i in range(len(time_points)):
            row = [time_points[i]] + [col[i] for col in data_columns]
            writer.writerow(row)
    
    print(f"\nДанные сохранены в файл: {filename}")

# ====================== ПОДГОТОВКА ДАННЫХ ======================
# Кэш для ускорения расчетов атмосферы
atmosphere_cache = {}

# Валидация параметров
assert ROCKET_PARAMS['m0'] > ROCKET_PARAMS['m_final'], "Стартовая масса должна быть больше конечной"
assert ROCKET_PARAMS['thrust'] > 0, "Тяга должна быть положительной"
assert ROCKET_PARAMS['burn_time'] > 0, "Время работы двигателя должно быть положительным"

# Параметры для решателя
t_end = 30.0
args = tuple(ROCKET_PARAMS.values())[1:]  # Исключаем diameter

# Решение дифференциальных уравнений
sol = solve_ivp(rocket_motion, [0, t_end], [0, 0], 
                args=args,
                method='RK45', rtol=1e-8, atol=1e-10,
                dense_output=True, max_step=0.1)

# Основные результаты
time_points = sol.t
altitudes = sol.y[0]
velocities = sol.y[1]

# Рассчитываем дополнительные параметры
rho_values = []
c_values = []
M_values = []
Cd_values = []
F_drag_values = []
acceleration = np.zeros_like(time_points)

for i, t in enumerate(time_points):
    rho, c = atmosphere(max(altitudes[i], 0))
    M = abs(velocities[i]) / c if c > 0 else 0
    Cd = calculate_Cd(M)
    F_drag = 0.5 * rho * velocities[i]**2 * Cd * A
    
    rho_values.append(rho)
    c_values.append(c)
    M_values.append(M)
    Cd_values.append(Cd)
    F_drag_values.append(F_drag)

# Рассчитываем ускорение (производная скорости)
acceleration[1:] = np.diff(velocities) / np.diff(time_points)
acceleration[0] = acceleration[1]  # Первое значение

# Рассчитываем массу в каждый момент времени
mass_values = np.where(
    time_points <= ROCKET_PARAMS['burn_time'],
    ROCKET_PARAMS['m0'] - (ROCKET_PARAMS['m0'] - ROCKET_PARAMS['m_final']) / ROCKET_PARAMS['burn_time'] * time_points,
    ROCKET_PARAMS['m_final']
)

# ====================== КЛЮЧЕВЫЕ ТОЧКИ ======================
# Параметры для удобства
burn_time = ROCKET_PARAMS['burn_time']
thrust = ROCKET_PARAMS['thrust']
m0 = ROCKET_PARAMS['m0']
m_final = ROCKET_PARAMS['m_final']

# Конец активной фазы
t_active_end = burn_time
v_active_end = sol.sol(t_active_end)[1]
y_active_end = sol.sol(t_active_end)[0]

# Максимальная скорость
idx_vmax = np.argmax(velocities)
v_max = velocities[idx_vmax]
t_vmax = time_points[idx_vmax]

# Апогей
t_apogee_index = np.where(velocities < 0)[0]
if t_apogee_index.size > 0:
    idx = t_apogee_index[0]
    t_prev = time_points[idx-1]
    t_curr = time_points[idx]
    v_prev = velocities[idx-1]
    v_curr = velocities[idx]
    alpha = -v_prev / (v_curr - v_prev)
    t_apogee = t_prev + alpha * (t_curr - t_prev)
    y_max = sol.sol(t_apogee)[0]
else:
    t_apogee = time_points[-1]
    y_max = altitudes[-1]

# Максимальная сила сопротивления
idx_max_drag = np.argmax(F_drag_values)
t_max_drag = time_points[idx_max_drag]
max_drag = F_drag_values[idx_max_drag]

# Максимальное число Маха
idx_max_M = np.argmax(M_values)
t_max_M = time_points[idx_max_M]
max_M_value = M_values[idx_max_M]

# Максимальный коэффициент сопротивления
max_Cd = max(Cd_values)
t_max_Cd = time_points[np.argmax(Cd_values)]

# ====================== ВЫВОД РЕЗУЛЬТАТОВ ======================
summary = {
    "Скорость в конце активной фазы (м/с)": f"{v_active_end:.2f}",
    "Высота в конце активной фазы (м)": f"{y_active_end:.2f}",
    "Максимальная скорость (м/с)": f"{v_max:.2f}",
    "Время макс. скорости (с)": f"{t_vmax:.2f}",
    "Высота апогея (м)": f"{y_max:.2f}",
    "Время апогея (с)": f"{t_apogee:.2f}",
    "Максимальная сила сопротивления (Н)": f"{max_drag:.2f}",
    "Время макс. сопротивления (с)": f"{t_max_drag:.2f}",
    "Максимальное число Маха": f"{max_M_value:.2f}",
    "Время макс. числа Маха (с)": f"{t_max_M:.2f}",
    "Максимальный коэффициент Cd": f"{max_Cd:.4f}",
    "Время макс. Cd (с)": f"{t_max_Cd:.2f}",
    "Площадь поперечного сечения (м²)": f"{A:.6f}"
}

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ РАСЧЕТА ТРАЕКТОРИИ РАКЕТЫ")
print("="*50)
print(f"{'Параметр':<40}{'Значение':<20}{'Ед. изм.'}")
print("-"*70)
for param, value in summary.items():
    unit = value.split()[-1] if ' ' in value else ''
    val = value.split()[0] if ' ' in value else value
    print(f"{param:<40}{val:<20}{unit}")

# ====================== СОХРАНЕНИЕ В CSV ======================
# Подготовка данных для CSV
data_columns = [
    altitudes,
    velocities,
    acceleration,
    mass_values,
    F_drag_values,
    Cd_values,
    M_values,
    rho_values,
    c_values
]

column_names = [
    "Время (с)",
    "Высота (м)",
    "Скорость (м/с)",
    "Ускорение (м/с²)",
    "Масса (кг)",
    "Сила сопротивления (Н)",
    "Коэф. сопротивления (Cd)",
    "Число Маха (M)",
    "Плотность воздуха (кг/м³)",
    "Скорость звука (м/с)"
]

# Сохраняем данные в CSV
csv_filename = os.path.join(os.getcwd(), "rocket_simulation_results.csv")
save_to_csv(
    csv_filename,
    summary,
    time_points,
    data_columns,
    column_names
)

# ====================== ВИЗУАЛИЗАЦИЯ ======================
# Создаем фигуру с 3x2 графиками
fig, axs = plt.subplots(3, 2, figsize=(14, 15))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# Стиль аннотаций
bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9, lw=0.8)

# График 1: Траектория полета
ax = axs[0, 0]
ax.plot(time_points, altitudes, 'b-', linewidth=2, label='Высота')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Высота (м)', color='b')
ax.tick_params(axis='y', labelcolor='b')
ax.set_title('Высота полета')
ax.axvline(x=burn_time, color='r', linestyle='--', alpha=0.7, label='Отсечка двигателя')
ax.axvline(x=t_apogee, color='g', linestyle='--', alpha=0.7, label='Апогей')
ax.legend(loc='upper left')

# Аннотации для высоты
ax.annotate(f'Апогей: {y_max:.0f} м', xy=(t_apogee, y_max), 
            xytext=(t_apogee+2, y_max*0.9), bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax.annotate(f'Отсечка: {y_active_end:.0f} м', xy=(burn_time, y_active_end), 
            xytext=(burn_time+2, y_active_end*0.8), bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

# График 2: Скорость
ax = axs[0, 1]
ax.plot(time_points, velocities, 'r-', linewidth=2, label='Скорость')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Скорость (м/с)', color='r')
ax.tick_params(axis='y', labelcolor='r')
ax.set_title('Скорость полета')
ax.axhline(y=v_max, color='g', linestyle=':', alpha=0.5)
ax.axvline(x=t_vmax, color='g', linestyle='--', alpha=0.5, label='Макс. скорость')

# Аннотации для скорости
ax.annotate(f'Макс. скорость: {v_max:.0f} м/с', xy=(t_vmax, v_max), 
            xytext=(t_vmax+2, v_max*0.9), bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax.annotate(f'Отсечка: {v_active_end:.0f} м/с', xy=(burn_time, v_active_end), 
            xytext=(burn_time+2, v_active_end*0.8), bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax.legend(loc='upper left')

# График 3: Ускорение
ax = axs[1, 0]
ax.plot(time_points, acceleration, 'm-', linewidth=2)
ax.set_xlabel('Время (с)')
ax.set_ylabel('Ускорение (м/с²)', color='m')
ax.tick_params(axis='y', labelcolor='m')
ax.set_title('Ускорение ракеты')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=burn_time, color='r', linestyle='--', alpha=0.7)

# Аннотация для ускорения
max_accel = np.max(acceleration)
t_max_accel = time_points[np.argmax(acceleration)]
ax.annotate(f'Макс. ускорение: {max_accel:.1f} м/с²', 
            xy=(t_max_accel, max_accel), 
            xytext=(t_max_accel+2, max_accel*0.9), 
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

# График 4: Сила сопротивления
ax = axs[1, 1]
ax.plot(time_points, F_drag_values, 'c-', linewidth=2)
ax.set_xlabel('Время (с)')
ax.set_ylabel('Сила сопротивления (Н)', color='c')
ax.tick_params(axis='y', labelcolor='c')
ax.set_title('Аэродинамическое сопротивление')
ax.axvline(x=t_max_drag, color='b', linestyle='--', alpha=0.5)

# Аннотация для силы сопротивления
ax.annotate(f'Макс. сопротивление: {max_drag:.1f} Н', 
            xy=(t_max_drag, max_drag), 
            xytext=(t_max_drag+2, max_drag*0.9), 
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

# График 5: Число Маха
ax = axs[2, 0]
ax.plot(time_points, M_values, 'g-', linewidth=2)
ax.set_xlabel('Время (с)')
ax.set_ylabel('Число Маха', color='g')
ax.tick_params(axis='y', labelcolor='g')
ax.set_title('Число Маха')
ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Звуковой барьер')
ax.axvline(x=t_max_M, color='b', linestyle='--', alpha=0.5)

# Аннотация для числа Маха
ax.annotate(f'Макс. число Маха: {max_M_value:.2f}', 
            xy=(t_max_M, max_M_value), 
            xytext=(t_max_M+2, max_M_value*0.9), 
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax.legend(loc='upper left')

# График 6: Коэффициент сопротивления
ax = axs[2, 1]
ax.plot(time_points, Cd_values, 'orange', linewidth=2)
ax.set_xlabel('Время (с)')
ax.set_ylabel('Коэффициент Cd', color='orange')
ax.tick_params(axis='y', labelcolor='orange')
ax.set_title('Коэффициент аэродинамического сопротивления')
ax.axvline(x=t_max_M, color='b', linestyle='--', alpha=0.5)

# Аннотация для Cd
max_Cd = np.max(Cd_values)
t_max_Cd = time_points[np.argmax(Cd_values)]
ax.annotate(f'Макс. Cd: {max_Cd:.3f}', 
            xy=(t_max_Cd, max_Cd), 
            xytext=(t_max_Cd+2, max_Cd*0.95), 
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

# Сохраняем и показываем графики
plt.suptitle(f"Анализ полета ракеты (d={ROCKET_PARAMS['diameter']*1000:.0f} мм, "
            f"тяга {ROCKET_PARAMS['thrust']:.0f} Н)", fontsize=14)
plt.savefig('rocket_analysis.png', dpi=150, bbox_inches='tight')
plt.show()