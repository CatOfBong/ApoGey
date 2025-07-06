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
# Константы стандартной атмосферы (обновленная модель до 100 км)
R_earth = 6371000.0  # Радиус Земли (м)

# Слои атмосферы: (h_min, h_max, L, T_base, p_base)
ATMOSPHERE_LAYERS = [
    (0, 11000, -0.0065, 288.15, 101325.0),    # Тропосфера
    (11000, 20000, 0, 216.65, 22632.040),      # Стратосфера (нижняя)
    (20000, 32000, 0.001, 216.65, 5474.88),    # Стратосфера (средняя)
    (32000, 47000, 0.0028, 228.65, 868.02),    # Стратосфера (верхняя)
    (47000, 100000, 0, 270.65, 110.91)         # Мезосфера
]

# Параметры ракеты
ROCKET_PARAMS = {
    'diameter': 0.06,       # Диаметр ракеты (м)
    'm0': 4.0,              # Стартовая масса (кг)
    'm_final': 3.0,         # Масса после выгорания топлива (кг)
    'thrust': 600.0,        # Тяга (Н)
    'burn_time': 1.97,      # Время работы двигателя (с)
    'g0': 9.80665           # Гравитация на уровне моря (м/с²)
}

# Рассчитываем площадь поперечного сечения
A = np.pi * (ROCKET_PARAMS['diameter']/2)**2

# Табличные данные коэффициента сопротивления (реалистичные)
MACH_POINTS = [0, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 2.0, 3.0, 4.0]
CD_POINTS = [0.15, 0.15, 0.25, 0.8, 1.1, 0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.15]

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def gravity(y):
    """Учет изменения гравитации с высотой"""
    return ROCKET_PARAMS['g0'] * (R_earth / (R_earth + y))**2

def atmosphere(y):
    """Рассчитывает параметры атмосферы на заданной высоте (0-100 км)"""
    # Используем g0 из параметров ракеты
    g0 = ROCKET_PARAMS['g0']
    
    # Для отрицательных высот используем параметры уровня моря
    if y < 0:
        p = ATMOSPHERE_LAYERS[0][4]
        T = ATMOSPHERE_LAYERS[0][3]
        rho = p / (R * T)
        c = np.sqrt(gamma * R * T)
        return rho, c
    
    # Для высот выше 100 км - экстраполяция
    if y > 100000:
        T = 270.65  # Температура мезопаузы
        p = 110.91 * np.exp(-g0 * (y - 47000) / (R * T))
        rho = p / (R * T)
        c = np.sqrt(gamma * R * T)
        return rho, c
    
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
            return rho, c
    
    # Если высота не попала ни в один слой (должно быть исключено предыдущими проверками)
    return 0.0, 300.0

def calculate_Cd(M):
    """Интерполяция коэффициента сопротивления по табличным данным"""
    M_abs = abs(M)
    return np.interp(M_abs, MACH_POINTS, CD_POINTS, right=CD_POINTS[-1])

def rocket_motion(t, state, *args):
    """Дифференциальные уравнения движения ракеты"""
    m0, m_final, thrust, burn_time = args
    y, v = state
    mass_flow = (m0 - m_final) / burn_time
    
    # Рассчитываем текущую гравитацию
    g_current = gravity(y)
    
    # Атмосферные параметры (кэшируем для текущей высоты)
    rho, c = atmosphere_cache.get(y, (None, None))
    if rho is None:
        rho, c = atmosphere(max(y, 0))
        atmosphere_cache[y] = (rho, c)
    
    # Рассчитываем число Маха и коэффициент сопротивления
    M = abs(v) / c if c > 0 else 0
    Cd = calculate_Cd(M)
    
    # Сила аэродинамического сопротивления
    F_drag = 0.5 * rho * v**2 * Cd * A
    drag_accel = -np.copysign(F_drag, v)  # Направление против скорости
    
    # Рассчитываем массу и тягу в текущий момент времени
    if t <= burn_time:
        m = m0 - mass_flow * t
        thrust_current = thrust
    else:
        m = m_final
        thrust_current = 0.0  # Двигатель выключен
    
    # Уравнение движения: dv/dt = (тяга + сопротивление)/масса - гравитация
    dvdt = (thrust_current + drag_accel) / m - g_current
    
    return [v, dvdt]

def apogee_event(t, state, *args):
    """Событие для точного определения апогея (v=0)"""
    return state[1]  # Вертикальная скорость
apogee_event.terminal = True
apogee_event.direction = -1

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

# ====================== ФИЗИЧЕСКИЕ КОНСТАНТЫ ======================
gamma = 1.4      # Показатель адиабаты для воздуха
R = 287.058      # Газовая постоянная (Дж/(кг·K))

# ====================== ПОДГОТОВКА ДАННЫХ ======================
# Кэш для ускорения расчетов атмосферы
atmosphere_cache = {}

# Валидация параметров
assert ROCKET_PARAMS['m0'] > ROCKET_PARAMS['m_final'], "Стартовая масса должна быть больше конечной"
assert ROCKET_PARAMS['thrust'] > 0, "Тяга должна быть положительной"
assert ROCKET_PARAMS['burn_time'] > 0, "Время работы двигателя должно быть положительным"

# Параметры для решателя
t_end = 60.0  # Увеличили время для достижения апогея
args = (ROCKET_PARAMS['m0'], ROCKET_PARAMS['m_final'], 
        ROCKET_PARAMS['thrust'], ROCKET_PARAMS['burn_time'])

# Решение дифференциальных уравнений с обработкой события
sol = solve_ivp(rocket_motion, [0, t_end], [0, 0], 
                args=args,
                events=apogee_event,
                method='DOP853',  # Высокоточный метод
                rtol=1e-10,       # Очень малая относительная погрешность
                atol=1e-12,       # Очень малая абсолютная погрешность
                dense_output=True,
                max_step=0.05)    # Частый вывод для точности

# Основные результаты
time_points = sol.t
altitudes = sol.y[0]
velocities = sol.y[1]

# Обработка события апогея
if sol.t_events[0].size > 0:
    t_apogee = sol.t_events[0][0]
    y_max = sol.sol(t_apogee)[0]
else:
    t_apogee = time_points[-1]
    y_max = altitudes[-1]

# Рассчитываем дополнительные параметры
rho_values = []
c_values = []
M_values = []
Cd_values = []
F_drag_values = []
g_values = []
acceleration = []

for i, t in enumerate(time_points):
    # Рассчитываем атмосферные параметры
    rho, c = atmosphere(max(altitudes[i], 0))
    M = abs(velocities[i]) / c if c > 0 else 0
    Cd = calculate_Cd(M)
    F_drag = 0.5 * rho * velocities[i]**2 * Cd * A
    
    rho_values.append(rho)
    c_values.append(c)
    M_values.append(M)
    Cd_values.append(Cd)
    F_drag_values.append(F_drag)
    g_values.append(gravity(altitudes[i]))
    
    # Рассчитываем ускорение напрямую из функции движения
    state = [altitudes[i], velocities[i]]
    _, dvdt = rocket_motion(t, state, *args)
    acceleration.append(dvdt)

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
t_active_end = min(burn_time, max(time_points))
v_active_end = sol.sol(t_active_end)[1] if t_active_end <= max(time_points) else 0
y_active_end = sol.sol(t_active_end)[0] if t_active_end <= max(time_points) else 0

# Максимальная скорость
idx_vmax = np.argmax(velocities)
v_max = velocities[idx_vmax]
t_vmax = time_points[idx_vmax]

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

# Максимальное ускорение
max_accel = max(acceleration)
t_max_accel = time_points[np.argmax(acceleration)]

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
    "Максимальное ускорение (м/с²)": f"{max_accel:.2f}",
    "Время макс. ускорения (с)": f"{t_max_accel:.2f}",
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
    c_values,
    g_values
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
    "Скорость звука (м/с)",
    "Гравитация (м/с²)"
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
# Создаем фигуру с 4x2 графиками
fig = plt.figure(figsize=(16, 22))  # Увеличена высота для лучшего расположения
gs = mpl.gridspec.GridSpec(4, 2, height_ratios=[1.2, 1, 1, 0.7])  # Увеличена первая строка
axs = [
    plt.subplot(gs[0, 0]),
    plt.subplot(gs[0, 1]),
    plt.subplot(gs[1, 0]),
    plt.subplot(gs[1, 1]),
    plt.subplot(gs[2, 0]),
    plt.subplot(gs[2, 1]),
    plt.subplot(gs[3, :])  # Широкий график для Cd
]
plt.subplots_adjust(hspace=0.6, wspace=0.3)  # Увеличены отступы

# Стиль аннотаций
bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9, lw=0.8)

# График 1: Траектория полета
ax = axs[0]
ax.plot(time_points, altitudes, 'b-', linewidth=2, label='Высота')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Высота (м)', color='b')
ax.tick_params(axis='y', labelcolor='b')
ax.set_title('Высота полета')
ax.axvline(x=burn_time, color='r', linestyle='--', alpha=0.7, label='Отсечка двигателя')
ax.axvline(x=t_apogee, color='g', linestyle='--', alpha=0.7, label='Апогей')
ax.legend(loc='upper left')  # Перемещено в левый верхний угол

# Аннотации для высоты
ax.annotate(f'Апогей: {y_max:.0f} м', xy=(t_apogee, y_max), 
            xytext=(t_apogee+5, y_max*0.8),  # Увеличен сдвиг по X
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax.annotate(f'Отсечка: {y_active_end:.0f} м', xy=(burn_time, y_active_end), 
            xytext=(burn_time+5, y_active_end*0.6),  # Увеличен сдвиг по X и Y
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

# График 2: Скорость
ax = axs[1]
ax.plot(time_points, velocities, 'r-', linewidth=2, label='Скорость')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Скорость (м/с)', color='r')
ax.tick_params(axis='y', labelcolor='r')
ax.set_title('Скорость полета')
ax.axhline(y=v_max, color='g', linestyle=':', alpha=0.5)
ax.axvline(x=t_vmax, color='g', linestyle='--', alpha=0.5, label='Макс. скорость')
ax.legend(loc='upper left')  # Перемещено в левый верхний угол

# Аннотации для скорости
ax.annotate(f'Отсечка: {v_active_end:.0f} м/с', xy=(burn_time, v_active_end), 
            xytext=(burn_time+5, v_active_end*0.7),  # Увеличен сдвиг
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

# График 3: Ускорение
ax = axs[2]
ax.plot(time_points, acceleration, 'm-', linewidth=2)
ax.set_xlabel('Время (с)')
ax.set_ylabel('Ускорение (м/с²)', color='m')
ax.tick_params(axis='y', labelcolor='m')
ax.set_title('Ускорение ракеты')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=burn_time, color='r', linestyle='--', alpha=0.7)

# Аннотация для ускорения
ax.annotate(f'Макс.: {max_accel:.1f} м/с²',  # Сокращенный текст 
            xy=(t_max_accel, max_accel), 
            xytext=(t_max_accel+3, max_accel*0.8),  # Увеличен сдвиг
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            fontsize=9)  # Уменьшен размер шрифта

# График 4: Сила сопротивления
ax = axs[3]
ax.plot(time_points, F_drag_values, 'c-', linewidth=2)
ax.set_xlabel('Время (с)')
ax.set_ylabel('Сила сопротивления (Н)', color='c')
ax.tick_params(axis='y', labelcolor='c')
ax.set_title('Аэродинамическое сопротивление')
ax.axvline(x=t_max_drag, color='b', linestyle='--', alpha=0.5)

# Аннотация для силы сопротивления
ax.annotate(f'Макс.: {max_drag:.1f} Н',  # Сокращенный текст
            xy=(t_max_drag, max_drag), 
            xytext=(t_max_drag+3, max_drag*0.7),  # Увеличен сдвиг
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

# График 5: Число Маха
ax = axs[4]
ax.plot(time_points, M_values, 'g-', linewidth=2)
ax.set_xlabel('Время (с)')
ax.set_ylabel('Число Маха', color='g')
ax.tick_params(axis='y', labelcolor='g')
ax.set_title('Число Маха')
ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Звуковой барьер')
ax.axvline(x=t_max_M, color='b', linestyle='--', alpha=0.5)
ax.legend(loc='upper left')  # Перемещено в левый верхний угол

# Аннотация для числа Маха
ax.annotate(f'Макс.: {max_M_value:.2f}',  # Сокращенный текст
            xy=(t_max_M, max_M_value), 
            xytext=(t_max_M+3, max_M_value*0.8),  # Увеличен сдвиг
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            fontsize=9)  # Уменьшен размер шрифта

# График 6: Гравитация
ax = axs[5]
ax.plot(time_points, g_values, 'purple', linewidth=2)
ax.set_xlabel('Время (с)')
ax.set_ylabel('Гравитация (м/с²)', color='purple')
ax.tick_params(axis='y', labelcolor='purple')
ax.set_title('Изменение гравитации с высотой')
ax.axhline(y=ROCKET_PARAMS['g0'], color='gray', linestyle='--', alpha=0.5, label='g₀ на уровне моря')
ax.legend(loc='upper right')  # Перемещено в правый верхний угол

# График 7: Зависимость Cd от числа Маха (специальный график)
ax = axs[6]
M_plot = np.linspace(0, 4.0, 200)
Cd_plot = [calculate_Cd(M) for M in M_plot]
ax.plot(M_plot, Cd_plot, 'r-', linewidth=2.5)
ax.set_xlabel('Число Маха')
ax.set_ylabel('Cd')
ax.set_title('Зависимость Cd от числа Маха')
ax.grid(True, alpha=0.3)
ax.axvline(x=1.0, color='b', linestyle='--', alpha=0.7)
ax.text(0.5, 0.85, 'Трансзвуковая зона', transform=ax.transAxes, fontsize=10,  # Уменьшен размер
        bbox=dict(facecolor='white', alpha=0.8))
ax.annotate('Пик Cd при M=1.0', xy=(1.0, 1.1),  # Сокращенный текст
            xytext=(1.5, 0.8), 
            arrowprops=dict(arrowstyle="->"),
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            fontsize=9)  # Уменьшен размер

# Отмечаем ключевые точки
for M, Cd in zip(MACH_POINTS, CD_POINTS):
    ax.plot(M, Cd, 'bo', markersize=6, alpha=0.7)

# Автоматическая оптимизация расположения
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Оставляет место для заголовка

# Сохраняем и показываем графики
plt.suptitle(f"Анализ полета ракеты (d={ROCKET_PARAMS['diameter']*1000:.0f} мм, "
            f"тяга {ROCKET_PARAMS['thrust']:.0f} Н, "
            f"апогей {y_max:.0f} м)", fontsize=16)
plt.savefig('rocket_analysis.png', dpi=150, bbox_inches='tight')
plt.show()