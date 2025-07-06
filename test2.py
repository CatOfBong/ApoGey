import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Константы стандартной атмосферы
T0 = 288.15
p0 = 101325.0
L = 0.0065
g0 = 9.80665
R = 287.058
gamma = 1.4
T11 = 216.65
p11 = p0 * (T11/T0) ** (g0/(R*L))

# Площадь поперечного сечения (м²)
A = np.pi*0.03**2  # Для диаметра 50 мм

def atmosphere(y):
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
    """Усовершенствованная функция с подробным выводом для отладки"""
    if M < 0.8:
        return 0.1
    elif M < 1.0:
        return 0.1 + 0.9 * (M - 0.8) / 0.2
    elif M < 1.2:
        return 1.0 - 0.8 * (M - 1.0) / 0.2
    else:
        return max(0.1, 0.4 / (M**0.6))

def rocket_motion(t, state, m0, m_final, F, t_thrust, g):
    y, v = state
    r = (m0 - m_final) / t_thrust
    
    rho, c = atmosphere(max(y, 0))
    M = abs(v) / c if c > 0 else 0
    Cd = calculate_Cd(M)
    
    F_drag = 0.5 * rho * v**2 * Cd * A
    drag_accel = -np.copysign(F_drag, v)
    
    if t <= t_thrust:
        m = m0 - r * t
        dvdt = F/m - g + drag_accel/m
    else:
        m = m_final
        dvdt = -g + drag_accel/m
    
    return [v, dvdt]

# Параметры ракеты
params = {
    'm0': 4.0,
    'm_final': 3.0,
    'F': 600.0,
    't_thrust': 1.97,
    'g': 9.81
}

t_end = 30.0
sol = solve_ivp(rocket_motion, [0, t_end], [0, 0], 
                args=tuple(params.values()),
                method='RK45', rtol=1e-8, atol=1e-10,
                dense_output=True, max_step=0.1)

# Расчёт дополнительных параметров
time_points = sol.t
altitudes = sol.y[0]
velocities = sol.y[1]

F_drag_values = []
Cd_values = []
M_values = []
rho_values = []

for i in range(len(time_points)):
    rho, c = atmosphere(max(altitudes[i], 0))
    M = abs(velocities[i]) / c if c > 0 else 0
    Cd = calculate_Cd(M)
    F_drag = 0.5 * rho * velocities[i]**2 * Cd * A
    
    F_drag_values.append(F_drag)
    Cd_values.append(Cd)
    M_values.append(M)
    rho_values.append(rho)

# Находим ключевые точки
t_active_end = params['t_thrust']
v_active_end = sol.sol(t_active_end)[1]
y_active_end = sol.sol(t_active_end)[0]

# Находим апогей
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

# Находим максимальную скорость
idx_vmax = np.argmax(velocities)
v_max = velocities[idx_vmax]
t_vmax = time_points[idx_vmax]

# Находим пиковые аэродинамические параметры
idx_max_drag = np.argmax(F_drag_values)
t_max_drag = time_points[idx_max_drag]
max_drag = F_drag_values[idx_max_drag]
max_M = M_values[idx_max_drag]
max_Cd = Cd_values[idx_max_drag]

idx_max_M = np.argmax(M_values)
t_max_M = time_points[idx_max_M]
max_M_value = M_values[idx_max_M]
Cd_at_max_M = Cd_values[idx_max_M]

# Вывод полной диагностики
print("\nТочные результаты расчета:")
print(f"Скорость в конце активной фазы (t={params['t_thrust']} с): {v_active_end:.2f} м/с")
print(f"Высота в конце активной фазы: {y_active_end:.2f} м")
print(f"Максимальная скорость: {v_max:.2f} м/с (t={t_vmax:.2f} с)")
print(f"Высота апогея: {y_max:.2f} м (t={t_apogee:.2f} с)")
print(f"Общее время подъёма: {t_apogee:.2f} с")
print("\nАэродинамические параметры:")
print(f"Максимальная сила сопротивления: {max_drag:.2f} Н (t={t_max_drag:.2f} с)")
print(f"Максимальное число Маха: {max_M_value:.3f} (t={t_max_M:.2f} с)")
print(f"Коэффициент Cd при max(M): {Cd_at_max_M:.4f}")
print(f"Максимальный коэффициент Cd: {max(Cd_values):.4f} (t={time_points[np.argmax(Cd_values)]:.2f} с)")
print(f"Плотность воздуха при max(F_drag): {rho_values[idx_max_drag]:.5f} кг/м³")
print("\n", A)

# Построение детальных графиков для отладки
plt.figure(figsize=(12, 16), dpi=100)

# График 1: Траектория и скорость
plt.subplot(4, 1, 1)
plt.plot(time_points, velocities, 'b-', label='Скорость')
plt.plot(time_points, altitudes, 'g-', label='Высота')
plt.axvline(x=params['t_thrust'], color='r', linestyle='--', label='Отсечка двигателя')
plt.axvline(x=t_apogee, color='purple', linestyle='--', label='Апогей')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с) / Высота (м)')
plt.title('Основные параметры полета')
plt.grid(True)
plt.legend(loc='upper left')
plt.twinx()
plt.plot(time_points, M_values, 'm:', label='Число Маха', alpha=0.7)
plt.ylabel('Число Маха')

# График 2: Аэродинамические силы
plt.subplot(4, 1, 2)
plt.plot(time_points, F_drag_values, 'r-', label='Сила сопротивления')
plt.axhline(y=max_drag, color='r', linestyle=':', alpha=0.5)
plt.plot(t_max_drag, max_drag, 'ro', markersize=8)
plt.text(t_max_drag, max_drag, f'  Max: {max_drag:.1f} Н', 
         bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='red', alpha=0.9))

# Рассчитаем силу тяги для графика
thrust = [params['F'] if t <= params['t_thrust'] else 0 for t in time_points]
plt.plot(time_points, thrust, 'b--', label='Сила тяги')

plt.xlabel('Время (с)')
plt.ylabel('Сила (Н)')
plt.title('Аэродинамические силы')
plt.grid(True)
plt.legend()

# График 3: Коэффициенты и плотность
plt.subplot(4, 1, 3)
plt.plot(time_points, Cd_values, 'b-', label='Коэффициент Cd')
plt.plot(time_points, np.array(rho_values)*1000, 'g--', label='Плотность воздуха (г/м³)')
plt.xlabel('Время (с)')
plt.ylabel('Cd / Плотность')
plt.title('Аэродинамические коэффициенты и плотность воздуха')
plt.grid(True)
plt.legend()

# График 4: Ускорения
acceleration = np.gradient(velocities, time_points)
plt.subplot(4, 1, 4)
plt.plot(time_points, acceleration, 'm-', label='Ускорение ракеты')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=params['t_thrust'], color='r', linestyle='--', alpha=0.7)

# Рассчитаем составляющие ускорения
thrust_accel = [params['F']/(params['m0'] - (params['m0']-params['m_final'])/params['t_thrust']*t) 
                if t <= params['t_thrust'] else 0 for t in time_points]
drag_accel = [-np.copysign(F_drag, v) / 
             (params['m0'] if t <= params['t_thrust'] else params['m_final']) 
             for t, v, F_drag in zip(time_points, velocities, F_drag_values)]
net_accel = [t_accel + d_accel - params['g'] for t_accel, d_accel in zip(thrust_accel, drag_accel)]

plt.plot(time_points, thrust_accel, 'b:', label='Ускорение от тяги')
plt.plot(time_points, net_accel, 'g--', label='Чистое ускорение')
plt.plot(time_points, drag_accel, 'r-.', label='Ускорение от сопротивления')

plt.xlabel('Время (с)')
plt.ylabel('Ускорение (м/с²)')
plt.title('Составляющие ускорения')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('rocket_full_analysis_debug.png', dpi=300)
plt.close()  # Закрываем фигуру, но не показываем

# Построение чистых графиков для результатов
plt.figure(figsize=(12, 10), dpi=120)

# Общие настройки стиля для подписей
label_style = {'bbox': dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.9, lw=0.5),
               'fontsize': 9, 'ha': 'right', 'va': 'center'}

# График скорости
plt.subplot(2, 1, 1)
plt.plot(time_points, velocities, 'b-', label='Скорость ракеты', linewidth=1.5)
plt.axvline(x=params['t_thrust'], color='r', linestyle='--', alpha=0.7, label='Отсечка двигателя')
plt.axvline(x=t_apogee, color='g', linestyle='--', alpha=0.7, label='Апогей')

# Ключевые линии и аннотации скорости
plt.axhline(y=v_active_end, color='r', linestyle=':', alpha=0.3)
plt.axhline(y=v_max, color='g', linestyle=':', alpha=0.3)

# Маркеры для ключевых точек
plt.plot(t_active_end, v_active_end, 'ro', markersize=6)
plt.plot(t_vmax, v_max, 'gD', markersize=6)

# Подписи значений слева с фоном
x_pos = -0.02 * max(time_points)  # Позиция слева от оси
plt.text(x_pos, v_active_end, f' Отсечка: {v_active_end:.2f} м/с', 
         color='r', **label_style)

if abs(v_max - v_active_end) > 1:  # Если разница существенная
    plt.text(x_pos, v_max, f' Макс.скорость: {v_max:.2f} м/с', 
             color='g', **label_style)
else:
    plt.text(x_pos, v_active_end, f' Отсечка/макс: {v_active_end:.2f} м/с', 
             color='purple', **label_style)

plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.title('Зависимость скорости от времени', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

# График высоты
plt.subplot(2, 1, 2)
plt.plot(time_points, altitudes, 'g-', label='Высота ракеты', linewidth=1.5)
plt.axvline(x=params['t_thrust'], color='r', linestyle='--', alpha=0.7)
plt.axvline(x=t_apogee, color='g', linestyle='--', alpha=0.7)

# Ключевые линии и аннотации высоты
plt.axhline(y=y_active_end, color='r', linestyle=':', alpha=0.3)
plt.axhline(y=y_max, color='g', linestyle=':', alpha=0.3)

# Маркеры для ключевых точек
plt.plot(t_active_end, y_active_end, 'ro', markersize=6)
plt.plot(t_apogee, y_max, 'go', markersize=6)

# Подписи значений слева с фоном
plt.text(x_pos, y_active_end, f' Отсечка: {y_active_end:.2f} м', 
         color='r', **label_style)
plt.text(x_pos, y_max, f' Апогей: {y_max:.2f} м', 
         color='g', **label_style)

# Добавим информацию о числе Маха справа
plt.twinx()
plt.plot(time_points, M_values, 'm:', label='Число Маха', alpha=0.7)
plt.ylabel('Число Маха', color='m')
plt.ylim(0, max(M_values)*1.1)

plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')
plt.title('Зависимость высоты от времени', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Установим отступы
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.08, hspace=0.3)

plt.tight_layout()
plt.savefig('rocket_results.png', dpi=300)
plt.show()