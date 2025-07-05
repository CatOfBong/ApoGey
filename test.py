import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def rocket_motion(t, state, m0, m_final, F, t_thrust, k, g):
    y, v = state
    r = (m0 - m_final) / t_thrust
    
    # Активная фаза (работа двигателя)
    if t <= t_thrust:
        m = m0 - r * t
        dvdt = F/m - g - k*v/m
    # Пассивная фаза (двигатель выключен)
    else:
        m = m_final
        dvdt = -g - k*v/m
    
    dydt = v
    return [dydt, dvdt]

# Исходные параметры
params = {
    'm0': 4.0,        # начальная масса (кг)
    'm_final': 3.0,   # конечная масса (кг)
    'F': 600.0,       # сила тяги (Н)
    't_thrust': 2.97, # время работы двигателя (с)
    'k': 0.1,         # коэффициент сопротивления (кг/с)
    'g': 9.81         # ускорение свободного падения (м/с²)
}

# Время интегрирования (до достижения апогея)
t_end = 35.0  # достаточно для достижения апогея

# Решение ОДУ
sol = solve_ivp(rocket_motion, [0, t_end], [0, 0], 
                args=tuple(params.values()),
                method='RK45', rtol=1e-12, atol=1e-15,
                dense_output=True)

# Находим время и скорость в конце активной фазы
t_active_end = params['t_thrust']
v_active_end = sol.sol(t_active_end)[1]
y_active_end = sol.sol(t_active_end)[0]

# Находим апогей
t_apogee_index = np.where(sol.y[1] < 0)[0]
if t_apogee_index.size > 0:
    idx = t_apogee_index[0]
    t_prev = sol.t[idx-1]
    t_curr = sol.t[idx]
    v_prev = sol.y[1, idx-1]
    v_curr = sol.y[1, idx]
    
    # Линейная интерполяция для относительно точного определения апогея
    alpha = -v_prev / (v_curr - v_prev)
    t_apogee = t_prev + alpha * (t_curr - t_prev)
    y_max = sol.sol(t_apogee)[0]
else:
    t_apogee = sol.t[-1]
    y_max = sol.y[0, -1]

# Находим максимальную скорость и время ее достижения
idx_vmax = np.argmax(sol.y[1])
v_max = sol.y[1, idx_vmax]
t_vmax = sol.t[idx_vmax]

# Проверка, нужно ли показывать отдельно максимальную скорость
show_max_speed = abs(v_max - v_active_end) > 1.0

# Вывод результатов
print("\nТочные результаты расчета:")
print(f"Скорость в конце активной фазы (t={params['t_thrust']} с): {v_active_end:.2f} м/с")
print(f"Высота в конце активной фазы: {y_active_end:.2f} м")
print(f"Максимальная скорость: {v_max:.2f} м/с (t={t_vmax:.2f} с)")
print(f"Высота апогея: {y_max:.2f} м (t={t_apogee:.2f} с)")
print(f"Общее время подъёма: {t_apogee:.2f} с")

# Построение графиков
plt.figure(figsize=(12, 10))

# Общие настройки стиля для подписей
label_style = {'bbox': dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.9, lw=0.5),
               'fontsize': 9, 'ha': 'right', 'va': 'center'}

# График скорости
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[1], 'b-', label='Скорость ракеты')
plt.axvline(x=params['t_thrust'], color='r', linestyle='--', alpha=0.7, label='Отсечка двигателя')
plt.axvline(x=t_apogee, color='g', linestyle='--', alpha=0.7, label='Апогей')

# Ключевые линии и аннотации скорости
plt.axhline(y=v_active_end, color='r', linestyle=':', alpha=0.3)
plt.axhline(y=v_max, color='g', linestyle=':', alpha=0.3)

# Маркеры для ключевых точек
plt.plot(t_active_end, v_active_end, 'ro', markersize=6)
plt.plot(t_vmax, v_max, 'gD', markersize=6) if show_max_speed else None

# Подписи значений слева с фоном
x_pos = -0.02 * max(sol.t)  # Позиция слева от оси
plt.text(x_pos, v_active_end, f' Отсечка: {v_active_end:.2f} м/с ', 
         color='r', **label_style)

if show_max_speed:
    plt.text(x_pos, v_max, f' Макс.скорость: {v_max:.2f} м/с ', 
             color='g', **label_style)
elif abs(v_max - v_active_end) > 0.01:  # Если почти равны, но не совсем
    plt.text(x_pos, v_active_end, f' Отсечка/макс: {v_active_end:.2f} м/с ', 
             color='purple', **label_style)

plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.title('Зависимость скорости от времени')
plt.grid(True, linestyle='--', alpha=0.7)

# Легенда без дублирования информации
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, loc='upper right')

# График высоты
plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[0], 'g-', label='Высота ракеты')
plt.axvline(x=params['t_thrust'], color='r', linestyle='--', alpha=0.7)
plt.axvline(x=t_apogee, color='g', linestyle='--', alpha=0.7)

# Ключевые линии и аннотации высоты
plt.axhline(y=y_active_end, color='r', linestyle=':', alpha=0.3)
plt.axhline(y=y_max, color='g', linestyle=':', alpha=0.3)

# Маркеры для ключевых точек
plt.plot(t_active_end, y_active_end, 'ro', markersize=6)
plt.plot(t_apogee, y_max, 'go', markersize=6)

# Подписи значений слева с фоном
plt.text(x_pos, y_active_end, f' Отсечка: {y_active_end:.2f} м ', 
         color='r', **label_style)
plt.text(x_pos, y_max, f' Апогей: {y_max:.2f} м ', 
         color='g', **label_style)

plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')
plt.title('Зависимость высоты от времени')
plt.grid(True, linestyle='--', alpha=0.7)

# Установим одинаковые отступы слева для обоих графиков
plt.subplots_adjust(left=0.12)

plt.tight_layout()
plt.savefig('out.png', dpi=300)
plt.show()