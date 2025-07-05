import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

# Параметры задачи
params = {
    'm0': 4.0,        # начальная масса (кг)
    'm_final': 3.0,   # конечная масса (кг)
    'F': 600.0,       # сила тяги (Н)
    't_thrust': 2.97, # время работы двигателя (с)
    'k': 0.3,         # коэффициент сопротивления (кг/с)
    'g': 9.81         # ускорение свободного падения (м/с²)
}

# Время интегрирования (до достижения апогея)
t_end = 20.0  # достаточно для достижения апогея

# Решение ОДУ
sol = solve_ivp(rocket_motion, [0, t_end], [0, 0], 
                args=tuple(params.values()),
                method='RK45', rtol=1e-6, atol=1e-9,
                dense_output=True)

# Находим время и скорость в конце активной фазы
t_active_end = params['t_thrust']
v_active_end = sol.sol(t_active_end)[1]
y_active_end = sol.sol(t_active_end)[0]

# Находим апогей (момент, когда скорость становится нулевой)
t_apogee_index = np.where(sol.y[1] < 0)[0]
if t_apogee_index.size > 0:
    idx = t_apogee_index[0]
    t_prev = sol.t[idx-1]
    t_curr = sol.t[idx]
    v_prev = sol.y[1, idx-1]
    v_curr = sol.y[1, idx]
    
    # Линейная интерполяция для точного определения апогея
    alpha = -v_prev / (v_curr - v_prev)
    t_apogee = t_prev + alpha * (t_curr - t_prev)
    y_max = sol.sol(t_apogee)[0]
else:
    t_apogee = sol.t[-1]
    y_max = sol.y[0, -1]

# Максимальная скорость
v_max = np.max(sol.y[1])

# Вывод результатов
print("\nТочные результаты расчета:")
print(f"Скорость в конце активной фазы (t={params['t_thrust']} с): {v_active_end:.2f} м/с")
print(f"Высота в конце активной фазы: {y_active_end:.2f} м")
print(f"Максимальная скорость: {v_max:.2f} м/с")
print(f"Высота апогея: {y_max:.2f} м")
print(f"Общее время подъёма: {t_apogee:.2f} с")

# Построение графиков
plt.figure(figsize=(12, 8))

# График скорости
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[1], 'b-')
plt.axvline(x=params['t_thrust'], color='r', linestyle='--', label='Конец работы двигателя')
plt.axvline(x=t_apogee, color='g', linestyle='--', label='Апогей')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.title('Зависимость скорости от времени')
plt.grid(True)
plt.legend()

# График высоты
plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[0], 'g-')
plt.axvline(x=params['t_thrust'], color='r', linestyle='--')
plt.axvline(x=t_apogee, color='g', linestyle='--')
plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')
plt.title('Зависимость высоты от времени')
plt.grid(True)

plt.tight_layout()
plt.savefig('rocket_motion.png')
plt.show()