import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import griddata
import matplotlib.patches as patches
from matplotlib.ticker import ScalarFormatter, EngFormatter
from matplotlib.colors import TwoSlopeNorm, Normalize

# ====================== НАСТРОЙКА ГРАФИКОВ ======================
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['figure.figsize'] = [15, 10]

# ====================== ФИЗИЧЕСКИЕ КОНСТАНТЫ ======================
R = 287.058
gamma = 1.4
R_earth = 6371000.0
g0 = 9.80665
KGF_TO_NEWTON = 9.80665

# ====================== МОДЕЛЬ АТМОСФЕРЫ ======================
ATMOSPHERE_LAYERS = [
    (0, 11000, -0.0065, 288.15, 101325.0),
    (11000, 20000, 0, 216.65, 22632.040),
    (20000, 32000, 0.001, 216.65, 5474.88),
    (32000, 47000, 0.0028, 228.65, 868.02),
    (47000, 100000, 0, 270.65, 110.91)
]

def gravity(y):
    return g0 * (R_earth / (R_earth + y))**2

def atmosphere(y):
    if y < 0:
        p = ATMOSPHERE_LAYERS[0][4]
        T = ATMOSPHERE_LAYERS[0][3]
        rho = p / (R * T)
        c = np.sqrt(gamma * R * T)
        return rho, c, T, p
    
    if y > 100000:
        T = 270.65
        p = 110.91 * np.exp(-g0 * (y - 47000) / (R * T))
        rho = p / (R * T)
        c = np.sqrt(gamma * R * T)
        return rho, c, T, p
    
    for layer in ATMOSPHERE_LAYERS:
        h_min, h_max, L, T_base, p_base = layer
        if h_min <= y <= h_max:
            dy = y - h_min
            if L == 0:
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
    T0 = 273.15
    mu0 = 1.716e-5
    S = 110.4
    mu = mu0 * (T / T0)**1.5 * (T0 + S) / (T + S)
    return mu

def calculate_reynolds(rho, V, L, mu):
    return rho * V * L / mu

# ====================== МОДЕЛЬ ТРДД ======================
def calculate_turbojet_thrust(altitude, velocity, thrust_sl_kgf=450):
    """Расчет тяги ТРДД-50АТ с учетом высоты и скорости"""
    rho, c, T, p = atmosphere(altitude)
    rho0 = 1.225  # плотность на уровне моря
    mach = velocity / c if c > 0 else 0
    
    # Коррекция тяги по плотности
    density_correction = (rho / rho0)**0.7
    
    # Коррекция по числу Маха
    if mach < 0.8:
        mach_correction = 1.0 - 0.3 * mach
    else:
        mach_correction = 0.76
    
    thrust_kgf = thrust_sl_kgf * density_correction * mach_correction
    return thrust_kgf

def calculate_fuel_flow(thrust_kgf, tsfc=0.71):
    """Расчет расхода топлива (кг/ч)"""
    return thrust_kgf * tsfc

# ====================== АЭРОДИНАМИЧЕСКИЕ РАСЧЕТЫ ======================
def calculate_lift_coefficient(AR, sweep_angle=0, mach=0, reynolds=1e6, alpha=0):
    alpha_rad = np.radians(alpha)
    CL_alpha = 2 * np.pi * AR / (2 + np.sqrt(4 + (AR / 0.95)**2 * (1 + np.tan(np.radians(sweep_angle))**2)))
    
    beta = np.sqrt(1 - mach**2) if mach < 1 else np.sqrt(mach**2 - 1)
    CL_alpha_mach = CL_alpha / beta if mach < 1 else CL_alpha / beta
    
    CL = CL_alpha_mach * alpha_rad
    
    Re_correction = 1.0 - 0.1 * np.log10(reynolds / 1e6)
    CL *= Re_correction
    
    return CL

def calculate_drag_coefficient(CD0, AR, CL, e=0.85, mach=0, sweep_angle=0):
    CDi = CL**2 / (np.pi * AR * e)
    
    CD_wave = 0
    if mach > 0.8:
        CD_wave = 0.02 * (mach - 0.8)**2 * (1 + 0.1 * np.tan(np.radians(sweep_angle)))
    
    return CD0 + CDi + CD_wave

def calculate_lift_force(rho, V, S, CL):
    return 0.5 * rho * V**2 * S * CL

def calculate_drag_force(rho, V, S, CD):
    return 0.5 * rho * V**2 * S * CD

def kgf_to_newton(kgf):
    return kgf * KGF_TO_NEWTON

def newton_to_kgf(newton):
    return newton / KGF_TO_NEWTON

def calculate_thrust_to_weight_ratio(thrust_N, mass, altitude):
    g = gravity(altitude)
    weight = mass * g
    return thrust_N / weight

# ====================== ФУНКЦИЯ ВЫБОРА ОПТИМАЛЬНОЙ ВЫСОТЫ ======================
def find_optimal_altitude(results):
    if not results:
        return 100
    
    df = pd.DataFrame(results)
    df_flyable = df[df['Возможность_полета'] == True]
    
    if df_flyable.empty:
        optimal_alt = df.loc[df['Требуемая_тяга_кгс'].idxmin(), 'Высота_м']
        print(f"Нет возможных режимов полета. Используется высота с минимальной тягой: {optimal_alt:.0f} м")
        return optimal_alt
    
    flyable_by_altitude = df_flyable.groupby('Высота_м').size()
    if not flyable_by_altitude.empty:
        max_flyable_alt = flyable_by_altitude.idxmax()
        max_flyable_count = flyable_by_altitude.max()
        print(f"Высота с максимальным количеством режимов ({max_flyable_count}): {max_flyable_alt:.0f} м")
    
    avg_thrust_by_altitude = df_flyable.groupby('Высота_м')['Требуемая_тяга_кгс'].mean()
    if not avg_thrust_by_altitude.empty:
        min_thrust_alt = avg_thrust_by_altitude.idxmin()
        min_avg_thrust = avg_thrust_by_altitude.min()
        print(f"Высота с минимальной средней тягой ({min_avg_thrust:.1f} кгс): {min_thrust_alt:.0f} м")
    
    df_flyable['Запас_тяги_кгс'] = df_flyable['Доступная_тяга_кгс'] - df_flyable['Требуемая_тяга_кгс']
    thrust_margin_by_altitude = df_flyable.groupby('Высота_м')['Запас_тяги_кгс'].mean()
    if not thrust_margin_by_altitude.empty:
        max_margin_alt = thrust_margin_by_altitude.idxmax()
        max_margin = thrust_margin_by_altitude.max()
        print(f"Высота с максимальным запасом тяги ({max_margin:.1f} кгс): {max_margin_alt:.0f} м")
    
    df_flyable['Эффективность'] = df_flyable['Коэф_подъемной_силы_CL'] / df_flyable['Коэф_сопротивления_CD']
    efficiency_by_altitude = df_flyable.groupby('Высота_м')['Эффективность'].mean()
    if not efficiency_by_altitude.empty:
        max_efficiency_alt = efficiency_by_altitude.idxmax()
        max_efficiency = efficiency_by_altitude.max()
        print(f"Высота с максимальной эффективностью ({max_efficiency:.2f}): {max_efficiency_alt:.0f} м")
    
    scores = {}
    
    for altitude in df_flyable['Высота_м'].unique():
        df_alt = df_flyable[df_flyable['Высота_м'] == altitude]
        
        flyable_score = len(df_alt) / len(df_flyable) if len(df_flyable) > 0 else 0
        
        thrust_score = 1 - (df_alt['Требуемая_тяга_кгс'].mean() - avg_thrust_by_altitude.min()) / \
                      (avg_thrust_by_altitude.max() - avg_thrust_by_altitude.min()) if len(avg_thrust_by_altitude) > 1 else 0.5
        
        margin_score = (df_alt['Запас_тяги_кгс'].mean() - thrust_margin_by_altitude.min()) / \
                      (thrust_margin_by_altitude.max() - thrust_margin_by_altitude.min()) if len(thrust_margin_by_altitude) > 1 else 0.5
        
        efficiency_score = (df_alt['Эффективность'].mean() - efficiency_by_altitude.min()) / \
                          (efficiency_by_altitude.max() - efficiency_by_altitude.min()) if len(efficiency_by_altitude) > 1 else 0.5
        
        total_score = (0.3 * flyable_score + 0.3 * thrust_score + 
                      0.2 * margin_score + 0.2 * efficiency_score)
        
        scores[altitude] = total_score
    
    if scores:
        optimal_altitude = max(scores.items(), key=lambda x: x[1])[0]
        best_score = max(scores.values())
        
        print(f"\nОптимальная высота: {optimal_altitude:.0f} м (общий балл: {best_score:.3f})")
        
        print("Оценки по высотам:")
        for alt, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {alt:.0f} м: {score:.3f}")
        
        return optimal_altitude
    else:
        return 100

# ====================== ОСНОВНЫЕ РАСЧЕТЫ ======================
def analyze_wing_performance(altitudes, velocities, CD0_range, mass, wingspan, chord, 
                           thrust_sl_kgf=450, tsfc=0.71,
                           sweep_angle=0, alpha=5, e=0.85):
    
    results = []
    S = wingspan * chord
    AR = wingspan / chord
    
    for altitude in altitudes:
        rho, c, T, p = atmosphere(altitude)
        mu = calculate_dynamic_viscosity(T)
        g = gravity(altitude)
        W = mass * g
        
        for V in velocities:
            # Расчет доступной тяги для данных условий
            thrust_available_kgf = calculate_turbojet_thrust(altitude, V, thrust_sl_kgf)
            thrust_available_N = kgf_to_newton(thrust_available_kgf)
            
            mach = V / c if c > 0 else 0
            L_ref = chord
            Re = calculate_reynolds(rho, V, L_ref, mu)
            
            for CD0 in CD0_range:
                CL = calculate_lift_coefficient(AR, sweep_angle, mach, Re, alpha)
                CD = calculate_drag_coefficient(CD0, AR, CL, e, mach, sweep_angle)
                
                L = calculate_lift_force(rho, V, S, CL)
                D = calculate_drag_force(rho, V, S, CD)
                
                # Расчет расхода топлива
                fuel_flow = calculate_fuel_flow(thrust_available_kgf, tsfc)
                
                # Расчет тяговооруженности
                available_twr = calculate_thrust_to_weight_ratio(thrust_available_N, mass, altitude)
                required_twr = calculate_thrust_to_weight_ratio(D, mass, altitude)
                
                # Проверка возможности полета
                thrust_check = thrust_available_N >= D
                lift_check = L >= W
                can_fly = lift_check and thrust_check
                
                result = {
                    'Высота_м': altitude,
                    'Скорость_мс': V,
                    'Число_Маха': mach,
                    'Коэф_подъемной_силы_CL': CL,
                    'Коэф_сопротивления_CD': CD,
                    'CD0': CD0,
                    'Подъемная_сила_Н': L,
                    'Сопротивление_Н': D,
                    'Требуемая_тяга_Н': D,
                    'Требуемая_тяга_кгс': newton_to_kgf(D),
                    'Доступная_тяга_кгс': thrust_available_kgf,
                    'Доступная_тяга_Н': thrust_available_N,
                    'Расход_топлива_кгч': fuel_flow,
                    'Требуемая_тяговооруженность': required_twr,
                    'Доступная_тяговооруженность': available_twr,
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
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow(["Анализ характеристик крылатого объекта"])
        writer.writerow([f"Время расчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow([])
        
        writer.writerow(["Параметры расчета:"])
        for key, value in parameters.items():
            writer.writerow([key, str(value)])
        writer.writerow([])
        
        writer.writerow(["Результаты расчетов:"])
        
        if results:
            headers = list(results[0].keys())
            writer.writerow(headers)
            
            for row in results:
                writer.writerow([row[key] for key in headers])
    
    print(f"Данные сохранены в файл: {filename}")

def create_interpolated_grid(df, param_name, grid_resolution=200):
    """Создает интерполированную сетку для smooth plotting"""
    x = df['Скорость_мс'].values
    y = df['CD0'].values
    z = df[param_name].values
    
    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    return xi, yi, zi

def plot_enhanced_results(results, parameters, fixed_altitude):
    """Улучшенная версия построения графиков с использованием точных расчетов ТРДД"""
    
    df = pd.DataFrame(results)
    df_fixed_altitude = df[np.abs(df['Высота_м'] - fixed_altitude) < 0.1].copy()
    
    if df_fixed_altitude.empty:
        print(f"Нет данных для высоты {fixed_altitude} м")
        return
    
    # Создаем figure с профессиональной компоновкой
    fig = plt.figure(figsize=(20, 16))
    
    # Определяем сетку для subplots
    gs = plt.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # График 1: ТВР с интерполяцией и заливкой
    ax1 = fig.add_subplot(gs[0, 0])
    
    try:
        # Создаем интерполированную сетку для TWR
        xi, yi, zi_twr = create_interpolated_grid(df_fixed_altitude, 'Требуемая_тяговооруженность')
        
        # Contour plot с заливкой
        contour = ax1.contourf(xi, yi, zi_twr, levels=20, alpha=0.7, cmap='viridis')
        ax1.contour(xi, yi, zi_twr, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Добавляем цветовую шкалу
        cbar = plt.colorbar(contour, ax=ax1)
        cbar.set_label('Требуемая ТВР', rotation=270, labelpad=20)
        
        # Доступная ТВР (меняется со скоростью)
        df_available_twr = df_fixed_altitude.groupby('Скорость_мс')['Доступная_тяговооруженность'].first().reset_index()
        df_available_twr = df_available_twr.sort_values('Скорость_мс')
        ax1.plot(df_available_twr['Скорость_мс'], df_available_twr['Доступная_тяговооруженность'], 
                color='r', linestyle='--', linewidth=3, label='Доступная ТВР')
        
        # Находим границу рабочей области
        working_region = np.zeros_like(zi_twr)
        for i in range(len(xi)):
            for j in range(len(yi)):
                speed = xi[i, j]
                # Находим ближайшую доступную ТВР для этой скорости
                idx = np.argmin(np.abs(df_available_twr['Скорость_мс'] - speed))
                available_twr_at_speed = df_available_twr.iloc[idx]['Доступная_тяговооруженность']
                if zi_twr[i, j] <= available_twr_at_speed:
                    working_region[i, j] = 1
        
        if np.any(working_region):
            # Заливаем рабочую область
            ax1.contourf(xi, yi, working_region, levels=[0.5, 1.5], alpha=0.3, colors=['green'], hatches=['/'])
        
        # Аннотация оптимальной точки
        min_twr_idx = np.unravel_index(np.nanargmin(zi_twr), zi_twr.shape)
        opt_speed = xi[min_twr_idx]
        opt_cd0 = yi[min_twr_idx]
        ax1.plot(opt_speed, opt_cd0, 'ro', markersize=10, markeredgecolor='white')
        ax1.annotate(f'Опт.: {opt_speed:.0f} м/с\nCD0: {opt_cd0:.3f}', 
                    xy=(opt_speed, opt_cd0), xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
    except Exception as e:
        print(f"Ошибка в графике 1: {e}")
        ax1.text(0.5, 0.5, 'Невозможно построить график\n(недостаточно данных)', 
                ha='center', va='center', transform=ax1.transAxes)
    
    ax1.set_xlabel('Скорость (м/с)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CD0', fontsize=12, fontweight='bold')
    ax1.set_title('Карта тяговооруженности с рабочей областью', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # График 2: Требуемая и доступная тяга
    ax2 = fig.add_subplot(gs[0, 1])
    
    try:
        xi, yi, zi_thrust = create_interpolated_grid(df_fixed_altitude, 'Требуемая_тяга_кгс')
        
        contour2 = ax2.contourf(xi, yi, zi_thrust, levels=20, alpha=0.7, cmap='plasma')
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label('Требуемая тяга (кгс)', rotation=270, labelpad=20)
        
        # Добавляем изолинии
        CS = ax2.contour(xi, yi, zi_thrust, levels=10, colors='white', alpha=0.6, linewidths=1)
        ax2.clabel(CS, inline=True, fontsize=8, fmt='%.0f')
        
        # Доступная тяга (меняется со скоростью)
        df_available_thrust = df_fixed_altitude.groupby('Скорость_мс')['Доступная_тяга_кгс'].first().reset_index()
        df_available_thrust = df_available_thrust.sort_values('Скорость_мс')
        ax2.plot(df_available_thrust['Скорость_мс'], df_available_thrust['Доступная_тяга_кгс'],
                color='r', linestyle='--', linewidth=3, label='Доступная тяга')
        
        # Зона превышения доступной тяги
        thrust_exceed = np.zeros_like(zi_thrust)
        for i in range(len(xi)):
            for j in range(len(yi)):
                speed = xi[i, j]
                idx = np.argmin(np.abs(df_available_thrust['Скорость_мс'] - speed))
                available_thrust_at_speed = df_available_thrust.iloc[idx]['Доступная_тяга_кгс']
                if zi_thrust[i, j] > available_thrust_at_speed:
                    thrust_exceed[i, j] = 1
        
        if np.any(thrust_exceed):
            ax2.contourf(xi, yi, thrust_exceed, levels=[0.5, 1.5], alpha=0.3, colors=['red'], label='Превышение тяги')
            
    except Exception as e:
        print(f"Ошибка в графике 2: {e}")
        ax2.text(0.5, 0.5, 'Невозможно построить график\n(недостаточно данных)', 
                ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_xlabel('Скорость (м/с)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('CD0', fontsize=12, fontweight='bold')
    ax2.set_title('Карта требуемой и доступной тяги', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    # График 3: Расход топлива
    ax3 = fig.add_subplot(gs[0, 2])
    
    try:
        df_fixed_altitude.loc[:, 'L/D'] = df_fixed_altitude['Коэф_подъемной_силы_CL'] / df_fixed_altitude['Коэф_сопротивления_CD']
        xi, yi, zi_fuel = create_interpolated_grid(df_fixed_altitude, 'Расход_топлива_кгч')
        
        contour3 = ax3.contourf(xi, yi, zi_fuel, levels=20, alpha=0.7, cmap='YlOrRd')
        cbar3 = plt.colorbar(contour3, ax=ax3)
        cbar3.set_label('Расход топлива (кг/ч)', rotation=270, labelpad=20)
        
        # Изолинии расхода топлива
        CS3 = ax3.contour(xi, yi, zi_fuel, levels=8, colors='black', alpha=0.5, linewidths=1)
        ax3.clabel(CS3, inline=True, fontsize=8, fmt='%.0f')
        
    except Exception as e:
        print(f"Ошибка в графике 3: {e}")
        ax3.text(0.5, 0.5, 'Невозможно построить график\n(недостаточно данных)', 
                ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_xlabel('Скорость (м/с)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('CD0', fontsize=12, fontweight='bold')
    ax3.set_title('Расход топлива ТРДД-50АТ', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.2)
    
    # График 4: Сравнение сил (dual axis)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4_right = ax4.twinx()
    
    try:
        median_cd0 = df_fixed_altitude['CD0'].median()
        df_median_cd0 = df_fixed_altitude[df_fixed_altitude['CD0'] == median_cd0]
        
        if not df_median_cd0.empty:
            df_grouped = df_median_cd0.groupby('Скорость_мс').agg({
                'Подъемная_сила_Н': 'mean',
                'Сопротивление_Н': 'mean',
                'Коэф_подъемной_силы_CL': 'mean',
                'Коэф_сопротивления_CD': 'mean',
                'Доступная_тяга_Н': 'first'
            }).reset_index()
            df_grouped = df_grouped.sort_values('Скорость_мс')
            
            # Силы (левая ось)
            l1 = ax4.plot(df_grouped['Скорость_мс'], df_grouped['Подъемная_сила_Н'], 
                         'g-', linewidth=3, label='Подъемная сила', alpha=0.8)
            l2 = ax4.plot(df_grouped['Скорость_мс'], df_grouped['Сопротивление_Н'], 
                         'r-', linewidth=3, label='Сопротивление', alpha=0.8)
            l5 = ax4.plot(df_grouped['Скорость_мс'], df_grouped['Доступная_тяга_Н'], 
                         'orange', linewidth=3, label='Доступная тяга', alpha=0.8)
            
            # Коэффициенты (правая ось)
            l3 = ax4_right.plot(df_grouped['Скорость_мс'], df_grouped['Коэф_подъемной_силы_CL'], 
                               'b--', linewidth=2, label='CL', alpha=0.7)
            l4 = ax4_right.plot(df_grouped['Скорость_мс'], df_grouped['Коэф_сопротивления_CD'], 
                               'm--', linewidth=2, label='CD', alpha=0.7)
            
            ax4.axhline(y=df_median_cd0['Вес_Н'].iloc[0], color='k', linestyle=':', 
                       linewidth=2, label='Вес объекта', alpha=0.6)
            
            # Объединяем легенды
            lines = l1 + l2 + l5 + l3 + l4
            labels = [l.get_label() for l in lines]
            labels.append('Вес объекта')
            ax4.legend(lines + [ax4.lines[-1]], labels, loc='upper left')
            
    except Exception as e:
        print(f"Ошибка в графике 4: {e}")
        ax4.text(0.5, 0.5, 'Невозможно построить график\n(недостаточно данных)', 
                ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_xlabel('Скорость (м/с)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Силы (Н)', fontsize=12, fontweight='bold', color='green')
    ax4_right.set_ylabel('Коэффициенты', fontsize=12, fontweight='bold', color='blue')
    ax4.set_title('Силы, тяга и коэффициенты vs Скорость', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.2)
    
    # График 5: Запас по тяге
    ax5 = fig.add_subplot(gs[1, 1])
    
    try:
        df_fixed_altitude.loc[:, 'Запас_тяги_кгс'] = df_fixed_altitude['Доступная_тяга_кгс'] - df_fixed_altitude['Требуемая_тяга_кгс']
        xi, yi, zi_margin = create_interpolated_grid(df_fixed_altitude, 'Запас_тяги_кгс')
        
        # Безопасная нормализация для TwoSlopeNorm
        vmin, vmax = np.nanmin(zi_margin), np.nanmax(zi_margin)
        if vmin < 0 < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        contour5 = ax5.contourf(xi, yi, zi_margin, levels=20, alpha=0.7, cmap='RdYlGn', norm=norm)
        cbar5 = plt.colorbar(contour5, ax=ax5)
        cbar5.set_label('Запас тяги (кгс)', rotation=270, labelpad=20)
        
        # Линия нулевого запаса
        if vmin < 0 < vmax:
            CS5 = ax5.contour(xi, yi, zi_margin, levels=[0], colors='black', linewidths=2, linestyles='--')
            ax5.clabel(CS5, inline=True, fontsize=10, fmt='%.0f')
            
    except Exception as e:
        print(f"Ошибка в графике 5: {e}")
        ax5.text(0.5, 0.5, 'Невозможно построить график\n(недостаточно данных)', 
                ha='center', va='center', transform=ax5.transAxes)
    
    ax5.set_xlabel('Скорость (м/с)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('CD0', fontsize=12, fontweight='bold')
    ax5.set_title('Запас тяги', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.2)
    
    # График 6: Область рабочих режимов
    ax6 = fig.add_subplot(gs[1, 2])
    
    try:
        # Создаем бинарную маску рабочих режимов
        working_mask = (df_fixed_altitude['Требуемая_тяговооруженность'] <= df_fixed_altitude['Доступная_тяговооруженность']) & \
                      (df_fixed_altitude['Подъемная_сила_Н'] >= df_fixed_altitude['Вес_Н'])
        
        df_working = df_fixed_altitude[working_mask]
        
        if not df_working.empty:
            # Scatter plot рабочих точек
            scatter = ax6.scatter(df_working['Скорость_мс'], df_working['CD0'], 
                                 c=df_working['Расход_топлива_кгч'], 
                                 cmap='coolwarm', s=50, alpha=0.7)
            plt.colorbar(scatter, ax=ax6, label='Расход топлива (кг/ч)')
            
            # Вычисляем выпуклую оболочку рабочих точек
            from scipy.spatial import ConvexHull
            points = df_working[['Скорость_мс', 'CD0']].values
            if len(points) > 2:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax6.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.5)
                
                # Заливаем область
                hull_points = points[hull.vertices]
                hull_patch = patches.Polygon(hull_points, alpha=0.2, color='green', label='Рабочая область')
                ax6.add_patch(hull_patch)
        else:
            ax6.text(0.5, 0.5, 'Нет рабочих режимов', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            
    except Exception as e:
        print(f"Ошибка в графике 6: {e}")
        ax6.text(0.5, 0.5, 'Невозможно построить график\n(недостаточно данных)', 
                ha='center', va='center', transform=ax6.transAxes)
    
    ax6.set_xlabel('Скорость (м/с)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('CD0', fontsize=12, fontweight='bold')
    ax6.set_title('Область рабочих режимов', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.2)
    ax6.legend()
    
    # График 7: Аэродинамическое качество
    ax7 = fig.add_subplot(gs[2, :])
    
    try:
        df_fixed_altitude.loc[:, 'L/D'] = df_fixed_altitude['Коэф_подъемной_силы_CL'] / df_fixed_altitude['Коэф_сопротивления_CD']
        
        # Анализ эффективности по скорости
        speeds = df_fixed_altitude['Скорость_мс'].unique()
        cd0_values = df_fixed_altitude['CD0'].unique()
        
        efficiency_data = []
        for cd0 in cd0_values:
            df_cd0 = df_fixed_altitude[df_fixed_altitude['CD0'] == cd0]
            if not df_cd0.empty:
                ld_vs_speed = df_cd0.groupby('Скорость_мс')['L/D'].mean()
                efficiency_data.append((cd0, ld_vs_speed))
        
        # Построение семейства кривых
        for cd0, ld_data in efficiency_data:
            ax7.plot(ld_data.index, ld_data.values, 
                    linewidth=2, alpha=0.7, label=f'CD0 = {cd0:.3f}')
        
        ax7.set_xlabel('Скорость (м/с)', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Аэродинамическое качество (L/D)', fontsize=12, fontweight='bold')
        ax7.set_title('Аэродинамическое качество vs Скорость', fontsize=14, fontweight='bold')
        ax7.legend(ncol=3, loc='upper center')
        
    except Exception as e:
        print(f"Ошибка в графике 7: {e}")
        ax7.text(0.5, 0.5, 'Невозможно построить график\n(недостаточно данных)', 
                ha='center', va='center', transform=ax7.transAxes)
    
    ax7.grid(True, alpha=0.2)
    
    # Общий заголовок
    fig.suptitle(
        f"Комплексный анализ характеристик Х-101 с ТРДД-50АТ\n"
        f"Масса: {parameters['Масса_кг']} кг | Тяга на ур.моря: {parameters['Тяга_на_уровне_моря_кгс']} кгс | "
        f"Размах: {parameters['Размах_крыла_м']} м | Высота: {fixed_altitude} м", 
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Добавляем информационную панель
    info_text = (
        f"Анализ выполнен для высоты: {fixed_altitude} м\n"
        f"Тяга на ур.моря: {parameters['Тяга_на_уровне_моря_кгс']} кгс\n"
        f"Уд.расход топлива: {parameters['Удельный_расход_топлива_кг_кгс_ч']} кг/кгс·ч\n"
        f"Масса: {parameters['Масса_кг']} кг\n"
        f"Размах крыла: {parameters['Размах_крыла_м']} м"
    )
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('enhanced_wing_analysis_turbojet.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def plot_professional_elements(results, parameters, fixed_altitude):
    """Отдельная визуализация профессиональных элементов с ТРДД"""
    
    df = pd.DataFrame(results)
    df_fixed_altitude = df[np.abs(df['Высота_м'] - fixed_altitude) < 0.1].copy()
    
    if df_fixed_altitude.empty:
        print("Нет данных для построения профессиональных элементов")
        return
    
    fig = plt.figure(figsize=(18, 12))
    
    # График 1: Техническая таблица с предельными значениями
    ax1 = fig.add_subplot(231)
    ax1.axis('off')
    
    try:
        # Вычисляем предельные значения
        df_flyable = df_fixed_altitude[df_fixed_altitude['Возможность_полета'] == True]
        
        if not df_flyable.empty:
            min_thrust = df_flyable['Требуемая_тяга_кгс'].min()
            max_speed = df_flyable['Скорость_мс'].max()
            min_speed = df_flyable['Скорость_мс'].min()
            max_ld = (df_flyable['Коэф_подъемной_силы_CL'] / df_flyable['Коэф_сопротивления_CD']).max()
            min_fuel = df_flyable['Расход_топлива_кгч'].min()
            max_fuel = df_flyable['Расход_топлива_кгч'].max()
            
            table_data = [
                ["Параметр", "Значение", "Единицы"],
                ["Минимальная тяга", f"{min_thrust:.1f}", "кгс"],
                ["Макс. скорость", f"{max_speed:.1f}", "м/с"],
                ["Мин. скорость", f"{min_speed:.1f}", "м/с"],
                ["Макс. L/D", f"{max_ld:.2f}", ""],
                ["Расход топлива", f"{min_fuel:.0f}-{max_fuel:.0f}", "кг/ч"],
                ["Диапазон CD0", f"{df_flyable['CD0'].min():.3f}-{df_flyable['CD0'].max():.3f}", ""],
                ["Тяга на ур.моря", f"{parameters['Тяга_на_уровне_моря_кгс']}", "кгс"]
            ]
        else:
            # Если нет летных режимов, показываем общую информацию
            min_thrust = df_fixed_altitude['Требуемая_тяга_кгс'].min()
            min_fuel = df_fixed_altitude['Расход_топлива_кгч'].min()
            max_fuel = df_fixed_altitude['Расход_топлива_кгч'].max()
            
            table_data = [
                ["Параметр", "Значение", "Единицы"],
                ["Минимальная тяга", f"{min_thrust:.1f}", "кгс"],
                ["Макс. скорость", "Н/Д", "м/с"],
                ["Мин. скорость", "Н/Д", "м/с"],
                ["Макс. L/D", "Н/Д", ""],
                ["Расход топлива", f"{min_fuel:.0f}-{max_fuel:.0f}", "кг/ч"],
                ["Диапазон CD0", f"{df_fixed_altitude['CD0'].min():.3f}-{df_fixed_altitude['CD0'].max():.3f}", ""],
                ["Тяга на ур.моря", f"{parameters['Тяга_на_уровне_моря_кгс']}", "кгс"],
                ["Режимы полета", "Нет", ""]
            ]
        
        table = ax1.table(cellText=table_data, cellLoc='center', 
                         loc='center', bbox=[0.1, 0.2, 0.8, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax1.set_title('Предельные характеристики ТРДД', fontsize=14, fontweight='bold')
        
    except Exception as e:
        print(f"Ошибка в таблице: {e}")
        ax1.text(0.5, 0.5, 'Невозможно построить таблицу', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # График 2: Векторная диаграмма сил для оптимальной точки
    ax2 = fig.add_subplot(232)
    
    try:
        df_flyable = df_fixed_altitude[df_fixed_altitude['Возможность_полета'] == True]
        
        if not df_flyable.empty:
            # Находим оптимальную точку (минимальная тяга)
            optimal_point = df_flyable.loc[df_flyable['Требуемая_тяга_кгс'].idxmin()]
            
            # Векторная диаграмма
            forces = {
                'Lift': optimal_point['Подъемная_сила_Н'],
                'Drag': optimal_point['Сопротивление_Н'],
                'Weight': optimal_point['Вес_Н'],
                'Thrust': optimal_point['Доступная_тяга_Н']
            }
            
            # Нормализуем для визуализации
            max_force = max(forces.values())
            scale_factor = 100 / max_force
            
            # Рисуем векторы
            colors = {'Lift': 'green', 'Drag': 'red', 'Weight': 'blue', 'Thrust': 'orange'}
            
            # Lift (вертикально вверх)
            ax2.arrow(0, 0, 0, forces['Lift'] * scale_factor, head_width=5, head_length=10, 
                      fc=colors['Lift'], ec=colors['Lift'], linewidth=3, label='Подъемная')
            
            # Weight (вертикально вниз)
            ax2.arrow(0, 0, 0, -forces['Weight'] * scale_factor, head_width=5, head_length=10, 
                      fc=colors['Weight'], ec=colors['Weight'], linewidth=3, label='Вес')
            
            # Drag (горизонтально назад)
            ax2.arrow(0, 0, -forces['Drag'] * scale_factor, 0, head_width=5, head_length=10, 
                      fc=colors['Drag'], ec=colors['Drag'], linewidth=3, label='Сопротивление')
            
            # Thrust (горизонтально вперед)
            ax2.arrow(0, 0, forces['Thrust'] * scale_factor, 0, head_width=5, head_length=10, 
                      fc=colors['Thrust'], ec=colors['Thrust'], linewidth=3, label='Тяга ТРДД')
            
            ax2.set_xlim(-120, 120)
            ax2.set_ylim(-120, 120)
            ax2.set_aspect('equal')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Нет данных для\nвекторной диаграммы', 
                    ha='center', va='center', transform=ax2.transAxes)
            
    except Exception as e:
        print(f"Ошибка в векторной диаграмме: {e}")
        ax2.text(0.5, 0.5, 'Невозможно построить\nвекторную диаграмму', 
                ha='center', va='center', transform=ax2.transAxes)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Векторная диаграмма сил\n(оптимальная точка)', fontsize=14, fontweight='bold')
    
    # График 3: 3D поверхность тяги
    ax3 = fig.add_subplot(233, projection='3d')
    
    try:
        xi, yi, zi_thrust = create_interpolated_grid(df_fixed_altitude, 'Требуемая_тяга_кгс')
        
        surf = ax3.plot_surface(xi, yi, zi_thrust, cmap='viridis', alpha=0.8, 
                               linewidth=0, antialiased=True)
        
        ax3.set_xlabel('Скорость (м/с)')
        ax3.set_ylabel('CD0')
        ax3.set_zlabel('Требуемая тяга (кгс)')
        
    except Exception as e:
        print(f"Ошибка в 3D графике: {e}")
        ax3.text(0.5, 0.5, 0.5, 'Невозможно построить\n3D поверхность', 
                ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_title('3D поверхность требуемой тяги', fontsize=14, fontweight='bold')
    
    # График 4: Radar chart характеристик
    ax4 = fig.add_subplot(234, polar=True)
    
    try:
        df_flyable = df_fixed_altitude[df_fixed_altitude['Возможность_полета'] == True]
        if not df_flyable.empty:
            # Выбираем лучшую точку по минимальной тяге
            best_point = df_flyable.loc[df_flyable['Требуемая_тяга_кгс'].idxmin()]
            
            # Параметры для radar chart
            categories = ['Эффективность', 'Запас тяги', 'Скорость', 'Экономичность', 'Надежность']
            
            # Нормализованные значения (0-1)
            efficiency = best_point['Коэф_подъемной_силы_CL'] / best_point['Коэф_сопротивления_CD'] / 15
            thrust_margin = (best_point['Доступная_тяга_кгс'] - best_point['Требуемая_тяга_кгс']) / best_point['Доступная_тяга_кгс']
            speed_norm = (best_point['Скорость_мс'] - 190) / (220 - 190)
            fuel_efficiency = 1 - (best_point['Расход_топлива_кгч'] / 400)  # Нормализация
            reliability = 0.8
            
            values = [efficiency, thrust_margin, speed_norm, fuel_efficiency, reliability]
            values += values[:1]
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2, label='Оптимальная точка')
            ax4.fill(angles, values, alpha=0.25)
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories)
        else:
            ax4.text(0.5, 0.5, 'Нет данных для\nradar chart', 
                    ha='center', va='center', transform=ax4.transAxes)
            
    except Exception as e:
        print(f"Ошибка в radar chart: {e}")
        ax4.text(0.5, 0.5, 'Невозможно построить\nradar chart', 
                ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_title('Radar chart характеристик ТРДД', fontsize=14, fontweight='bold')
    
    # График 5: Сравнение с эталоном
    ax5 = fig.add_subplot(235)
    
    try:
        # Создаем эталонные кривые
        speeds_ref = np.linspace(190, 220, 50)
        
        # Идеальная кривая (квадратичная зависимость)
        thrust_ideal = 200 + 0.02 * (speeds_ref - 200)**2
        
        # Фактическая кривая для среднего CD0
        median_cd0 = df_fixed_altitude['CD0'].median()
        df_median = df_fixed_altitude[df_fixed_altitude['CD0'] == median_cd0]
        thrust_actual = df_median.groupby('Скорость_мс')['Требуемая_тяга_кгс'].mean()
        
        # Доступная тяга
        thrust_available = df_fixed_altitude.groupby('Скорость_мс')['Доступная_тяга_кгс'].first()
        
        ax5.plot(speeds_ref, thrust_ideal, 'g--', linewidth=2, label='Идеальный профиль', alpha=0.7)
        ax5.plot(thrust_actual.index, thrust_actual.values, 'b-', linewidth=3, label='Фактический профиль')
        ax5.plot(thrust_available.index, thrust_available.values, 'r-', linewidth=3, label='Доступная тяга ТРДД')
        
        ax5.set_xlabel('Скорость (м/с)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Тяга (кгс)', fontsize=12, fontweight='bold')
        ax5.legend()
        
    except Exception as e:
        print(f"Ошибка в графике сравнения: {e}")
        ax5.text(0.5, 0.5, 'Невозможно построить график\nсравнения', 
                ha='center', va='center', transform=ax5.transAxes)
    
    ax5.set_title('Сравнение с эталоном', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # График 6: Распределение расхода топлива
    ax6 = fig.add_subplot(236)
    
    try:
        df_flyable = df_fixed_altitude[df_fixed_altitude['Возможность_полета'] == True]
        if not df_flyable.empty:
            parameters_to_plot = ['Требуемая_тяга_кгс', 'Расход_топлива_кгч', 'Коэф_подъемной_силы_CL']
            colors_hist = ['skyblue', 'lightgreen', 'lightcoral']
            
            for i, param in enumerate(parameters_to_plot):
                ax6.hist(df_flyable[param], bins=20, alpha=0.6, color=colors_hist[i], 
                        label=param.replace('_', ' '))
            
            ax6.set_xlabel('Значения параметров', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Частота', fontsize=12, fontweight='bold')
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'Нет данных для\nпостроения гистограммы', 
                   ha='center', va='center', transform=ax6.transAxes)
            
    except Exception as e:
        print(f"Ошибка в гистограмме: {e}")
        ax6.text(0.5, 0.5, 'Невозможно построить гистограмму', 
               ha='center', va='center', transform=ax6.transAxes)
    
    ax6.set_title('Распределение параметров ТРДД', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('professional_elements_turbojet.png', dpi=300, bbox_inches='tight')
    plt.show()

# ====================== ПАРАМЕТРЫ РАСЧЕТА ======================
if __name__ == "__main__":
    altitudes = np.arange(25, 150, 10)
    velocities = np.linspace(100, 290, 100)
    CD0_range = np.arange(0.1, 0.41, 0.02)
    
    mass = 2200
    wingspan = 4.3
    chord = 0.6
    
    # Параметры ТРДД-50АТ
    thrust_sl_kgf = 450
    tsfc = 0.71
    
    sweep_angle = 8
    alpha = 10
    e = 0.85
    
    parameters = {
        "Масса_кг": mass,
        "Размах_крыла_м": wingspan,
        "Хорда_м": chord,
        "Площадь_крыла_м2": wingspan * chord,
        "Удлинение_крыла": wingspan / chord,
        "Тяга_на_уровне_моря_кгс": thrust_sl_kgf,
        "Удельный_расход_топлива_кг_кгс_ч": tsfc,
        "Угол_стреловидности_град": sweep_angle,
        "Угол_атаки_град": alpha,
        "Коэффициент_Освальда": e,
        "Диапазон_высот_м": f"{altitudes[0]} - {altitudes[-1]}",
        "Диапазон_скоростей_мс": f"{velocities[0]:.1f} - {velocities[-1]:.1f}",
        "Диапазон_CD0": f"{CD0_range[0]:.3f} - {CD0_range[-1]:.3f}"
    }
    
    print("Выполнение аэродинамических расчетов с моделью ТРДД...")
    results = analyze_wing_performance(
        altitudes=altitudes,
        velocities=velocities,
        CD0_range=CD0_range,
        mass=mass,
        wingspan=wingspan,
        chord=chord,
        thrust_sl_kgf=thrust_sl_kgf,
        tsfc=tsfc,
        sweep_angle=sweep_angle,
        alpha=alpha,
        e=e
    )
    
    print("\nАнализ оптимальной высоты...")
    FIXED_ALTITUDE = find_optimal_altitude(results)
    parameters["Оптимальная_высота_для_графиков_м"] = FIXED_ALTITUDE
    
    csv_filename = os.path.join(os.getcwd(), "wing_analysis_turbojet_results.csv")
    save_to_csv(csv_filename, results, parameters)
    
    print(f"\nРасчет завершен. Обработано {len(results)} комбинаций параметров.")
    
    flyable_count = sum(1 for r in results if r['Возможность_полета'])
    print(f"Возможных режимов полета: {flyable_count} из {len(results)}")
    
    if flyable_count > 0:
        best_regime = min(results, key=lambda x: x['Требуемая_тяга_кгс'] if x['Возможность_полета'] else float('inf'))
        print(f"Наименьшая требуемая тяга: {best_regime['Требуемая_тяга_кгс']:.1f} кгс "
              f"(высота: {best_regime['Высота_м']} м, скорость: {best_regime['Скорость_мс']} м/с, CD0: {best_regime['CD0']:.3f})")
        
        # Анализ расхода топлива
        min_fuel_regime = min(results, key=lambda x: x['Расход_топлива_кгч'] if x['Возможность_полета'] else float('inf'))
        print(f"Минимальный расход топлива: {min_fuel_regime['Расход_топлива_кгч']:.1f} кг/ч "
              f"(высота: {min_fuel_regime['Высота_м']} м, скорость: {min_fuel_regime['Скорость_мс']} м/с)")
    
    print("\nПостроение улучшенных графиков с ТРДД...")
    plot_enhanced_results(results, parameters, FIXED_ALTITUDE)
    
    print("\nПостроение профессиональных элементов с ТРДД...")
    plot_professional_elements(results, parameters, FIXED_ALTITUDE)