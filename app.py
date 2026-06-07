import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import line_search
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# Configuración visual
st.set_page_config(page_title="OptiWeb - Métodos de Optimización", layout="wide")
st.title("🧮 Aplicación Web de Optimización Numérica")
st.caption("Proyecto Final - Métodos de Optimización")

# --- PANEL DE ENTRADAS ---
st.sidebar.header("Datos de Entrada")

num_vars = st.sidebar.number_input("Número de variables", min_value=1, max_value=20, value=2)
vars_symbols = sp.symbols(f'x1:{num_vars+1}')
st.sidebar.info(f"Variables habilitadas: {', '.join([str(v) for v in vars_symbols])}")

func_str = st.sidebar.text_input("Función objetivo (ej: x1**2 + x2**2)", value="x1**2 + 2*x2**2")
metodo = st.sidebar.selectbox("Método de optimización", ["Gradiente", "Gradiente Conjugado", "Newton"])
start_str = st.sidebar.text_input("Punto de partida (separado por comas)", value=", ".join(["1.0"] * num_vars))

max_iter = st.sidebar.number_input("Número máximo de iteraciones", min_value=1, max_value=1000, value=100)
tol = st.sidebar.number_input("Tolerancia de convergencia", min_value=1e-7, max_value=1e-1, value=1e-5, format="%.7f")

st.sidebar.subheader("Condiciones de Wolfe")
c1 = st.sidebar.number_input("c1 (Armijo)", min_value=1e-4, max_value=0.3, value=1e-4, format="%.4f")
c2 = st.sidebar.number_input("c2 (Curvatura)", min_value=0.1, max_value=0.9, value=0.9, format="%.2f")

# --- PROCESAMIENTO MATEMÁTICO (VALIDACIÓN ROBUSTA) ---
try:
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    local_dict = {str(s): s for s in vars_symbols}
    
    # 1. Validar lectura de la función
    try:
        f_expr = parse_expr(func_str, transformations=transformations, local_dict=local_dict)
    except Exception as math_err:
        st.error(f"⚠️ **Error en la función objetivo:** La expresión matemática es inválida o contiene errores de tipeo. Revisa si olvidaste algún operador (ej: usar '*' para multiplicar explícitamente).")
        st.stop()
        
    # 2. Validar cálculo analítico de derivadas
    try:
        grad_expr = [sp.diff(f_expr, v) for v in vars_symbols]
        hessian_expr = [[sp.diff(g, v) for v in vars_symbols] for g in grad_expr]
    except Exception as diff_err:
        st.error(f"⚠️ **Error de diferenciación:** No se pudieron calcular las derivadas simbólicas de la función introducida.")
        st.stop()

    f_num = sp.lambdify(vars_symbols, f_expr, 'numpy')
    grad_num = sp.lambdify(vars_symbols, grad_expr, 'numpy')
    hessian_num = sp.lambdify(vars_symbols, hessian_expr, 'numpy')

    def f(x): return float(f_num(*x))
    def grad(x): return np.array(grad_num(*x), dtype=float).flatten()
    def hessian(x): return np.array(hessian_num(*x), dtype=float).reshape(num_vars, num_vars)

    # 3. Validar consistencia del punto de partida
    try:
        x0 = np.array([float(x.strip()) for x in start_str.split(",")], dtype=float)
    except Exception:
        st.error("⚠️ **Error en el punto de partida:** Asegúrate de ingresar únicamente números separados por comas (ej: 1.0, 2.5).")
        st.stop()
        
    if len(x0) != num_vars:
        st.error(f"⚠️ **Dimensión no coincide:** Ingresaste {len(x0)} coordenadas en el punto de partida, pero seleccionaste {num_vars} variables en la configuración.")
        st.stop()
        
except Exception as e:
    st.error(f"Error inesperado en el procesamiento matemático: {e}")
    st.stop()

# --- ALGORITMO ---
def optimizar(metodo, x0, max_iter, tol, c1, c2):
    x = x0.copy()
    historial_error = []
    historial_tabla = []
    historial_x = [x0.copy()]
    iteraciones = 0
    criterio = "Máximo de iteraciones alcanzado"
    d = -grad(x)
    
    for k in range(max_iter):
        g = grad(x)
        error_actual = np.linalg.norm(g)
        historial_error.append(error_actual)
        historial_tabla.append({
            "Iteración": k + 1, 
            "Punto Actual (x)": str(np.round(x, 4)), 
            "f(x)": f(x), 
            "||∇f|| (Error)": error_actual
        })
        
        if error_actual < tol:
            criterio = f"Convergencia alcanzada exitosamente (||∇f|| < {tol})"
            break
            
        if metodo == "Gradiente":
            d = -g
        elif metodo == "Gradiente Conjugado" and k > 0:
            g_anterior = grad(historial_x[-2])
            beta = np.dot(g, g) / (np.dot(g_anterior, g_anterior) + 1e-10)
            d = -g + beta * d
        elif metodo == "Newton":
            H = hessian(x)
            try:
                H = (H + H.T) / 2.0  # Forzar simetría numérica exacta
                if num_vars > 1:
                    vals = np.linalg.eigvals(H)
                    min_vac = np.min(vals)
                    # Si no es definida positiva, regularizamos (Levenberg-Marquardt)
                    if min_vac <= 1e-5:
                        H += (abs(min_vac) + 0.5) * np.eye(num_vars)
                else:
                    if H[0,0] <= 1e-5:
                        H[0,0] = 0.5
                
                d = np.linalg.solve(H, -g)
                # Resguardo: si la dirección no es de descenso, recurrir al gradiente
                if np.dot(d, g) >= 0:
                    d = -g
            except Exception:
                d = -g
        
        # Búsqueda de línea con Wolfe y resguardo ante fallas de región
        res_wolfe = line_search(f, grad, x, d, c1=c1, c2=c2)
        alpha = res_wolfe[0]
        if alpha is None:
            alpha = 0.1
            for _ in range(5):
                if f(x + alpha * d) < f(x):
                    break
                alpha *= 0.5
        
        x = x + alpha * d 
        historial_x.append(x.copy())
        iteraciones += 1
        
    return x, f(x), iteraciones, error_actual, criterio, historial_error, historial_tabla

# --- RENDERIZADO DE RESULTADOS ---
if st.sidebar.button("Ejecutar Optimización"):
    x_min, f_min, iters, err_final, criterio, errores, tabla_pasos = optimizar(metodo, x0, max_iter, tol, c1, c2)
    
    st.header("📊 Resultados del Método")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric(label="Punto mínimo encontrado (x*)", value=str(np.round(x_min, 5)))
        st.metric(label="Valor de la función objetivo f(x*)", value=f"{f_min:.6f}")
        st.metric(label="Número de iteraciones realizadas", value=iters)
        st.metric(label="Error final (||∇f||)", value=f"{err_final:.2e}")
        
        # Corrección del texto truncado: Mostrado en un contenedor amplio
        st.info(f"**Criterio de parada alcanzado:** {criterio}")
    
    with col2:
        # Corrección del gráfico: Ejes explícitos, legibles y con márgenes limpios
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(errores) + 1), errores, marker='o', color='#FF4B4B', linewidth=2)
        ax.set_yscale('log')
        ax.set_xlabel("Iteraciones", fontsize=11, fontweight='bold', labelpad=6)
        ax.set_ylabel("Error (||∇f||)", fontsize=11, fontweight='bold', labelpad=6)
        ax.set_title("Gráfico de Convergencia", fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.subheader("📐 Modelamiento Simbólico Analítico")
        st.write("**Gradiente analítico calculado $\\nabla f$:**")
        st.latex(sp.latex(grad_expr))
    
    with v_col2:
        st.subheader("🧮 Matriz Hessiana Analítica")
        st.write("**Matriz Hessiana $H$:**")
        st.latex(sp.latex(sp.Matrix(hessian_expr)))

    st.subheader("📋 Historial Completo Paso a Paso")
    st.dataframe(tabla_pasos, use_container_width=True)

else:
    st.info("Configura los parámetros en el panel izquierdo y presiona 'Ejecutar Optimización'.")
