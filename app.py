import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import line_search

# Configuración visual
st.set_page_config(page_title="OptiWeb - Métodos de Optimización", layout="wide")
st.title("🧮 Aplicación Web de Optimización Numérica")
st.caption("Proyecto Final - Métodos de Optimización")

# --- PANEL DE ENTRADAS ---
st.sidebar.header("Datos de Entrada")

num_vars = st.sidebar.number_input("Número de variables", min_value=1, max_value=20, value=2)
vars_symbols = sp.symbols(f'x1:{num_vars+1}')
st.sidebar.info(f"Variables habilitadas: {', '.join([str(v) for v in vars_symbols])}")

func_str = st.sidebar.text_input("Función objetivo (ej: x1\\*\\*2 + x2\\*\\*2)", value="x1**2 + 2*x2**2")
metodo = st.sidebar.selectbox("Método de optimización", ["Gradiente", "Gradiente Conjugado", "Newton"])
start_str = st.sidebar.text_input("Punto de partida (separado por comas)", value=", ".join(["1.0"] * num_vars))

max_iter = st.sidebar.number_input("Número máximo de iteraciones", min_value=1, max_value=1000, value=100)
tol = st.sidebar.number_input("Tolerancia de convergencia", min_value=1e-7, max_value=1e-1, value=1e-5, format="%.7f")

st.sidebar.subheader("Condiciones de Wolfe")
c1 = st.sidebar.number_input("c1 (Armijo)", min_value=1e-4, max_value=0.3, value=1e-4, format="%.4f")
c2 = st.sidebar.number_input("c2 (Curvatura)", min_value=0.1, max_value=0.9, value=0.9, format="%.2f")

# --- PROCESAMIENTO MATEMÁTICO ---
# Importamos herramientas avanzadas de parseo
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

try:
    # Creamos un filtro para que entienda operaciones implícitas (ej: 2(x) -> 2*x) y potencias con ^
    transformaciones = standard_transformations + (implicit_multiplication_application, convert_xor)
    
    # Procesamos la cadena de texto con el filtro inteligente
    f_expr = parse_expr(func_str, transformations=transformations)
    
    grad_expr = [sp.diff(f_expr, v) for v in vars_symbols]
    hessian_expr = [[sp.diff(g, v) for v in vars_symbols] for g in grad_expr]

    f_num = sp.lambdify(vars_symbols, f_expr, 'numpy')
    grad_num = sp.lambdify(vars_symbols, grad_expr, 'numpy')
    hessian_num = sp.lambdify(vars_symbols, hessian_expr, 'numpy')

    def f(x): return float(f_num(*x))
    def grad(x): return np.array(grad_num(*x), dtype=float).flatten()
    def hessian(x): return np.array(hessian_num(*x), dtype=float).reshape(num_vars, num_vars)

    x0 = np.array([float(x.strip()) for x in start_str.split(",")], dtype=float)
except Exception as e:
    st.error(f"Error en la entrada matemática: {e}")
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
            criterio = f"Convergencia alcanzada (||∇f|| < {tol})"
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
                d = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                d = -g
        
        # Búsqueda de línea bajo condiciones de Wolfe
        res_wolfe = line_search(f, grad, x, d, c1=c1, c2=c2)
        alpha = res_wolfe[0] if res_wolfe[0] is not None else 0.01
        
        x = x + alpha * d 
        historial_x.append(x.copy())
        iteraciones += 1
        
    return x, f(x), iteraciones, error_actual, criterio, historial_error, historial_tabla

# --- RENDERIZADO DE RESULTADOS ---
if st.sidebar.button("Ejecutar Optimización"):
    x_min, f_min, iters, err_final, criterio, errores, tabla_pasos = optimizar(metodo, x0, max_iter, tol, c1, c2)
    
    st.header("📊 Resultados del Método")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Punto mínimo encontrado (x*)", value=str(np.round(x_min, 5)))
        st.metric(label="Valor de la función objetivo f(x*)", value=f"{f_min:.6f}")
        st.metric(label="Número de iteraciones realizadas", value=iters)
        st.metric(label="Criterio de parada / Error final", value=f"{err_final:.2e} ({criterio})")
    
        
    with col2:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(errores) + 1), errores, marker='o', color='#FF4B4B')
        ax.set_yscale('log')
        ax.set_xlabel("Iteraciones")
        ax.set_ylabel("Error (||∇f||)")
        ax.set_title("Gráfico de Convergencia")
        ax.grid(True, which="both", linestyle="--")
        st.pyplot(fig)

    # --- SECCIÓN DE VALOR AGREGADO ---
    st.markdown("---")
    
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.subheader("📐 Modelamiento Simbólico Analítico (Valor Agregado)")
        st.write("**Gradiente analítico calculado $\\nabla f$:**")
        st.latex(sp.latex(grad_expr))
    
    with v_col2:
        st.subheader("🧮 Matriz Hessiana Analítica")
        st.write("**Matriz Hessiana $H$:**")
        st.latex(sp.latex(sp.Matrix(hessian_expr)))

    st.subheader("📋 Historial Completo Paso a Paso")
    st.dataframe(tabla_pasos, use_container_width=True)

else:
    st.info("Configura los parámetros en el panel izquierdo (puedes subir el número de variables a 10 o más) y presiona 'Ejecutar Optimización'.")
