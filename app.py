import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import line_search
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# Configuracion visual de la pagina web
st.set_page_config(page_title="OptiWeb - Metodos de Optimizacion", layout="wide")
st.title("Aplicacion Web de Optimizacion Numerica")
st.caption("Proyecto Final — Metodos de Optimizacion")

# --- PANEL DE ENTRADAS ORDENADO (SIDEBAR) ---
st.sidebar.header("Configuracion del Sistema")

with st.sidebar.expander("1. Variables y Funcion Objetivo", expanded=True):
    num_vars = st.number_input("Numero de variables", min_value=1, max_value=20, value=2)
    vars_symbols = sp.symbols(f'x1:{num_vars+1}')
    st.info(f"Variables habilitadas: {', '.join([str(v) for v in vars_symbols])}")
    func_str = st.text_input("Funcion objetivo", value="x1**2 + 2*x2**2")

with st.sidebar.expander("2. Algoritmo de Optimizacion", expanded=True):
    modo_comparativo = st.checkbox("Habilitar Modo Comparativo", value=False)
    
    if not modo_comparativo:
        metodo = st.selectbox("Metodo a ejecutar", ["Gradiente", "Gradiente Conjugado", "Newton"])
        metodos_seleccionados = [metodo]
    else:
        # Implementacion del checklist con casillas independientes
        st.write("Selecciona los metodos a comparar:")
        check_gradiente = st.checkbox("Gradiente", value=True)
        check_conjugado = st.checkbox("Gradiente Conjugado", value=True)
        check_newton = st.checkbox("Newton", value=False)
        
        # Construccion de la lista basada en la seleccion del usuario
        metodos_seleccionados = []
        if check_gradiente:
            metodos_seleccionados.append("Gradiente")
        if check_conjugado:
            metodos_seleccionados.append("Gradiente Conjugado")
        if check_newton:
            metodos_seleccionados.append("Newton")
            
        if len(metodos_seleccionados) < 2:
            st.warning("Aviso: Selecciona al menos 2 metodos para poder realizar una comparativa.")
        metodo = None
        
    start_str = st.text_input("Punto de partida (separado por comas)", value=", ".join(["1.0"] * num_vars))
    max_iter = st.number_input("Iteraciones maximas", min_value=1, max_value=1000, value=100)
    tol = st.number_input("Tolerancia (Epsilon)", min_value=1e-7, max_value=1e-1, value=1e-5, format="%.7f")

with st.sidebar.expander("3. Condiciones de Busqueda de Linea (Wolfe)", expanded=False):
    c1 = st.number_input("c1 (Armijo)", min_value=1e-4, max_value=0.3, value=1e-4, format="%.4f")
    c2 = st.number_input("c2 (Curvatura)", min_value=0.1, max_value=0.9, value=0.9, format="%.2f")

# --- PROCESAMIENTO MATEMATICO ---
try:
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    local_dict = {str(s): s for s in vars_symbols}
    
    try:
        f_expr = parse_expr(func_str, transformations=transformations, local_dict=local_dict)
    except Exception:
        st.error("Error en la funcion objetivo: Expresion matematica invalida o con errores de sintaxis.")
        st.stop()
        
    try:
        grad_expr = [sp.diff(f_expr, v) for v in vars_symbols]
        hessian_expr = [[sp.diff(g, v) for v in vars_symbols] for g in grad_expr]
    except Exception:
        st.error("Error de diferenciacion: No se pudieron computar las derivadas simbolicas de este modelo.")
        st.stop()

    f_num = sp.lambdify(vars_symbols, f_expr, 'numpy')
    grad_num = sp.lambdify(vars_symbols, grad_expr, 'numpy')
    hessian_num = sp.lambdify(vars_symbols, hessian_expr, 'numpy')

    def f(x): return float(f_num(*x))
    def grad(x): return np.array(grad_num(*x), dtype=float).flatten()
    def hessian(x): return np.array(hessian_num(*x), dtype=float).reshape(num_vars, num_vars)

    try:
        x0 = np.array([float(x.strip()) for x in start_str.split(",")], dtype=float)
    except Exception:
        st.error("Error en punto de partida: Utiliza unicamente numeros separados por comas.")
        st.stop()
        
    if len(x0) != num_vars:
        st.error(f"Inconsistencia de dimensiones: Definiste {num_vars} variables pero ingresaste {len(x0)} coordenadas iniciales.")
        st.stop()
        
except Exception as e:
    st.error(f"Error critico en el motor matematico: {e}")
    st.stop()

# --- NUCLEO DEL ALGORITMO ---
def optimizar(metodo_elegido, x0, max_iter, tol, c1, c2):
    x = x0.copy()
    historial_error = []
    historial_tabla = []
    historial_x = [x0.copy()]
    iteraciones = 0
    criterio = "Maximo de iteraciones alcanzado por limite preventivo."
    d = -grad(x)
    
    for k in range(max_iter):
        g = grad(x)
        error_actual = np.linalg.norm(g)
        historial_error.append(error_actual)
        historial_tabla.append({
            "Iteracion": k + 1, 
            "Punto de Inspeccion (x)": str(np.round(x, 4)), 
            "f(x)": round(f(x), 6), 
            "||Grad f|| (Norma Gradiente)": error_actual
        })
        
        if error_actual < tol:
            criterio = f"Convergencia alcanzada exitosamente (Norma del Gradiente < {tol})."
            break
            
        if metodo_elegido == "Gradiente":
            d = -g
        elif metodo_elegido == "Gradiente Conjugado" and k > 0:
            g_anterior = grad(historial_x[-2])
            beta = np.dot(g, g) / (np.dot(g_anterior, g_anterior) + 1e-10)
            d = -g + beta * d
        elif metodo_elegido == "Newton":
            H = hessian(x)
            try:
                H = (H + H.T) / 2.0
                if num_vars > 1:
                    vals = np.linalg.eigvals(H)
                    min_vac = np.min(vals)
                    if min_vac <= 1e-5:
                        H +=
