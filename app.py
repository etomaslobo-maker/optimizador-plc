import streamlit as st
import numpy as np
from scipy.optimize import linprog
import plotly.graph_objects as go

st.set_page_config(page_title="Optimizador de Programación Lineal", layout="wide")

st.title("📈 Optimizador de Programación Lineal Continua (PLC)")
st.markdown("""
Esta aplicación permite resolver modelos de Programación Lineal, analizar la saturación de sus restricciones 
y, en el caso de 2 variables, visualizar la región factible gráficamente.
""")

st.sidebar.header("⚙️ Configuración del Modelo")

# 1. Configuración de dimensiones
opt_type = st.sidebar.selectbox("Objetivo", ["Maximizar", "Minimizar"])
num_vars = st.sidebar.number_input("Número de variables", min_value=2, max_value=10, value=2)
num_cons = st.sidebar.number_input("Número de restricciones", min_value=1, max_value=20, value=3)

# 2. Entrada de la Función Objetivo
st.subheader("🎯 Función Objetivo")
cols_obj = st.columns(num_vars)
c = np.zeros(num_vars)
for i in range(num_vars):
    c[i] = cols_obj[i].number_input(f"Coeficiente X{i+1}", value=1.0, step=1.0, key=f"obj_{i}")

# 3. Entrada de Restricciones
st.subheader("🚧 Restricciones")
A = np.zeros((num_cons, num_vars))
b = np.zeros(num_cons)
senses = []

for i in range(num_cons):
    st.write(f"**Restricción {i+1}**")
    cols_cons = st.columns(num_vars + 2)
    for j in range(num_vars):
        A[i, j] = cols_cons[j].number_input(f"X{j+1} (R{i+1})", value=1.0, step=1.0, key=f"A_{i}_{j}")
    
    sense = cols_cons[num_vars].selectbox("Signo", ["<=", ">=", "=="], key=f"sense_{i}")
    senses.append(sense)
    
    b[i] = cols_cons[num_vars+1].number_input("Término indep.", value=10.0, step=1.0, key=f"b_{i}")

# 4. Botón de Resolución
if st.button("🚀 Resolver Modelo", type="primary"):
    
    # Preparar datos para scipy.linprog (que por defecto minimiza)
    c_opt = c if opt_type == "Minimizar" else -c
    
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    for i in range(num_cons):
        if senses[i] == "<=":
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif senses[i] == ">=":
            A_ub.append(-A[i]) # Invertir signo para scipy
            b_ub.append(-b[i])
        else:
            A_eq.append(A[i])
            b_eq.append(b[i])
            
    # Formatear a numpy arrays o None si están vacíos
    A_ub = np.array(A_ub) if len(A_ub) > 0 else None
    b_ub = np.array(b_ub) if len(b_ub) > 0 else None
    A_eq = np.array(A_eq) if len(A_eq) > 0 else None
    b_eq = np.array(b_eq) if len(b_eq) > 0 else None
    bounds = [(0, None) for _ in range(num_vars)] # Variables no negativas por defecto

    # Resolver
    res = linprog(c_opt, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        st.success("✅ ¡Solución Óptima Encontrada!")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("### 🌟 Resultados")
            st.metric(label="Valor de la Función Objetivo (Z)", value=round(res.fun if opt_type == "Minimizar" else -res.fun, 4))
            for idx, val in enumerate(res.x):
                st.write(f"**X{idx+1}:** {round(val, 4)}")
                
        with col_res2:
            st.markdown("### 📊 Análisis de Saturación")
            for i in range(num_cons):
                # Calcular el valor del lado izquierdo de la restricción evaluado en el punto óptimo
                lhs_val = np.dot(A[i], res.x)
                diferencia = round(abs(b[i] - lhs_val), 4)
                estado = "🔴 Saturada (Activa)" if diferencia == 0 else f"🟢 No Saturada (Holgura/Exceso: {diferencia})"
                st.write(f"**Restricción {i+1}:** LHS = {round(lhs_val, 4)} | RHS = {b[i]} ➔ {estado}")
        
        # 5. Gráfico para 2 variables
        if num_vars == 2:
            st.markdown("---")
            st.markdown("### 📉 Resolución Gráfica Interactiva")
            
            fig = go.Figure()
            x_vals = np.linspace(0, max(res.x[0]*2, 20), 400)
            
            # Dibujar restricciones
            for i in range(num_cons):
                if A[i, 1] != 0:
                    y_vals = (b[i] - A[i, 0] * x_vals) / A[i, 1]
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'R{i+1} ({senses[i]} {b[i]})'))
                else:
                    x_line = b[i] / A[i, 0]
                    fig.add_vline(x=x_line, line_dash="dash", annotation_text=f'R{i+1}')

            # Dibujar punto óptimo
            fig.add_trace(go.Scatter(x=[res.x[0]], y=[res.x[1]], mode='markers+text', 
                                     marker=dict(color='red', size=12, symbol='star'),
                                     text=['Óptimo'], textposition='top right', name='Solución Óptima'))
            
            fig.update_layout(xaxis_title="X1", yaxis_title="X2", 
                              xaxis=dict(range=[0, max(res.x[0]*1.5, 10)]), 
                              yaxis=dict(range=[0, max(res.x[1]*1.5, 10)]),
                              hovermode="closest", height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Interacciona con el gráfico: puedes hacer zoom, desplazar los ejes y pasar el ratón por encima de las líneas para ver los valores exactos.")

    else:
        st.error(f"❌ El modelo no se pudo resolver. Estado: {res.message}")
