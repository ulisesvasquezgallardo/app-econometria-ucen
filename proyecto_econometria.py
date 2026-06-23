import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from datetime import datetime

st.set_page_config(page_title="Análisis de Multicolinealidad", layout="wide")

# ==============================================================================
# INICIALIZACIÓN DE MEMORIA (SESSION STATE)
# ==============================================================================
if 'df_confirmado' not in st.session_state:
    st.session_state.df_confirmado = None
if 'metodo_anterior' not in st.session_state:
    st.session_state.metodo_anterior = "Subir archivo CSV"

# ==============================================================================
# BARRA LATERAL: BRANDING Y MENÚ DE OPCIONES
# ==============================================================================
try:
    st.sidebar.image("Logo_nuevo_ucen.png", use_container_width=True)
except Exception:
    st.sidebar.warning("Logo no encontrado. Asegúrate de que 'Logo_nuevo_ucen.png' esté en la misma carpeta.")

st.sidebar.markdown("---")
st.sidebar.header("Opciones de Entrada de Datos")
metodo = st.sidebar.radio("📥 Elige cómo cargar los datos:", ["Subir archivo CSV", "Ingresar datos manualmente"])

if metodo != st.session_state.metodo_anterior:
    st.session_state.df_confirmado = None
    st.session_state.metodo_anterior = metodo

if metodo == "Subir archivo CSV":
    st.sidebar.info("Ajusta estos parámetros para evitar errores de lectura.")
    separador = st.sidebar.selectbox("¿Qué separa tus columnas?", [";", ",", "\t"])
    decimales = st.sidebar.selectbox("¿Qué separa tus decimales?", [",", "."])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Acerca del Proyecto")
st.sidebar.markdown("**Curso:** Econometría")
st.sidebar.markdown("**Profesor:** Dr. Omar Carrasco Carvajal")
st.sidebar.markdown("**Autores:**")
st.sidebar.markdown("""
* Ulises Vásquez
* Benjamín Cabello
* Michelle Chepo
* Leonardo Novoa
* Nicolas Asperti
""")
st.sidebar.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>Universidad Central de Chile © 2024</div>",
    unsafe_allow_html=True)


# ==============================================================================
# FUNCIÓN PARA IMITAR LA ESTÉTICA Y ESTRUCTURA DE EVIEWS
# ==============================================================================
def generar_tabla_eviews(modelo, dep_var):
    now = datetime.now()
    n_obs = int(modelo.nobs)
    r2 = modelo.rsquared
    adj_r2 = modelo.rsquared_adj
    se_reg = np.sqrt(modelo.scale)
    ssr = modelo.ssr
    log_lik = modelo.llf
    f_stat = modelo.fvalue
    prob_f = modelo.f_pvalue if modelo.f_pvalue is not None else np.nan
    mean_dep = modelo.model.endog.mean()
    sd_dep = modelo.model.endog.std(ddof=1)
    aic = modelo.aic
    bic = modelo.bic
    dw = durbin_watson(modelo.resid)

    coef_html = ""
    for var in modelo.params.index:
        coef = modelo.params[var]
        std_err = modelo.bse[var]
        t_stat = modelo.tvalues[var]
        p_val = modelo.pvalues[var]
        var_name = "C" if var == "const" else var

        coef_html += f"""<tr>
<td style="text-align: left; padding: 2px 0;">{var_name}</td>
<td style="text-align: right; padding: 2px 0;">{coef:.6f}</td>
<td style="text-align: right; padding: 2px 0;">{std_err:.6f}</td>
<td style="text-align: right; padding: 2px 0;">{t_stat:.6f}</td>
<td style="text-align: right; padding: 2px 0;">{p_val:.4f}</td>
</tr>"""

    html = f"""<div style="font-family: Arial, sans-serif; font-size: 13px; background-color: white; padding: 15px; border: 1px solid #d3d3d3; color: black; max-width: 650px;">
<div style="margin-bottom: 10px;">
<b>Dependent Variable:</b> {dep_var}<br>
<b>Method:</b> Least Squares<br>
<b>Date:</b> {now.strftime("%m/%d/%y")} &nbsp;&nbsp; <b>Time:</b> {now.strftime("%H:%M")}<br>
<b>Included observations:</b> {n_obs}<br>
</div>
<table style="width: 100%; border-top: 1.5px solid black; border-bottom: 1.5px solid black; border-collapse: collapse; margin-bottom: 10px;">
<tr>
<th style="text-align: left; border-bottom: 1px solid black; padding: 4px 0;">Variable</th>
<th style="text-align: right; border-bottom: 1px solid black; padding: 4px 0;">Coefficient</th>
<th style="text-align: right; border-bottom: 1px solid black; padding: 4px 0;">Std. Error</th>
<th style="text-align: right; border-bottom: 1px solid black; padding: 4px 0;">t-Statistic</th>
<th style="text-align: right; border-bottom: 1px solid black; padding: 4px 0;">Prob.</th>
</tr>
{coef_html}
</table>
<table style="width: 100%; border-collapse: collapse; font-size: 13px;">
<tr>
<td style="text-align: left; padding: 2px 0;">R-squared</td><td style="text-align: right;">{r2:.6f}</td>
<td style="text-align: left; padding-left: 20px;">Mean dependent var</td><td style="text-align: right;">{mean_dep:.6f}</td>
</tr>
<tr>
<td style="text-align: left; padding: 2px 0;">Adjusted R-squared</td><td style="text-align: right;">{adj_r2:.6f}</td>
<td style="text-align: left; padding-left: 20px;">S.D. dependent var</td><td style="text-align: right;">{sd_dep:.6f}</td>
</tr>
<tr>
<td style="text-align: left; padding: 2px 0;">S.E. of regression</td><td style="text-align: right;">{se_reg:.6f}</td>
<td style="text-align: left; padding-left: 20px;">Akaike info criterion</td><td style="text-align: right;">{aic:.6f}</td>
</tr>
<tr>
<td style="text-align: left; padding: 2px 0;">Sum squared resid</td><td style="text-align: right;">{ssr:.6f}</td>
<td style="text-align: left; padding-left: 20px;">Schwarz criterion</td><td style="text-align: right;">{bic:.6f}</td>
</tr>
<tr>
<td style="text-align: left; padding: 2px 0;">Log likelihood</td><td style="text-align: right;">{log_lik:.6f}</td>
<td style="text-align: left; padding-left: 20px;">Durbin-Watson stat</td><td style="text-align: right;">{dw:.6f}</td>
</tr>
<tr>
<td style="text-align: left; padding: 2px 0;">F-statistic</td><td style="text-align: right;">{f_stat:.6f}</td>
<td></td><td></td>
</tr>
<tr>
<td style="text-align: left; padding: 2px 0;">Prob(F-statistic)</td><td style="text-align: right;">{prob_f:.6f}</td>
<td></td><td></td>
</tr>
</table>
<div style="border-top: 1.5px solid black; margin-top: 5px;"></div>
</div>"""
    return html


# ==============================================================================
# CARGA Y VALIDACIÓN DE DATOS (CSV Y MANUAL)
# ==============================================================================
st.title("Modelo de Regresión y Detección de Multicolinealidad")

if metodo == "Subir archivo CSV":
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=separador, decimal=decimales)
        df = df.apply(pd.to_numeric, errors='coerce')

        nulos_csv = df.isnull().sum().sum()
        if nulos_csv > 0:
            st.warning(
                f"⚠️ Alerta: Se detectaron {nulos_csv} datos faltantes o textos inválidos en el archivo. Esas filas incompletas serán eliminadas del análisis.")

        st.session_state.df_confirmado = df
    else:
        st.session_state.df_confirmado = None

elif metodo == "Ingresar datos manualmente":
    st.write("### 📝 Ingreso Manual de Datos")
    st.info(
        "1. Define los nombres de tus variables abajo.\n2. Rellena los datos en la tabla (puedes agregar filas arrastrando o en la esquina inferior).\n3. Presiona el botón de confirmar.")

    nombres_cols = st.text_input("✏️ Nombres de las columnas (separados por coma):", "Y, X1, X2, X3")
    cols_list = [c.strip() for c in nombres_cols.split(",") if c.strip() != ""]

    if "tabla_manual" not in st.session_state or list(st.session_state.tabla_manual.columns) != cols_list:
        st.session_state.tabla_manual = pd.DataFrame(columns=cols_list, index=range(12))

    df_manual = st.data_editor(st.session_state.tabla_manual, num_rows="dynamic", use_container_width=True)

    if st.button("✅ Confirmar Datos e Iniciar Análisis"):
        df_limpio = df_manual.dropna(how='all')
        if df_limpio.empty:
            st.error("⚠️ La tabla está completamente vacía. Ingresa datos antes de continuar.")
        elif df_limpio.isnull().values.any():
            faltantes = df_limpio.isnull().sum().sum()
            st.error(
                f"⚠️ Error: Hay {faltantes} celdas vacías en las filas utilizadas. Por favor, rellena todos los datos de las observaciones o elimina la fila vacía.")
        elif len(df_limpio) <= len(cols_list):
            st.error(
                f"⚠️ Error: Tienes {len(cols_list)} variables pero solo {len(df_limpio)} observaciones. ¡Los grados de libertad deben ser positivos!")
        else:
            try:
                df_limpio = df_limpio.replace(',', '.', regex=True).astype(float)
                st.session_state.df_confirmado = df_limpio
                st.success("¡Datos guardados con éxito! Desplázate hacia abajo para configurar el modelo.")
            except ValueError:
                st.error(
                    "⚠️ Error: Todos los valores en la tabla deben ser números. Verifica que no hayas introducido letras o caracteres extraños.")

# ==============================================================================
# PROCESAMIENTO Y ANÁLISIS ECONÓMETRICO
# ==============================================================================
df_listo = st.session_state.df_confirmado

if df_listo is not None:
    st.write("---")
    st.write("### Vista previa de los datos listos para el análisis")
    st.dataframe(df_listo.head())

    cols = df_listo.columns.tolist()
    dep_var = st.selectbox("Selecciona la variable dependiente (Y):", cols)
    indep_vars = st.multiselect("Selecciona las variables independientes (X):", [c for c in cols if c != dep_var])

    if dep_var and indep_vars:
        Y = df_listo[dep_var].dropna()
        X = df_listo[indep_vars].loc[Y.index].dropna()
        Y = Y.loc[X.index]
        X_with_const = sm.add_constant(X)

        # 1. Regresión Principal
        st.write("---")
        st.header("1. Resultados de la Regresión Principal")
        model = sm.OLS(Y, X_with_const).fit()
        st.markdown(generar_tabla_eviews(model, dep_var), unsafe_allow_html=True)

        # 2. Análisis de R^2 y valores t
        st.write("---")
        st.header("2. Detección: Relación entre t y R-cuadrado")
        r_squared = model.rsquared
        pvalues = model.pvalues[1:]

        high_r2 = r_squared > 0.80
        non_sig_t = (pvalues > 0.05).sum() > (len(pvalues) / 2)

        multicolinealidad_sintoma = False
        if high_r2 and non_sig_t:
            multicolinealidad_sintoma = True
            st.error(
                f"**Síntoma detectado:** El R-cuadrado es alto ({r_squared:.4f}) y la mayoría de las pruebas t son no significativas (p > 0.05). Esto indicaría la presencia de multicolinealidad severa.")
        else:
            st.success(
                f"**Sin síntomas iniciales:** El modelo presenta un R-cuadrado de {r_squared:.4f} sin evidencia concluyente de multicolinealidad basada únicamente en la prueba t convencional.")

        with st.expander("📖 Ver justificación teórica del síntoma clásico"):
            st.write("""
            **El Síntoma Clásico:** Teóricamente, un $R^2$ alto (por ejemplo, superior a 0.8) indica que el modelo en su conjunto explica muy bien las variaciones de la variable dependiente (Y). 
            Sin embargo, si simultáneamente la mayoría de los estadísticos $t$ son **no significativos** (valor p > 0.05), estamos ante un problema. 

            Esto ocurre porque las variables explicativas están tan correlacionadas entre sí que el modelo MCO (Mínimos Cuadrados Ordinarios) no puede aislar ni separar el efecto individual de cada variable, "confundiendo" la significancia de cada una y elevando sus varianzas.
            """)

        # 3. Matriz de correlación
        if multicolinealidad_sintoma or st.checkbox("Forzar análisis de correlación y regresiones auxiliares"):
            st.write("---")
            st.header("3. Matriz de Correlación entre Regresores")
            corr_matrix = X.corr()
            st.dataframe(corr_matrix)

            alta_corr = False
            for col in corr_matrix.columns:
                for row in corr_matrix.index:
                    if col != row and abs(corr_matrix.loc[row, col]) > 0.8:
                        alta_corr = True

            if alta_corr:
                st.warning(
                    "Se observa una alta correlación (superior a 0.8) entre al menos dos variables explicativas, lo que sugiere un problema severo.")
            else:
                st.info(
                    "No se observa una correlación superior a 0.8 entre las parejas de variables explicativas en la matriz.")

            with st.expander("📖 Ver justificación teórica sobre la Colinealidad vs. Multicolinealidad"):
                st.write("""
                **Colinealidad y la regla empírica del 0.8:**
                La matriz de correlación evalúa la relación lineal de a pares (solo dos variables a la vez). Una correlación superior a $0.8$ en valor absoluto sugiere una colinealidad grave entre esa pareja.

                **Nota:** Aunque una correlación baja en esta matriz es una buena señal, no descarta la multicolinealidad. Podría existir multicolinealidad perfecta o grave sin que ninguna correlación de a pares sea alta, ya que una variable podría ser una combinación lineal de *varias* otras variables conjuntas. Por eso necesitamos regresiones auxiliares.
                """)

            # 4. Regresiones Auxiliares y Test F
            st.write("---")
            st.header("4. Regresiones Auxiliares (Test F) y Regla de Klein")

            n = len(Y)
            k = len(indep_vars) + 1
            alpha = 0.05

            if (k - 2) > 0 and (n - k + 1) > 0:
                f_tabla = stats.f.ppf(1 - alpha, k - 2, n - k + 1)
                st.write(f"**Parámetros de prueba:** n = {n}, k = {k}, Nivel de significancia = {alpha * 100}%")
                st.write(f"**F Tabla:** {f_tabla:.5f}")
                st.write(f"**R-cuadrado Modelo Global:** {r_squared:.5f}")

                f_results = []
                tol_results = []
                problema_fuerte = False
                tol_severo_detectado = False
                klein_severo_detectado = False

                for i, col in enumerate(indep_vars):
                    st.subheader(f"Regresión Auxiliar: {col} contra las restantes Xs")
                    y_aux = X[col]
                    x_aux_cols = [c for c in indep_vars if c != col]
                    x_aux = sm.add_constant(X[x_aux_cols])

                    model_aux = sm.OLS(y_aux, x_aux).fit()
                    st.markdown(generar_tabla_eviews(model_aux, col), unsafe_allow_html=True)

                    r2_aux = model_aux.rsquared

                    if r2_aux < 1.0 and r2_aux > 0:
                        f_calc = (r2_aux / (k - 2)) / ((1 - r2_aux) / (n - k + 1))
                    elif r2_aux >= 1.0:
                        f_calc = float('inf')
                    else:
                        f_calc = 0.0

                    decision_f = "Variable es altamente colineal" if f_calc > f_tabla else "Variable no es colineal"
                    if f_calc > f_tabla:
                        problema_fuerte = True

                    decision_klein = "Problema Severo (R² aux > R² modelo)" if r2_aux > r_squared else "Normal"
                    if r2_aux > r_squared:
                        klein_severo_detectado = True

                    f_results.append({
                        "Estimador": f"R² {col}.restantes",
                        "R² Auxiliar": round(r2_aux, 6),
                        "F Calculado": round(f_calc, 5) if f_calc != float('inf') else "Infinito",
                        "Decisión F": decision_f,
                        "Regla de Klein": decision_klein
                    })

                    tol_i = 1 - r2_aux
                    fiv_i = 1 / tol_i if tol_i > 0 else float('inf')

                    # NUEVA CLASIFICACIÓN DE TOLERANCIA (TOL) - Ajustada a la teoría del PDF
                    if tol_i < 0.10:
                        decision_tol = "Colinealidad SEVERA"
                        tol_severo_detectado = True
                    elif 0.10 <= tol_i <= 0.70:
                        decision_tol = "Variable de Colinealidad Normal"
                    else:
                        decision_tol = "Variable No Esta Relacionada"

                    tol_results.append({
                        "Variable": col,
                        "R-Cuadrado Auxiliar": round(r2_aux, 6),
                        "TOL_i": round(tol_i, 6),
                        "FIV_i": round(fiv_i, 6) if fiv_i != float('inf') else "Infinito",
                        "Decisión": decision_tol
                    })

                st.write("**Resumen de Regresiones Auxiliares:**")
                st.table(pd.DataFrame(f_results).astype(str))

                with st.expander("📖 Ver justificación teórica (Regresiones Auxiliares y Regla de Klein)"):
                    st.write("""
                    **Regresiones Auxiliares (Test F):**
                    Se toma cada variable independiente y se hace 'dependiente' del resto de regresores para detectar colinealidad compleja. 
                    Si el $F$ calculado es mayor que el $F$ crítico de la tabla, se rechaza la hipótesis nula, demostrando que la variable está explicada por el resto del modelo.

                    **Regla Práctica de Klein:**
                    Dice que si el $R^2$ de la regresión auxiliar es mayor que el $R^2$ del modelo original global, la multicolinealidad es un problema severo.
                    """)

                # 5. Test TOL
                st.write("---")
                st.header("5. Índice de Tolerancia (TOL) e Inflación de Varianza (FIV)")
                st.table(pd.DataFrame(tol_results).astype(str))

                with st.expander("📖 Ver justificación teórica sobre TOL y FIV"):
                    st.write("""
                    **TOL y FIV:**
                    * **TOL (Índice de Tolerancia):** Mide qué proporción de la varianza de una variable explicativa NO está influenciada o explicada por las demás variables ($TOL = 1 - R^2_{auxiliar}$).
                    * **FIV (Factor de Inflación de Varianza):** Es el inverso del TOL ($FIV = 1 / TOL$). Muestra cuántas veces se infla la varianza del coeficiente estimado.

                    **Escala de Decisión Econométrica:**
                    * **TOL < 0.10 (FIV > 10):** Colinealidad SEVERA. La variable está críticamente amarrada a las demás.
                    * **0.10 ≤ TOL ≤ 0.70:** Colinealidad Normal. Escenario común y tolerable.
                    * **TOL > 0.70:** Variable No Está Relacionada.
                    """)

                # ==============================================================================
                # 6. Conclusión Final
                # ==============================================================================
                st.write("---")
                st.header("Conclusión Final del Análisis")

                # Caso 1: El problema es realmente grave (Síntoma clásico activo, TOL crítico, o Regla de Klein rota)
                if tol_severo_detectado or multicolinealidad_sintoma or klein_severo_detectado:
                    st.error(
                        "❌ Conclusión: De acuerdo a las pruebas desarrolladas, el modelo analizado PRESENTA PROBLEMAS DE MULTICOLINEALIDAD SEVERA O FUERTE. "
                        "Los estimadores individuales son inestables y los errores estándar están inflados artificialmente.")

                # Caso 2: Hay colinealidad estadísticamente demostrada por el Test F, pero su magnitud es inofensiva
                elif problema_fuerte:
                    st.warning(
                        "⚠️ Conclusión: Se detectó colinealidad estadísticamente significativa mediante las regresiones auxiliares (Test F). "
                        "Sin embargo, de acuerdo a los índices TOL/FIV y la Regla de Klein, la magnitud del problema es NORMAL. "
                        "El modelo es utilizable y las estimaciones son estables.")

                # Caso 3: No hay absolutamente nada de colinealidad
                else:
                    st.success(
                        "✅ Conclusión: De acuerdo a las pruebas desarrolladas, el modelo analizado está SIN PROBLEMAS de multicolinealidad. "
                        "Las variables explicativas aportan información independiente al modelo.")

                with st.expander("⚠️ Consecuencias prácticas y posibles soluciones"):
                    st.write("""
                    **¿Qué pasa si el modelo tiene multicolinealidad severa?**
                    Aunque los estimadores MCO siguen siendo Insesgados, el problema principal radica en sus varianzas:
                    1.  **Varianzas infladas:** Los intervalos de confianza se vuelven demasiado amplios.
                    2.  **Pérdida de significancia:** Los estadísticos $t$ disminuyen, por lo que es probable que consideres incorrectamente que una variable "no importa" cuando sí lo hace.
                    3.  **Inestabilidad:** Cambiar o quitar una sola observación de tus datos, o agregar una variable más, hará que los coeficientes den saltos drásticos.

                    **Posibles soluciones sugeridas:**
                    * **Aumentar el tamaño de la muestra:** Traer más datos aporta más variabilidad y puede romper el efecto de colinealidad.
                    * **Eliminar variables:** Quitar la variable más colineal (¡Cuidado! Esto podría introducir sesgo por variable omitida).
                    * **Transformar las variables:** Utilizar ratios o primeras diferencias.
                    """)
            else:
                st.error(
                    "⚠️ No hay suficientes grados de libertad para calcular el Test F de las regresiones auxiliares. Por favor, asegúrate de tener más observaciones que variables.")
