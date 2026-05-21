import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de página estilo Dashboard Profesional
st.set_page_config(
    page_title="80/20 Portfolio Manager Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS seguro
st.html("<style>.block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;} h1, h2, h3 {margin-bottom: 1rem;} [data-testid='stMetricValue'] {font-size: 24px;}</style>")

# ==============================================================================
# 1. DESCARGA PURA DE DATOS (Caché estricto por Ticker)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def download_raw_financial_data(ticker_symbol):
    """Descarga los datos crudos de Yahoo Finance una sola vez por hora por ticker."""
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        
        # Descarga de históricos de ganancias
        earnings_data = ticker_obj.get_earnings_dates(limit=50)
        if earnings_data is None or earnings_data.empty:
            return None, "No se encontraron fechas de ganancias."
            
        # Descarga de precios históricos (Ticker + Benchmark)
        benchmark = "SPY"
        start_date = "2020-01-01"
        data_raw = yf.download([ticker_symbol, benchmark], start=start_date, auto_adjust=True, progress=False)
        if data_raw.empty:
            return None, f"No se pudieron descargar precios para {ticker_symbol}."
            
        # Info y financieros para el checklist
        info = ticker_obj.info
        fin = ticker_obj.financials.T
        cf = ticker_obj.cashflow.T
        
        payload = {
            "earnings_data": earnings_data,
            "data_raw": data_raw,
            "info": info,
            "fin": fin,
            "cf": cf
        }
        return payload, None
    except Exception as e:
        return None, str(e)

# ==============================================================================
# SIDEBAR / FORMULARIO (Evita ejecuciones en ráfaga)
# ==============================================================================
st.sidebar.header("⚙️ Configuración del Activo")

# Usamos un formulario para que los cambios en los números no disparen ejecuciones intermedias
with st.sidebar.form(key="config_form"):
    ticker_input = st.text_input("Ticker de la Empresa:", value="MSFT").upper().strip()
    st.subheader("Múltiplos P/E manuales")
    input_bajo = st.number_input("Desvío BAJO:", value=21.0, step=1.0)
    input_medio = st.number_input("Desvío MEDIO:", value=30.0, step=1.0)
    input_alto = st.number_input("Desvío ALTO:", value=40.0, step=1.0)
    
    # Botón obligatorio para aplicar cambios
    submit_button = st.form_submit_button(label="🚀 Calcular / Actualizar")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Anti-Bloqueo:** Modificá los múltiplos que quieras y hacé clic en el botón para procesar sin saturar a Yahoo Finance.")

# ==============================================================================
# PROCESAMIENTO Y LÓGICA DE CÁLCULO
# ==============================================================================
st.title(f"📊 80/20 Portfolio Manager Dashboard: {ticker_input}")

if ticker_input:
    # Solicitamos los datos crudos (Usa caché, no toca internet si ya se buscó este ticker)
    with st.spinner("Trayendo datos financieros base de forma segura..."):
        raw_data, download_error = download_raw_financial_data(ticker_input)
        
    if download_error:
        st.error(f"❌ Error de comunicación con Yahoo Finance: {download_error}")
        st.warning("⚠️ Yahoo Finance limitó temporalmente las consultas desde este servidor. Esperá 5-10 minutos y volvé a intentar.")
    elif raw_data is not None:
        
        # --- PROCESAMIENTO MATEMÁTICO LOCAL ---
        try:
            # Reestructurar EPS
            df_eps = raw_data["earnings_data"][['Reported EPS']].dropna().copy()
            df_eps = df_eps.sort_index()
            df_eps.index = df_eps.index.date
            df_eps.index.name = 'Fecha de Reporte'
            df_eps.columns = ['EPS Reportado']
            df_eps = df_eps.reset_index()
            df_eps["Fecha de Reporte"] = pd.to_datetime(df_eps["Fecha de Reporte"]).dt.normalize()
            df_eps["EPS Reportado"] = df_eps["EPS Reportado"].astype(float)
            df_eps = df_eps.sort_values("Fecha de Reporte")
            df_eps["eps_ttm"] = df_eps["EPS Reportado"].rolling(window=4).sum()

            # Aplicar inputs del formulario
            df_eps["desvio_bajo"] = input_bajo
            df_eps["desvio_medio"] = input_medio
            df_eps["desvio_alto"] = input_alto

            df_eps["Precio_bajo"] = df_eps["desvio_bajo"] * df_eps["eps_ttm"]
            df_eps["Precio_medio"] = df_eps["desvio_medio"] * df_eps["eps_ttm"]
            df_eps["Precio_alto"] = df_eps["desvio_alto"] * df_eps["eps_ttm"]

            # Procesar precios locales
            data_raw = raw_data["data_raw"]
            benchmark = "SPY"
            if isinstance(data_raw.columns, pd.MultiIndex):
                prices = data_raw['Close'][ticker_input].to_frame(name='Close')
                spy_prices = data_raw['Close'][benchmark]
            else:
                prices = data_raw[['Close']].copy()
                spy_prices = data_raw['Close']

            # Uniones e interpolación
            df_union = df_eps.set_index("Fecha de Reporte")
            df = prices.join(df_union[['eps_ttm', 'Precio_bajo', 'Precio_medio', 'Precio_alto']])
            cols_interp = ['eps_ttm', 'Precio_bajo', 'Precio_medio', 'Precio_alto']
            df[cols_interp] = df[cols_interp].interpolate(method='linear').ffill().bfill()

            # Indicadores Técnicos
            def rsi_wilder(series, periods=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0))
                loss = (-delta.where(delta < 0, 0))
                avg_gain = gain.rolling(window=periods).mean()
                avg_loss = loss.rolling(window=periods).mean()
                for i in range(periods, len(series)):
                    avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (periods - 1) + gain.iloc[i]) / periods
                    avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (periods - 1) + loss.iloc[i]) / periods
                return 100 - (100 / (1 + (avg_gain / avg_loss)))

            df['RSI'] = rsi_wilder(df['Close'])
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['Disp_200'] = ((df['Close'] - df['EMA200']) / df['EMA200']) * 100
            
            d_avg_pos = df['Disp_200'][df['Disp_200'] > 0].mean()
            d_avg_neg = df['Disp_200'][df['Disp_200'] < 0].mean()
            
            df['RS'] = df['Close'] / spy_prices
            df['RS_SMA'] = df['RS'].rolling(window=50).mean()

            # MÁTICAS CLAVE EN PANTALLA
            current_price = df['Close'].iloc[-1]
            current_pe = current_price / df['eps_ttm'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            current_disp = df['Disp_200'].iloc[-1]
            rs_status = "LIDERANDO" if df['RS'].iloc[-1] > df['RS_SMA'].iloc[-1] else "REZAGADO"
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precio Actual", f"${current_price:.2f}")
            m2.metric("P/E Actual (TTM)", f"{current_pe:.2f}x")
            m3.metric("RSI (14)", f"{current_rsi:.2f}")
            m4.metric("Disp. EMA200", f"{current_disp:.2f}%")
            
            tab_graphs, tab_checklist, tab_cheatsheet = st.tabs(["📈 Gráficos Técnicos", "📋 Checklist Financiero", "📑 Manual de Estrategia"])
            
            # --- TAB 1: GRÁFICOS ---
            with tab_graphs:
                st.subheader("Análisis de Confluencia Profesional")
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14), sharex=True, gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
                fig.patch.set_facecolor('#ffffff')
                
                ax1.plot(df.index, df['Close'], color='black', lw=1.5, label=f'Precio {ticker_input}')
                ax1.plot(df.index, df['Precio_medio'], color='blue', ls='--', alpha=0.6, label='P/E Medio')
                ax1.fill_between(df.index, df['Precio_bajo'], df['Precio_alto'], color='gray', alpha=0.15, label='Zona de Valor')
                ax1.set_ylabel("Precio USD")
                ax1.legend(loc='upper left', fontsize='small').set_zorder(5)
                ax1.grid(True, alpha=0.2)
                
                ax2.plot(df.index, df['RS'], color='navy', lw=1.2, label='RS vs SPY')
                ax2.plot(df.index, df['RS_SMA'], color='orange', ls=':', lw=1.2, label='RS SMA 50')
                ax2.fill_between(df.index, df['RS'], df['RS_SMA'], where=(df['RS'] > df['RS_SMA']), color='green', alpha=0.15)
                ax2.set_ylabel("Fuerza Rel.")
                ax2.legend(loc='upper left', fontsize='x-small')
                ax2.grid(True, alpha=0.2)
                
                ax3.plot(df.index, df['Disp_200'], color='purple', lw=1)
                ax3.axhline(0, color='black', lw=0.8)
                ax3.axhline(d_avg_pos, color='green', ls='--', alpha=0.5)
                ax3.axhline(d_avg_neg, color='red', ls='--', alpha=0.5)
                ax3.fill_between(df.index, df['Disp_200'], 0, where=(df['Disp_200']>=0), color='green', alpha=0.08)
                ax3.fill_between(df.index, df['Disp_200'], 0, where=(df['Disp_200']<0), color='red', alpha=0.08)
                ax3.set_ylabel("% Disp. EMA200")
                ax3.grid(True, alpha=0.2)
                
                ax4.plot(df.index, df['RSI'], color='darkcyan', lw=1)
                ax4.axhline(70, color='red', ls='--', alpha=0.4)
                ax4.axhline(30, color='green', ls='--', alpha=0.4)
                ax4.set_ylim(0, 100)
                ax4.set_ylabel("RSI (14)")
                ax4.grid(True, alpha=0.2)
                
                plt.tight_layout()
                st.pyplot(fig)
                st.info(f"**Status de Fuerza Relativa:** Actualmente el activo se encuentra **{rs_status}** respecto al índice de referencia SPY.")

            # --- TAB 2: CHECKLIST FINANCIERO ---
            with tab_checklist:
                st.subheader("Análisis de Fundamentales (AF)")
                try:
                    info = raw_data["info"]
                    fin = raw_data["fin"]
                    cf = raw_data["cf"]

                    price = info.get('currentPrice', 1)
                    target = info.get('targetMeanPrice', price)
                    upside = ((target - price) / price) * 100 if target else 0

                    rev_now, rev_prev = fin['Total Revenue'].iloc[0], fin['Total Revenue'].iloc[1]
                    net_inc_now, net_inc_prev = fin['Net Income'].iloc[0], fin['Net Income'].iloc[1]
                    fcf_now, fcf_prev = cf['Free Cash Flow'].iloc[0], cf['Free Cash Flow'].iloc[1]

                    ebitda = info.get('ebitda', 1)
                    total_debt = info.get('totalDebt', 0)
                    total_cash = info.get('totalCash', 0)
                    nd_ebitda = (total_debt - total_cash) / ebitda if ebitda > 0 else 0

                    op_margin = info.get('operatingMargins', 0) * 100
                    net_margin = (net_inc_now / rev_now) * 100
                    roe = info.get('returnOnEquity', 0) * 100
                    peg = info.get('pegRatio', 0)

                    checklist_data = [
                        {"Métrica": "Crecimiento Ingresos (YoY)", "Valor": f"{((rev_now/rev_prev)-1)*100:+.2f}%", "Estado": "✅" if rev_now > rev_prev else "❌", "Nota": "Vital para escala"},
                        {"Métrica": "Crecimiento Benef. Neto (YoY)", "Valor": f"{((net_inc_now/net_inc_prev)-1)*100:+.2f}%", "Estado": "✅" if net_inc_now > net_inc_prev else "❌", "Nota": "Eficiencia final"},
                        {"Métrica": "Crecimiento FCF (YoY)", "Valor": f"{((fcf_now/fcf_prev)-1)*100:+.2f}%", "Estado": "✅" if fcf_now > fcf_prev else "❌", "Nota": "Caja real disponible"},
                        {"Métrica": "Margen Neto Actual", "Valor": f"{net_margin:.2f}%", "Estado": "✅" if net_margin > 20 else "🟡", "Nota": "Poder de marca"},
                        {"Métrica": "Net Debt / EBITDA", "Valor": f"{nd_ebitda:.2f}x", "Estado": "✅" if nd_ebitda < 2 else "❌", "Nota": "Nivel de deuda"},
                        {"Métrica": "Op. Margin", "Valor": f"{op_margin:.2f}%", "Estado": "✅" if op_margin > 30 else "🟡", "Nota": "Salud operativa"},
                        {"Métrica": "ROE Actual", "Valor": f"{roe:.2f}%", "Estado": "✅" if roe > 15 else "❌", "Nota": "Retorno capital"},
                        {"Métrica": "PEG Ratio", "Valor": f"{peg:.2f}", "Estado": "✅" if 0 < peg < 1.8 else "🟡", "Nota": "Crecimiento/Precio"},
                        {"Métrica": "Upside Target", "Valor": f"{upside:.2f}%", "Estado": "✅" if upside > 15 else "🟡", "Nota": "Potencial analistas"}
                    ]

                    df_check = pd.DataFrame(checklist_data)
                    st.dataframe(df_check, use_container_width=True, hide_index=True)

                    score = 0
                    if rev_now > rev_prev: score += 15
                    if net_inc_now > net_inc_prev: score += 15
                    if nd_ebitda < 2.5: score += 20
                    if op_margin > 25: score += 15
                    if roe > 15: score += 15
                    if 0 < peg < 2.0: score += 10
                    if upside > 15: score += 10

                    if score >= 85:
                        veredicto = "💎 MUY ATRACTIVO: Activo premium con todo a favor."
                        color_box = "success"
                    elif 65 <= score < 85:
                        veredicto = "✅ ATRACTIVO: Sólido, buscar confirmación técnica."
                        color_box = "info"
                    elif 45 <= score < 65:
                        veredicto = "🟡 NEUTRAL: Ver evolución de márgenes o deuda."
                        color_box = "warning"
                    else:
                        veredicto = "🚨 EVITAR: Fundamentos débiles o riesgo alto."
                        color_box = "error"

                    col_sc, col_ver = st.columns([1, 3])
                    with col_sc:
                        st.metric("SCORE PONDERADO", f"{score} / 100")
                    with col_ver:
                        if color_box == "success": st.success(veredicto)
                        elif color_box == "info": st.info(veredicto)
                        elif color_box == "warning": st.warning(veredicto)
                        else: st.error(veredicto)
                except Exception as ex:
                    st.error(f"Error procesando fundamentales locales: {ex}")

            # --- TAB 3: CHEAT SHEET ---
            with tab_cheatsheet:
                st.markdown("## Protocolo de Decisión 80/20")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("""
                    ### 1. Valoración: Canales de P/E (TTM)
                    * **Bajo el Canal Gris:** Infravaloración Extrema → **Compra Fuerte** (Margen de Seguridad).
                    * **Zona Baja del Canal:** Valor Justo Atractivo → **Acumular**.
                    * **Zona Alta del Canal:** Valor Justo Exigente → **Mantener**.
                    * **Sobre el Canal Gris:** Sobrevaloración → **Venta / Toma de Ganancias**.
                    
                    ### 2. Fuerza Relativa (RS vs SPY)
                    * **RS > Media:** El activo es un **Líder**. Supera al mercado.
                    * **RS < Media:** El activo es un **Rezagado**. El SPY rinde más.
                    """)
                with c2:
                    st.markdown("""
                    ### 3. Estructura: Dispersión EMA 200
                    * **Extremo Negativo:** Pánico. Probabilidad de rebote inminente.
                    * **Promedio Negativo:** Soporte institucional. Buen punto de entrada.
                    * **Cerca de 0%:** Precio en equilibrio. Tendencia sana.
                    * **Extremo Positivo:** Euforia. Peligro de corrección fuerte.
                    
                    ### 4. Momentum: RSI Semanal (Wilder)
                    * **RSI < 30:** Sobreventa. Buscar **Divergencias Alcistas**.
                    * **RSI 40 - 60:** Zona neutral. La tendencia previa manda.
                    * **RSI > 70:** Sobrecompra. Movimiento maduro.
                    """)
        except Exception as e:
            st.error(f"Error general en el cálculo matemático: {e}")
else:
    st.warning("Por favor, ingrese un Ticker válido en el panel izquierdo para comenzar el análisis.")
