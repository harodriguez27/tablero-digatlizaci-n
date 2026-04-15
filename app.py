import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context, no_update
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
import numpy as np
import os 
from datetime import datetime
import psycopg2

# Extraer variables de entorno
db_host = os.environ.get('DB_HOST')
db_port = os.environ.get('DB_PORT', '5432')
db_name = os.environ.get('DB_NAME')
db_user = os.environ.get('DB_USER')
db_pass = os.environ.get('DB_PASS')

query = """
SELECT 
    t.*,
    td.*,
    d.nombre AS dependencia,
    ct.nombre AS tipo_tramite,
    cs.nombre AS sector,
    cd.nombre AS direccion,
    cea.nombre AS estatus_acuerdo,
    ced.nombre AS estatus_digitalizacion,
    st.nombre AS solucion_tecnologica_nombre
FROM tramites t
LEFT JOIN tramites_digitalizacion td 
    ON td.tramite_id = t.id
LEFT JOIN dependencias d 
    ON d.id = t.dependencia_id
LEFT JOIN catalogos ct 
    ON ct.id = t.tipo_tramite_id
LEFT JOIN catalogos cs 
    ON cs.id = td.sector_id
LEFT JOIN catalogos cd 
    ON cd.id = td.direccion_id
LEFT JOIN catalogos cea 
    ON cea.id = td.estatus_acuerdo_id
LEFT JOIN catalogos ced 
    ON ced.id = td.estatus_digitalizacion_id
LEFT JOIN catalogos st 
    ON st.id = td.solucion_tecnologica;
"""
try:
    # Conexión usando las variables de entorno
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_pass
    )
    
    df_seguimiento = pd.read_sql_query(query, conn)
    print("Datos cargados exitosamente.")

except Exception as e:
    print(f"Error al conectar a la base de datos: {e}")

finally:
    # Nos aseguramos de cerrar la conexión si existe
    if 'conn' in locals():
        conn.close()
        print("Conexión cerrada.")

# Carga archivo externo de frecuencias
df_2025_2026 = pd.read_excel("Consolidado_Tramites_2025_2026_Final.xlsx")

df_seguimiento_good = df_seguimiento[df_seguimiento['homoclave_actual'].notna()].copy()
df_seguimiento_sh   = df_seguimiento[df_seguimiento['homoclave_actual'].isna()].copy()
df_seguimiento = pd.merge(df_seguimiento_good, df_2025_2026, on='homoclave_actual', how='left', indicator=True)

df_seguimiento = df_seguimiento.loc[:, ~df_seguimiento.columns.duplicated()]
df_seguimiento_sh = df_seguimiento_sh.loc[:, ~df_seguimiento_sh.columns.duplicated()]
df_seguimiento = pd.concat([df_seguimiento, df_seguimiento_sh], ignore_index=True)

df_seguimiento['digitalizado_actualmente'] = df_seguimiento['digitalizado_actualmente'].map({True: 'Sí'}).fillna('No')
df_seguimiento['digitalizado_atdt'] = df_seguimiento['digitalizado_atdt'].map({True: 'Sí'}).fillna('No')
df_seguimiento['e2e_atdt'] = df_seguimiento['e2e_atdt'].map({True: 'Sí'}).fillna('No')
df_seguimiento['solucion_tecnologica_nombre'] = df_seguimiento['solucion_tecnologica_nombre'].fillna('Sin dato')
df_seguimiento['solucion_tecnologica_nombre'] = np.where(
    df_seguimiento['digitalizado_atdt'] == 'No', 
    'Sin dato', 
    df_seguimiento['solucion_tecnologica_nombre']
)
df_seguimiento['solucion_tecnologica_nombre'] = np.where(
    df_seguimiento['tipo_tramite'] == 'Beca CNBBBJ', 
    'Beca', 
    df_seguimiento['solucion_tecnologica_nombre']
)

columnas_texto = [
    "sector", "dependencia", "homoclave_actual", "nombre", 
    "tipo_tramite", "responsable_estimacion"
]

# Aplicamos la conversión y limpieza en un bucle
for col in columnas_texto:
    if col in df_seguimiento.columns:
        df_seguimiento[col] = df_seguimiento[col].fillna('Sin dato').astype(str).str.strip()

# --- CONFIGURACIÓN DE FECHA ACTUAL ---
meses = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
]
ahora = datetime.now()
fecha_hoy = f"{ahora.day} de {meses[ahora.month - 1]} de {ahora.year}"

# Columnas para la tabla
dff_columnas_mapping = [
    "sector", "dependencia", "homoclave_actual", "nombre", 
    "tipo_tramite", "solucion_tecnologica_nombre", "responsable_estimacion", 
    "frecuencia_2024", "digitalizado_actualmente", 
    "digitalizado_atdt", "e2e_atdt"
]
app = dash.Dash(__name__)
server = app.server

# --- 2. FUNCIONES DE APOYO ---
def format_number(n):
    return f"{int(n):,}"

def crear_tarjeta_indicador(titulo, total_principal, sub_items):
    columnas_desglose = []
    for item in sub_items:
        columnas_desglose.append(
            html.Div(style={'flex': '1', 'padding': '0 10px'}, children=[
                html.P(item['label'], style={'margin': '0', 'fontSize': '12px', 'fontWeight': 'bold', 'color': '#1a3e35'}),
                html.H4(format_number(item['valor']), style={'margin': '2px 0', 'color': '#1a1a1a', 'fontSize': '18px'}),
                html.P("Volumen de uso", style={'margin': '0', 'fontSize': '10px', 'color': '#666'}),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}, children=[
                    html.Span(format_number(item['volumen']), style={'fontSize': '11px', 'fontWeight': '600'}),
                    html.Div(f"{item['pct_vol']:.1f}%", style={
                        'backgroundColor': '#eef2f1', 'padding': '2px 8px', 'borderRadius': '10px',
                        'fontSize': '10px', 'color': '#1a3e35', 'fontWeight': 'bold'
                    })
                ])
            ])
        )

    return html.Div(style={
        'flex': '1', 'backgroundColor': 'white', 'borderRadius': '8px', 'border': '1px solid #d1dbd9',
        'margin': '0 10px', 'padding': '15px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.05)'
    }, children=[
        html.H3(titulo, style={'marginTop': '0', 'fontSize': '13px', 'color': '#1a3e35', 'fontWeight': '800', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
        html.H2(format_number(total_principal), style={'margin': '10px 0', 'fontSize': '28px', 'color': '#1a3e35'}),
        html.Div(style={'display': 'flex'}, children=columnas_desglose)
    ])

# --- 3. LAYOUT DEL TABLERO ---
app.layout = html.Div(style={'backgroundColor': '#fcfcfc', 'padding': '40px', 'fontFamily': 'Arial, sans-serif'}, children=[
    
    html.H1("DIGITALIZACIÓN DE TRÁMITES", style={'textAlign': 'center', 'color': '#1a3e35', 'fontWeight': '900', 'letterSpacing': '1px', 'marginBottom': '5px'}),
    html.P(f"Datos al {fecha_hoy}", style={'textAlign': 'center', 'color': '#666', 'marginBottom': '30px'}),

    # SELECTORES (CON BOTÓN DE LIMPIAR AGREGADO)
    html.Div(style={'display': 'flex', 'gap': '30px', 'marginBottom': '30px', 'maxWidth': '1200px', 'margin': '0 auto 30px auto', 'alignItems': 'flex-end'}, children=[
        html.Div([
            html.Label("Selecciona un sector", style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#444'}),
            dcc.Dropdown(id='filter-Sector', options=[{'label': i, 'value': i} for i in sorted(df_seguimiento['sector'].unique().astype(str))] if not df_seguimiento.empty else [], multi=True, placeholder="Seleccionar")
        ], style={'flex': '1'}),
        html.Div([
            html.Label("Selecciona una dependencia", style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#444'}),
            dcc.Dropdown(id='filter-Dependencia', options=[{'label': i, 'value': i} for i in sorted(df_seguimiento['dependencia'].unique().astype(str))] if not df_seguimiento.empty else [], multi=True, placeholder="Seleccionar")
        ], style={'flex': '1'}),
        html.Div([
            html.Label("Selecciona si está digitalizado", style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#444'}),
            dcc.Dropdown(id='filter-Digitalizado', options=[{'label': i, 'value': i} for i in sorted(df_seguimiento['digitalizado_actualmente'].unique())] if not df_seguimiento.empty else [], multi=True, placeholder="Seleccionar")
        ], style={'flex': '1'}),
        html.Div([
            html.Label("Selecciona si está digitalizado por la ATDT", style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#444'}),
            dcc.Dropdown(id='filter-ATDT', options=[{'label': i, 'value': i} for i in sorted(df_seguimiento['digitalizado_atdt'].unique())] if not df_seguimiento.empty else [], multi=True, placeholder="Seleccionar")
        ], style={'flex': '1'}),
        html.Button("Limpiar Filtros", id='btn-limpiar', n_clicks=0, style={
            'backgroundColor': '#fcfcfc', 'border': '1px solid #1a3e35', 'color': '#1a3e35', 'padding': '8px 15px', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '12px'
        })
    ]),

    # FILA DE TARJETAS (KPIs)
    html.Div(id='kpi-row', style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '40px', 'maxWidth': '1400px', 'margin': '0 auto'}),

    # SECCIÓN SANKEY
    html.Div(style={'maxWidth': '1200px', 'margin': '40px auto', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '12px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-end', 'marginBottom': '10px'}, children=[
            html.Div([
                html.H2("Digitalización de trámites federales", style={'color': '#000', 'fontSize': '24px', 'margin': '0'}),
                html.P("Periodo: Enero 2025 - Abril 2026", style={'color': '#999', 'fontSize': '14px', 'margin': '5px 0 0 0'}),
            ]),
            html.Div(style={'textAlign': 'right', 'fontSize': '12px', 'color': '#666'}, children=[
                html.Span("dependencia: "), html.Strong("Todas", id='txt-dep-sankey', style={'backgroundColor': '#8fa19e', 'color': 'white', 'padding': '2px 10px', 'borderRadius': '10px'}),
                html.Br(),
                html.Span("Sector: ", style={'marginTop': '5px', 'display': 'inline-block'}), html.Strong("Todos", id='txt-sec-sankey', style={'backgroundColor': '#8fa19e', 'color': 'white', 'padding': '2px 10px', 'borderRadius': '10px'}),
            ])
        ]),
        dcc.Graph(id='sankey-principal', config={'displayModeBar': False})
    ]),

    # SECCIÓN: SOLUCIONES POR RESPONSABLE
    html.Div(style={'maxWidth': '1350px', 'margin': '40px auto', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '12px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '20px'}, children=[
            html.H2("Solución tecnológica por responsable", style={'color': '#1a3e35', 'fontSize': '24px', 'margin': '0', 'fontWeight': 'bold'}),
            html.Div(style={'fontSize': '12px'}, children=[
                html.Span("Sector: "), html.Strong("Todas", id='txt-sec-resp', style={'backgroundColor': '#1a3e35', 'color': 'white', 'padding': '2px 10px', 'borderRadius': '10px', 'marginRight': '10px'}),
                html.Span("Dependencias: "), html.Strong("Todas", id='txt-dep-resp', style={'backgroundColor': '#1a3e35', 'color': 'white', 'padding': '2px 10px', 'borderRadius': '10px'})
            ])
        ]),
        html.Div(id='mini-cards-soluciones', style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '10px', 'marginBottom': '20px'}),
        dcc.Graph(id='barras-responsables', config={'displayModeBar': False})
    ]),

    # SECCIÓN: VOLUMEN DE USO POR AÑO
    html.Div(style={'maxWidth': '1350px', 'margin': '40px auto', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '12px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start', 'marginBottom': '20px'}, children=[
            html.Div([
                html.H2("Volumen de uso por año", style={'color': '#000', 'fontSize': '28px', 'margin': '0', 'fontWeight': 'bold'}),
                html.P("Actos por año", style={'color': '#444', 'fontSize': '18px', 'margin': '5px 0'}),
                html.P("Actualización: Abril 2026", style={'color': '#999', 'fontSize': '14px', 'fontStyle': 'italic'}),
            ]),
            html.Div(style={'textAlign': 'right', 'fontSize': '13px', 'color': '#666'}, children=[
                html.Span("Dependencias: "), html.Strong("Todas", id='txt-dep-vol', style={'backgroundColor': '#7a8c89', 'color': 'white', 'padding': '4px 12px', 'borderRadius': '15px'}),
                html.Br(),
                html.Span("Sector: ", style={'marginTop': '10px', 'display': 'inline-block'}), html.Strong("Todos", id='txt-sec-vol', style={'backgroundColor': '#7a8c89', 'color': 'white', 'padding': '4px 12px', 'borderRadius': '15px'}),
            ])
        ]),
        html.Div(id='cards-años-uso', style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginBottom': '30px'}),
        dcc.Graph(id='grafica-uso-lineas', config={'displayModeBar': False})
    ]),

    # SECCIÓN: DISTRIBUCIÓN DE TRÁMITES POR SECTOR (TREEMAP)
    html.Div(style={'maxWidth': '1350px', 'margin': '40px auto', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '12px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'}, children=[
        html.Div(style={'marginBottom': '20px'}, children=[
            html.H2("Distribución de trámites por sector", style={'color': '#000', 'fontSize': '28px', 'margin': '0', 'fontWeight': 'bold'}),
            html.P("Comparativa por cantidad de trámites", style={'color': '#444', 'fontSize': '18px', 'margin': '5px 0'}),
            html.P("Actualización: Abril 2026", style={'color': '#999', 'fontSize': '14px', 'fontStyle': 'italic'}),
        ]),
        dcc.Graph(id='treemap-sectores', config={'displayModeBar': False})
    ]),

    # --- SECCIÓN: TABLA DETALLADA ---
    html.Div(style={'maxWidth': '1350px', 'margin': '40px auto', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '12px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'}, children=[
        html.H2("Detalle de Trámites", style={'color': '#000', 'fontSize': '28px', 'margin': '0 0 20px 0', 'fontWeight': 'bold'}),
        
        html.Div(style={'display': 'flex', 'alignItems': 'flex-end', 'gap': '20px', 'marginBottom': '25px'}, children=[
            html.Div([
                html.Label("Busca por homoclave", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                dcc.Input(id='input-homoclave', type='text', placeholder="Ingresa homoclave", style={'width': '250px', 'padding': '8px', 'marginTop': '5px'})
            ]),
            html.Div([
                html.Label("Estatus de digitalización", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                dcc.Dropdown(id='table-filter-estatus', options=[{'label': i, 'value': i} for i in sorted(df_seguimiento['digitalizado_actualmente'].unique())], placeholder="Seleccionar", style={'width': '250px', 'marginTop': '5px'})
            ]),
            html.Button("Buscar", id='btn-buscar', n_clicks=0, style={'backgroundColor': '#1a4e44', 'color': 'white', 'padding': '10px 25px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
            html.Button("Exportar datos", id='btn-exportar', n_clicks=0, style={'backgroundColor': 'white', 'border': '1px solid #1a4e44', 'color': '#1a4e44', 'padding': '10px 20px', 'borderRadius': '4px', 'cursor': 'pointer'}),
            dcc.Download(id="download-dataframe-excel"),
            html.Div(id='card-conteo-tramites', style={'marginLeft': 'auto', 'padding': '10px', 'backgroundColor': '#f0f4f3', 'borderRadius': '6px', 'borderLeft': '5px solid #1a4e44'})
        ]),

        dash_table.DataTable(
            id='tabla-tramites',
            sort_action="native",
            columns=[{"name": i, "id": i} for i in dff_columnas_mapping],
            page_size=20,
            style_table={'height': '500px', 'overflowY': 'auto'},
            style_header={'backgroundColor': '#1a3e35', 'color': 'white', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left', 'padding': '12px', 'minWidth': '180px', 'whiteSpace': 'normal'}
        )
    ])
])

# --- 4. CALLBACKS ---
# CALLBACK 1: FILTROS EN CASCADA
@app.callback(
    [Output('filter-Sector', 'options'),
     Output('filter-Dependencia', 'options'),
     Output('filter-Digitalizado', 'options'),
     Output('filter-ATDT', 'options')],
    [Input('filter-Sector', 'value'),
     Input('filter-Dependencia', 'value'),
     Input('filter-Digitalizado', 'value'),
     Input('filter-ATDT', 'value')]
)
def update_filter_options(sel_sector, sel_dep, sel_dig, sel_atdt):
    ctx = callback_context
    if not ctx.triggered:
        def get_all(col):
            return [{'label': str(i), 'value': i} for i in sorted(df_seguimiento[col].unique())]
        return get_all('sector'), get_all('dependencia'), get_all('digitalizado_actualmente'), get_all('digitalizado_atdt')

    def get_opts(df, col):
        return [{'label': str(i), 'value': i} for i in sorted(df[col].unique())]

    # Sector
    df_sec = df_seguimiento.copy()
    if sel_dep:  df_sec = df_sec[df_sec['dependencia'].isin(sel_dep)]
    if sel_dig:  df_sec = df_sec[df_sec['digitalizado_actualmente'].isin(sel_dig)]
    if sel_atdt: df_sec = df_sec[df_sec['digitalizado_atdt'].isin(sel_atdt)]
    
    # Dependencia
    df_dep = df_seguimiento.copy()
    if sel_sector: df_dep = df_dep[df_dep['sector'].isin(sel_sector)]
    if sel_dig:    df_dep = df_dep[df_dep['digitalizado_actualmente'].isin(sel_dig)]
    if sel_atdt:   df_dep = df_dep[df_dep['digitalizado_atdt'].isin(sel_atdt)]
    
    # Digitalizado
    df_dig_opts = df_seguimiento.copy()
    if sel_sector: df_dig_opts = df_dig_opts[df_dig_opts['sector'].isin(sel_sector)]
    if sel_dep:    df_dig_opts = df_dig_opts[df_dig_opts['dependencia'].isin(sel_dep)]
    if sel_atdt:   df_dig_opts = df_dig_opts[df_dig_opts['digitalizado_atdt'].isin(sel_atdt)]

    # ATDT
    df_atdt_opts = df_seguimiento.copy()
    if sel_sector: df_atdt_opts = df_atdt_opts[df_atdt_opts['sector'].isin(sel_sector)]
    if sel_dep:    df_atdt_opts = df_atdt_opts[df_atdt_opts['dependencia'].isin(sel_dep)]
    if sel_dig:    df_atdt_opts = df_atdt_opts[df_atdt_opts['digitalizado_actualmente'].isin(sel_dig)]

    return get_opts(df_sec, 'sector'), get_opts(df_dep, 'dependencia'), get_opts(df_dig_opts, 'digitalizado_actualmente'), get_opts(df_atdt_opts, 'digitalizado_atdt')

# CALLBACK 2: LIMPIAR FILTROS
@app.callback(
    [Output('filter-Sector', 'value'),
     Output('filter-Dependencia', 'value'),
     Output('filter-Digitalizado', 'value'),
     Output('filter-ATDT', 'value'),
     Output('input-homoclave', 'value'),
     Output('table-filter-estatus', 'value')],
    [Input('btn-limpiar', 'n_clicks')],
    prevent_initial_call=True
)
def clear_all_filters(n):
    return None, None, None, None, "", None

# CALLBACK 3: EXPORTAR EXCEL
@app.callback(
    Output("download-dataframe-excel", "data"),
    Input("btn-exportar", "n_clicks"),
    State("tabla-tramites", "data"),
    prevent_initial_call=True
)
def export_excel(n_clicks, table_data):
    if not table_data: return no_update
    df_export = pd.DataFrame(table_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Trámites')
    return dcc.send_bytes(output.getvalue(), "Detalle_Tramites.xlsx")

# CALLBACK 4: ACTUALIZACIÓN DEL DASHBOARD (EL PRINCIPAL)
@app.callback(
    [Output('kpi-row', 'children'),
     Output('sankey-principal', 'figure'),
     Output('mini-cards-soluciones', 'children'),
     Output('barras-responsables', 'figure'),
     Output('cards-años-uso', 'children'),
     Output('grafica-uso-lineas', 'figure'),
     Output('treemap-sectores', 'figure'),
     Output('tabla-tramites', 'data'),
     Output('card-conteo-tramites', 'children'),
     Output('txt-dep-sankey', 'children'), Output('txt-sec-sankey', 'children'),
     Output('txt-dep-resp', 'children'), Output('txt-sec-resp', 'children'),
     Output('txt-dep-vol', 'children'), Output('txt-sec-vol', 'children')],
    [Input('filter-Sector', 'value'),
     Input('filter-Dependencia', 'value'),
     Input('filter-Digitalizado', 'value'),
     Input('filter-ATDT', 'value'),
     Input('btn-buscar', 'n_clicks')],
    [State('input-homoclave', 'value'),
     State('table-filter-estatus', 'value')]
)
def update_dashboard(sector, dependencia, digitalizado, atdt, n_clicks, homoclave, estatus):
    if df_seguimiento.empty: 
        return [], go.Figure(), [], go.Figure(), [], go.Figure(), go.Figure(), [], "", "Todas", "Todos", "Todas", "Todas", "Todas", "Todos"

    dff = df_seguimiento.copy()
    
    # --- FILTRADO POR DROPDOWNS PRINCIPALES ---
    if sector: dff = dff[dff['sector'].isin(sector)]
    if dependencia: dff = dff[dff['dependencia'].isin(dependencia)]
    if digitalizado: dff = dff[dff['digitalizado_actualmente'].isin(digitalizado)]
    if atdt: dff = dff[dff['digitalizado_atdt'].isin(atdt)]

    # --- LÓGICA DE MÉTRICAS ORIGINAL ---
    total_tra = len(dff)
    vol_total = dff['frecuencia_2024'].sum() or 1
    df_dig = dff[dff['digitalizado_actualmente'] == 'Sí']
    df_pre = dff[dff['digitalizado_actualmente'] != 'Sí']
    df_atdt = dff[dff['digitalizado_atdt'] == 'Sí']
    df_otras = df_dig[df_dig['digitalizado_atdt'] != 'Sí']
    df_punta = df_atdt[df_atdt['e2e_atdt'] == 'Sí']
    df_hibrido = df_atdt[df_atdt['e2e_atdt'] == 'No']
    df_dig_punta = df_dig[df_dig['e2e_atdt'] == 'Sí']
    df_dig_hibrido = df_dig[df_dig['e2e_atdt'] == 'No']

    tarjetas = [
        crear_tarjeta_indicador("Total de trámites", total_tra, [
            {'label': 'Digitalizados', 'valor': len(df_dig), 'volumen': df_dig['frecuencia_2024'].sum(), 'pct_vol': (df_dig['frecuencia_2024'].sum()/vol_total)*100},
            {'label': 'Presenciales', 'valor': len(df_pre), 'volumen': df_pre['frecuencia_2024'].sum(), 'pct_vol': (df_pre['frecuencia_2024'].sum()/vol_total)*100}
        ]),
        crear_tarjeta_indicador("Trámites digitalizados", len(df_dig), [
            {'label': 'Otras dep.', 'valor': len(df_otras), 'volumen': df_otras['frecuencia_2024'].sum(), 'pct_vol': (df_otras['frecuencia_2024'].sum()/vol_total)*100},
            {'label': 'ATDT', 'valor': len(df_atdt), 'volumen': df_atdt['frecuencia_2024'].sum(), 'pct_vol': (df_atdt['frecuencia_2024'].sum()/vol_total)*100}
        ]),
        crear_tarjeta_indicador("Trámites digitalizados por la ATDT", len(df_atdt), [
            {'label': 'Punta a punta', 'valor': len(df_punta), 'volumen': df_punta['frecuencia_2024'].sum(), 'pct_vol': (df_punta['frecuencia_2024'].sum()/vol_total)*100},
            {'label': 'Híbridos', 'valor': len(df_hibrido), 'volumen': df_hibrido['frecuencia_2024'].sum(), 'pct_vol': (df_hibrido['frecuencia_2024'].sum()/vol_total)*100}
        ])
    ]

    # --- SANKEY ORIGINAL ---
    label_universo = f"<b>{format_number(total_tra)}</b><br>Trámites"
    label_digitales = f"<b>{format_number(len(df_dig))}</b><br>Digitales · {len(df_dig)/total_tra*100:.0f}%<br><span style='font-size:10px'>{format_number(df_dig['frecuencia_2024'].sum())} actos · {df_dig['frecuencia_2024'].sum()/vol_total*100:.1f}% del uso</span>"
    label_presenciales = f"<b>{format_number(len(df_pre))}</b><br>Presenciales · {len(df_pre)/total_tra*100:.0f}%<br><span style='font-size:10px'>{format_number(df_pre['frecuencia_2024'].sum())} actos · {df_pre['frecuencia_2024'].sum()/vol_total*100:.1f}% del uso</span>"
    label_punta = f"<b>{format_number(len(df_dig_punta))} Punta a punta</b><br><span style='font-size:10px'>{len(df_dig_punta)/total_tra*100:.1f}% del total<br>{format_number(df_dig_punta['frecuencia_2024'].sum())} actos</span>"
    label_hibrido = f"<b>{format_number(len(df_dig_hibrido))} Híbridos</b><br><span style='font-size:10px'>{len(df_dig_hibrido)/total_tra*100:.1f}% del total<br>{format_number(df_dig_hibrido['frecuencia_2024'].sum())} actos</span>"
    label_pre_final = f"<b>{format_number(len(df_pre))} Presenciales</b><br><span style='font-size:10px'>{len(df_pre)/total_tra*100:.1f}% del total<br>{format_number(df_pre['frecuencia_2024'].sum())} actos</span>"

    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement = "snap",
        node = dict(pad = 50, thickness = 15, line = dict(color = "white", width = 0),
            label = [label_universo, label_digitales, label_presenciales, label_punta, label_hibrido, label_pre_final],
            color = ["#1a3e35", "#386611", "#b2c4c9", "#9de296", "#f2e085", "#b2c4c9"]
        ),
        link = dict(source = [0, 0, 1, 1, 2], target = [1, 2, 3, 4, 5],
            value = [len(df_dig), len(df_pre), len(df_dig_punta), len(df_dig_hibrido), len(df_pre)],
            color = ["rgba(209, 225, 200, 0.4)", "rgba(220, 230, 235, 0.4)", "rgba(200, 230, 200, 0.3)", "rgba(230, 230, 200, 0.3)", "rgba(220, 230, 235, 0.3)"]
        )
    )])
    fig_sankey.update_layout(font_size=12, height=450, margin=dict(l=0, r=10, t=60, b=20),
        annotations=[
            dict(x=0, y=1.15, showarrow=False, text="<b>Universo</b><br>de trámites", xanchor='left', font=dict(size=12, color="#444")),
            dict(x=0.5, y=1.15, showarrow=False, text="<b>Modalidad</b><br>de trámite", xanchor='center', font=dict(size=12, color="#444")),
            dict(x=1, y=1.15, showarrow=False, text="<b>Tipo de trámite</b><br>Digitalizado", xanchor='right', font=dict(size=12, color="#444"))
        ])

    # --- LÓGICA GRÁFICA DE RESPONSABLES ORIGINAL ---
    config_soluciones = [
        {'label': 'Actualización a Sistema (Dependencia)', 'color': '#8cb54a', 'key': 'dependencia'},
        {'label': 'Actualización a Sistema (FSW)', 'color': '#1a7a6a', 'key': 'FSW'},
        {'label': 'Beca', 'color': '#1e3d59', 'key': 'Beca'},
        {'label': 'Motor Transaccional', 'color': '#2d5731', 'key': 'FSW'},
        {'label': 'Nuevo Desarrollo (Dependencia)', 'color': '#7c5532', 'key': 'dependencia'},
        {'label': 'Nuevo Desarrollo (FSW)', 'color': '#802f4a', 'key': 'FSW'}
    ]

    mini_cards = []
    fig_barras = go.Figure()
    df_fsw_all = dff[dff['solucion_tecnologica_nombre'].str.contains('FSW|Beca|Motor', case=False, na=False)]
    df_dep_all = dff[dff['solucion_tecnologica_nombre'].str.contains('Dependencia', case=False, na=False)]

    for conf in config_soluciones:
        sub = dff[dff['solucion_tecnologica_nombre'] == conf['label']]
        vol_sub = sub['frecuencia_2024'].sum()
        mini_cards.append(html.Div(style={'flex': '1', 'display': 'flex', 'border': '1px solid #ddd', 'borderRadius': '6px', 'overflow': 'hidden', 'height': '75px'}, children=[
            html.Div(style={'backgroundColor': conf['color'], 'width': '60%', 'padding': '10px', 'color': 'white', 'textAlign': 'center'}, children=[
                html.P(conf['label'], style={'margin': '0', 'fontSize': '9px', 'fontWeight': 'bold'}),
                html.H3(len(sub), style={'margin': '5px 0', 'fontSize': '20px'})
            ]),
            html.Div(style={'width': '40%', 'padding': '8px', 'backgroundColor': '#fcfcfc'}, children=[
                html.P("Volumen", style={'margin': '0', 'fontSize': '9px', 'color': '#666'}),
                html.B(format_number(vol_sub), style={'fontSize': '11px'}),
                html.P(f"{(vol_sub/vol_total*100):.1f}%", style={'margin': '0', 'fontSize': '10px', 'color': '#999'})
            ])
        ]))

        for resp in ['Fábrica de SW', 'Dependencia']:
            df_comp = df_fsw_all if resp == 'Fábrica de SW' else df_dep_all
            val = len(df_comp[df_comp['solucion_tecnologica_nombre'] == conf['label']])
            if val > 0:
                fig_barras.add_trace(go.Bar(
                    y=[resp], x=[val], orientation='h', name=conf['label'],
                    marker=dict(color=conf['color']), showlegend=False,
                    hovertemplate=f"{conf['label']}: {val}<extra></extra>"
                ))

    fig_barras.update_layout(
        barmode='stack', height=350, margin=dict(l=150, r=100, t=10, b=40),
        plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False),
        yaxis=dict(categoryorder='array', categoryarray=['Fábrica de SW', 'Dependencia'], tickfont=dict(size=14, color='#333', weight='bold'))
    )

    fig_barras.add_annotation(x=len(df_dep_all), y='Dependencia', text=f"  {len(df_dep_all)}  ", xanchor='left', showarrow=False, bgcolor="#eee", font=dict(size=14, weight='bold'))
    fig_barras.add_annotation(x=len(df_fsw_all), y='Fábrica de SW', text=f"  {len(df_fsw_all)}  ", xanchor='left', showarrow=False, bgcolor="#eee", font=dict(size=14, weight='bold'))

    # --- LÓGICA DINÁMICA: VOLUMEN POR AÑO ORIGINAL ---
    vol_2024 = dff['frecuencia_2024'].sum()
    vol_2025 = dff['TOTAL ANUAL 2025'].sum()
    vol_2026 = dff['TOTAL ANUAL 2026'].sum()
    
    info_años = [
        {'año': '2024', 'valor': vol_2024, 'color': '#8ca39f'},
        {'año': '2025', 'valor': vol_2025, 'color': '#1a4e44'},
        {'año': '2026', 'valor': vol_2026, 'color': '#5a122e'}
    ]
    
    cards_uso_año = []
    for info in info_años:
        cards_uso_año.append(html.Div(style={
            'display': 'flex', 'backgroundColor': 'white', 'borderRadius': '12px', 'overflow': 'hidden',
            'border': '1px solid #eee', 'minWidth': '280px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'
        }, children=[
            html.Div(info['año'], style={
                'backgroundColor': info['color'], 'color': 'white', 'padding': '20px', 
                'fontSize': '24px', 'fontWeight': 'bold', 'display': 'flex', 'alignItems': 'center'
            }),
            html.Div(style={'padding': '10px 15px'}, children=[
                html.P("Volumen de uso anual", style={'margin': '0', 'fontSize': '11px', 'color': '#666'}),
                html.H4(format_number(info['valor']), style={'margin': '5px 0', 'fontSize': '18px', 'color': '#333'})
            ])
        ]))

    meses_nombres = ['TOTAL RECIBIDOS EN EL MES_Enero', 'TOTAL RECIBIDOS EN EL MES_Febrero', 'TOTAL RECIBIDOS EN EL MES_Marzo', 'TOTAL RECIBIDOS EN EL MES_Abril', 'TOTAL RECIBIDOS EN EL MES_Mayo', 'TOTAL RECIBIDOS EN EL MES_Junio', 
                     'TOTAL RECIBIDOS EN EL MES_Julio', 'TOTAL RECIBIDOS EN EL MES_Agosto', 'TOTAL RECIBIDOS EN EL MES_Septiembre', 'TOTAL RECIBIDOS EN EL MES_Octubre', 'TOTAL RECIBIDOS EN EL MES_Noviembre', 'TOTAL RECIBIDOS EN EL MES_Diciembre']
    meses_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    fig_lineas = go.Figure()
    for info in info_años:
        año = info['año']
        y_vals = []
        for m in meses_nombres:
            col_name = f"{m} {año}"
            if col_name in dff.columns:
                suma = dff[col_name].sum()
                y_vals.append(suma if suma != 0 else None)
            else:
                y_vals.append(None)
        
        fig_lineas.add_trace(go.Scatter(
            x=meses_labels, y=y_vals, mode='lines+markers', name=año,
            connectgaps=False, line=dict(color=info['color'], width=3), marker=dict(size=8),
            hovertemplate=f"Año {año}<br>%{{x}}: %{{y:,.0f}}<extra></extra>"
        ))

    fig_lineas.update_layout(
        plot_bgcolor='white', height=450, margin=dict(l=60, r=40, t=20, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, linecolor='#eee'),
        yaxis=dict(title="Total de Actos", gridcolor='#f0f0f0', zeroline=False)
    )

    # --- LÓGICA TREEMAP ORIGINAL ---
    df_tree = dff.groupby('sector').agg({'sector': 'count', 'frecuencia_2024': 'sum'}).rename(columns={'sector': 'count'}).reset_index()
    fig_tree = px.treemap(
        df_tree, path=[px.Constant("Distribución"), 'sector'], values='count', color='count',
        color_continuous_scale=['#b2c4c9', '#3b6e63', '#1a3e35'],
    )
    fig_tree.update_traces(
        textinfo="label+value+percent parent",
        texttemplate="<b>%{label}</b><br>%{value}<br>%{percentParent:.1%}",
        hovertemplate="<b>%{label}</b><br>Cantidad: %{value}<br>Volumen de uso: %{customdata[0]:,}<extra></extra>",
        customdata=df_tree[['frecuencia_2024']],
        marker=dict(line=dict(width=1, color='white'))
    )
    fig_tree.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500, coloraxis_showscale=False)

    # --- LÓGICA DINÁMICA DE LA TABLA ---
    dff_tabla = dff.copy()
    if homoclave:
        dff_tabla = dff_tabla[dff_tabla['homoclave_actual'].str.contains(homoclave, case=False, na=False)]
    if estatus:
        dff_tabla = dff_tabla[dff_tabla['digitalizado_actualmente'] == estatus]

    conteo_texto = html.Div([
        html.P("Resultados encontrados:", style={'margin': '0', 'fontSize': '12px'}),
        html.H4(f"{len(dff_tabla)} trámites", style={'margin': '0', 'color': '#1a4e44'})
    ])

    # Textos dinámicos para las etiquetas
    txt_sec = sector[0] if (sector and len(sector)==1) else "Múltiples" if (sector and len(sector)>1) else "Todos"
    txt_dep = dependencia[0] if (dependencia and len(dependencia)==1) else "Múltiples" if (dependencia and len(dependencia)>1) else "Todas"

    return (tarjetas, fig_sankey, mini_cards, fig_barras, cards_uso_año, fig_lineas, fig_tree, 
            dff_tabla.to_dict('records'), conteo_texto, 
            txt_dep, txt_sec, txt_sec, txt_dep, txt_dep, txt_sec)

if __name__ == '__main__':
    app.run_server(debug=True)
