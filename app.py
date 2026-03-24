import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN DE DATOS ---
df_seguimiento = pd.read_pickle('datos_tablero.pkl')

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
    html.P("Datos al 22 de marzo de 2026", style={'textAlign': 'center', 'color': '#666', 'marginBottom': '30px'}),

    # SELECTORES
    html.Div(style={'display': 'flex', 'gap': '30px', 'marginBottom': '30px', 'maxWidth': '1200px', 'margin': '0 auto 30px auto'}, children=[
        html.Div([
            html.Label("Selecciona un sector", style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#444'}),
            dcc.Dropdown(id='filter-Sector', options=[{'label': i, 'value': i} for i in sorted(df_seguimiento['Sector'].unique())] if not df_seguimiento.empty else [], multi=True, placeholder="Seleccionar")
        ], style={'flex': '1'}),
        html.Div([
            html.Label("Selecciona una dependencia", style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#444'}),
            dcc.Dropdown(id='filter-Dependencia', options=[{'label': i, 'value': i} for i in sorted(df_seguimiento['Dependencia'].unique())] if not df_seguimiento.empty else [], multi=True, placeholder="Seleccionar")
        ], style={'flex': '1'})
    ]),

    # FILA DE TARJETAS (KPIs)
    html.Div(id='kpi-row', style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '40px', 'maxWidth': '1400px', 'margin': '0 auto'}),

    # SECCIÓN SANKEY
    html.Div(style={'maxWidth': '1200px', 'margin': '40px auto', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '12px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-end', 'marginBottom': '10px'}, children=[
            html.Div([
                html.H2("Digitalización de trámites federales", style={'color': '#000', 'fontSize': '24px', 'margin': '0'}),
                html.P("Periodo: Enero - Marzo 2026", style={'color': '#999', 'fontSize': '14px', 'margin': '5px 0 0 0'}),
            ]),
            html.Div(style={'textAlign': 'right', 'fontSize': '12px', 'color': '#666'}, children=[
                html.Span("Dependencia: "), html.Strong("Todas", style={'backgroundColor': '#8fa19e', 'color': 'white', 'padding': '2px 10px', 'borderRadius': '10px'}),
                html.Br(),
                html.Span("Sector: ", style={'marginTop': '5px', 'display': 'inline-block'}), html.Strong("Todos", style={'backgroundColor': '#8fa19e', 'color': 'white', 'padding': '2px 10px', 'borderRadius': '10px'}),
            ])
        ]),
        dcc.Graph(id='sankey-principal', config={'displayModeBar': False})
    ]),

    # SECCIÓN: SOLUCIONES POR RESPONSABLE
    html.Div(style={'maxWidth': '1350px', 'margin': '40px auto', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '12px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '20px'}, children=[
            html.H2("Solución tecnológica por responsable", style={'color': '#1a3e35', 'fontSize': '24px', 'margin': '0', 'fontWeight': 'bold'}),
            html.Div(style={'fontSize': '12px'}, children=[
                html.Span("Sector: "), html.Strong("Todas", style={'backgroundColor': '#1a3e35', 'color': 'white', 'padding': '2px 10px', 'borderRadius': '10px', 'marginRight': '10px'}),
                html.Span("Dependencias: "), html.Strong("Todas", style={'backgroundColor': '#1a3e35', 'color': 'white', 'padding': '2px 10px', 'borderRadius': '10px'})
            ])
        ]),
        html.Div(id='mini-cards-soluciones', style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '10px', 'marginBottom': '20px'}),
        dcc.Graph(id='barras-responsables', config={'displayModeBar': False})
    ]),

    # --- NUEVA SECCIÓN: VOLUMEN DE USO POR AÑO (Agregada al final) ---
    html.Div(style={'maxWidth': '1350px', 'margin': '40px auto', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '12px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start', 'marginBottom': '20px'}, children=[
            html.Div([
                html.H2("Volumen de uso por año", style={'color': '#000', 'fontSize': '28px', 'margin': '0', 'fontWeight': 'bold'}),
                html.P("Actos por año", style={'color': '#444', 'fontSize': '18px', 'margin': '5px 0'}),
                html.P("Actualización: Marzo 2026", style={'color': '#999', 'fontSize': '14px', 'fontStyle': 'italic'}),
            ]),
            html.Div(style={'textAlign': 'right', 'fontSize': '13px', 'color': '#666'}, children=[
                html.Span("Dependencias: "), html.Strong("Todas", style={'backgroundColor': '#7a8c89', 'color': 'white', 'padding': '4px 12px', 'borderRadius': '15px'}),
                html.Br(),
                html.Span("Sector: ", style={'marginTop': '10px', 'display': 'inline-block'}), html.Strong("Todos", style={'backgroundColor': '#7a8c89', 'color': 'white', 'padding': '4px 12px', 'borderRadius': '15px'}),
            ])
        ]),
        html.Div(id='cards-años-uso', style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginBottom': '30px'}),
        dcc.Graph(id='grafica-uso-lineas', config={'displayModeBar': False})
    ])
])

# --- 4. CALLBACK ---
@app.callback(
    [Output('kpi-row', 'children'),
     Output('sankey-principal', 'figure'),
     Output('mini-cards-soluciones', 'children'),
     Output('barras-responsables', 'figure'),
     Output('cards-años-uso', 'children'),
     Output('grafica-uso-lineas', 'figure')],
    [Input('filter-Sector', 'value'),
     Input('filter-Dependencia', 'value')]
)
def update_dashboard(sector, dependencia):
    if df_seguimiento.empty: return [], go.Figure(), [], go.Figure(), [], go.Figure()

    dff = df_seguimiento.copy()
    if sector: dff = dff[dff['Sector'].isin(sector)]
    if dependencia: dff = dff[dff['Dependencia'].isin(dependencia)]

    # --- LÓGICA DE MÉTRICAS ORIGINAL ---
    total_tra = len(dff)
    vol_total = dff['Frecuencia 2024'].sum() or 1
    df_dig = dff[dff['tramite_digitalizados_actualizado_2026'] == 'Digitalizado']
    df_pre = dff[dff['tramite_digitalizados_actualizado_2026'] != 'Digitalizado']
    df_atdt = dff[dff['tramite_digitalizados_atdt'] == 'Digitalizado ATDT']
    df_otras = df_dig[df_dig['tramite_digitalizados_atdt'] != 'Digitalizado ATDT']
    df_punta = df_atdt[df_atdt['tramite_e2e_actualizado_2026'] == 'SI']
    df_hibrido = df_atdt[df_atdt['tramite_e2e_actualizado_2026'] == 'NO']
    df_dig_punta = df_dig[df_dig['tramite_e2e_actualizado_2026'] == 'SI']
    df_dig_hibrido = df_dig[df_dig['tramite_e2e_actualizado_2026'] == 'NO']

    tarjetas = [
        crear_tarjeta_indicador("Total de trámites", total_tra, [
            {'label': 'Digitalizados', 'valor': len(df_dig), 'volumen': df_dig['Frecuencia 2024'].sum(), 'pct_vol': (df_dig['Frecuencia 2024'].sum()/vol_total)*100},
            {'label': 'Presenciales', 'valor': len(df_pre), 'volumen': df_pre['Frecuencia 2024'].sum(), 'pct_vol': (df_pre['Frecuencia 2024'].sum()/vol_total)*100}
        ]),
        crear_tarjeta_indicador("Trámites digitalizados", len(df_dig), [
            {'label': 'Otras dep.', 'valor': len(df_otras), 'volumen': df_otras['Frecuencia 2024'].sum(), 'pct_vol': (df_otras['Frecuencia 2024'].sum()/vol_total)*100},
            {'label': 'ATDT', 'valor': len(df_atdt), 'volumen': df_atdt['Frecuencia 2024'].sum(), 'pct_vol': (df_atdt['Frecuencia 2024'].sum()/vol_total)*100}
        ]),
        crear_tarjeta_indicador("Trámites digitalizados por la ATDT", len(df_atdt), [
            {'label': 'Punta a punta', 'valor': len(df_punta), 'volumen': df_punta['Frecuencia 2024'].sum(), 'pct_vol': (df_punta['Frecuencia 2024'].sum()/vol_total)*100},
            {'label': 'Híbridos', 'valor': len(df_hibrido), 'volumen': df_hibrido['Frecuencia 2024'].sum(), 'pct_vol': (df_hibrido['Frecuencia 2024'].sum()/vol_total)*100}
        ])
    ]

    # --- SANKEY ORIGINAL ---
    label_universo = f"<b>{format_number(total_tra)}</b><br>Trámites"
    label_digitales = f"<b>{format_number(len(df_dig))}</b><br>Digitales · {len(df_dig)/total_tra*100:.0f}%<br><span style='font-size:10px'>{format_number(df_dig['Frecuencia 2024'].sum())} actos · {df_dig['Frecuencia 2024'].sum()/vol_total*100:.1f}% del uso</span>"
    label_presenciales = f"<b>{format_number(len(df_pre))}</b><br>Presenciales · {len(df_pre)/total_tra*100:.0f}%<br><span style='font-size:10px'>{format_number(df_pre['Frecuencia 2024'].sum())} actos · {df_pre['Frecuencia 2024'].sum()/vol_total*100:.1f}% del uso</span>"
    label_punta = f"<b>{format_number(len(df_dig_punta))} Punta a punta</b><br><span style='font-size:10px'>{len(df_dig_punta)/total_tra*100:.1f}% del total<br>{format_number(df_dig_punta['Frecuencia 2024'].sum())} actos</span>"
    label_hibrido = f"<b>{format_number(len(df_dig_hibrido))} Híbridos</b><br><span style='font-size:10px'>{len(df_dig_hibrido)/total_tra*100:.1f}% del total<br>{format_number(df_dig_hibrido['Frecuencia 2024'].sum())} actos</span>"
    label_pre_final = f"<b>{format_number(len(df_pre))} Presenciales</b><br><span style='font-size:10px'>{len(df_pre)/total_tra*100:.1f}% del total<br>{format_number(df_pre['Frecuencia 2024'].sum())} actos</span>"

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

    # --- LÓGICA GRÁFICA DE RESPONSABLES ---
    config_soluciones = [
        {'label': 'Actualización a Sistema (Dependencia)', 'color': '#8cb54a', 'key': 'Dependencia'},
        {'label': 'Actualización a Sistema (FSW)', 'color': '#1a7a6a', 'key': 'FSW'},
        {'label': 'Beca', 'color': '#1e3d59', 'key': 'Beca'},
        {'label': 'Motor Transaccional (FSW)', 'color': '#2d5731', 'key': 'FSW'},
        {'label': 'Nuevo Desarrollo (Dependencia)', 'color': '#7c5532', 'key': 'Dependencia'},
        {'label': 'Nuevo Desarrollo (FSW)', 'color': '#802f4a', 'key': 'FSW'}
    ]

    mini_cards = []
    fig_barras = go.Figure()

    df_fsw_all = dff[dff['Solución tecnológica'].str.contains('FSW|Beca|Motor', case=False, na=False)]
    df_dep_all = dff[dff['Solución tecnológica'].str.contains('Dependencia', case=False, na=False)]

    for conf in config_soluciones:
        sub = dff[dff['Solución tecnológica'] == conf['label']]
        vol_sub = sub['Frecuencia 2024'].sum()
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
            val = len(df_comp[df_comp['Solución tecnológica'] == conf['label']])
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

    # --- LÓGICA DINÁMICA: VOLUMEN POR AÑO ---
    vol_2024 = dff['Frecuencia 2024'].sum()
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
        
        # CAMBIO CLAVE: Usamos None en lugar de 0 si la columna no existe o está vacía
        y_vals = []
        for m in meses_nombres:
            col_name = f"{m} {año}"
            if col_name in dff.columns:
                suma = dff[col_name].sum()
                # Si la suma es 0 o NaN, podrías decidir si poner None
                y_vals.append(suma if suma != 0 else None)
            else:
                y_vals.append(None)
        
        fig_lineas.add_trace(go.Scatter(
            x=meses_labels, 
            y=y_vals, 
            mode='lines+markers', 
            name=año,
            connectgaps=False, # Esto asegura que no se una el último dato con el siguiente si hay huecos
            line=dict(color=info['color'], width=3), 
            marker=dict(size=8),
            hovertemplate=f"Año {año}<br>%{{x}}: %{{y:,.0f}}<extra></extra>"
        ))

    fig_lineas.update_layout(
        plot_bgcolor='white', height=450, margin=dict(l=60, r=40, t=20, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, linecolor='#eee'),
        yaxis=dict(title="Total de Actos", gridcolor='#f0f0f0', zeroline=False)
    )

    return tarjetas, fig_sankey, mini_cards, fig_barras, cards_uso_año, fig_lineas

if __name__ == '__main__':
    app.run_server(debug=True)
