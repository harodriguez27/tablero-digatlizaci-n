import pandas as pd
import numpy as np
import time
import gspread
from gspread.exceptions import APIError
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
import re
from gspread_dataframe import set_with_dataframe
import ssl
import requests
import urllib3
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter
from pathlib import Path
from time import sleep
import psycopg2
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import calendar
import locale
from unidecode import unidecode
import plotly.express as px
import os
import plotly.graph_objects as go
from datetime import date
import random
import dash
from dash import dcc, html, Input, Output 
locale.setlocale(locale.LC_TIME, "es_ES.utf8")

# --- 1. CONFIGURACIÓN Y DATOS (Simulados para el ejemplo) ---
df_seguimiento = pd.read_pickle('datos_tablero.pkl')

# --- 2. FUNCIONES DE APOYO ---
def format_number(n):
    """Formatea números con separador de miles."""
    return f"{int(n):,}"

def calcular_metricas_tarjetas(df_filtrado, df_original):
    base_dep = df_original['Dependencia'].nunique() if 'Dependencia' in df_original.columns else 1
    base_tra = len(df_original)
    base_freq = df_original['Frecuencia 2024'].sum() if 'Frecuencia 2024' in df_original.columns else 1

    def get_stats(sub_df):
        dep = sub_df['Dependencia'].nunique() if 'Dependencia' in sub_df.columns else 0
        tra = len(sub_df)
        freq = sub_df['Frecuencia 2024'].sum() if 'Frecuencia 2024' in sub_df.columns else 0
        return {
            "dep": dep, "tra": tra, "freq": freq,
            "p_dep": (dep / base_dep * 100) if base_dep > 0 else 0,
            "p_tra": (tra / base_tra * 100) if base_tra > 0 else 0,
            "p_freq": (freq / base_freq * 100) if base_freq > 0 else 0
        }

    # Segmentación de DataFrames
    df_digitalizados = df_filtrado[df_filtrado['tramite_digitalizados'] == 'Digitalizado']
    df_atdt = df_filtrado[df_filtrado['tramite_digitalizados_atdt'] == 'Digitalizado ATDT']
    
    # NUEVAS SEGMENTACIONES (Lógica corregida)
    mask_atdt = df_filtrado['tramite_digitalizados'] == 'Digitalizado'
    df_punta_a_punta = df_filtrado[mask_atdt & (df_filtrado['Trámite Digital E2E'] == 'SI')]
    df_solo_entrada = df_filtrado[mask_atdt & (df_filtrado['Trámite Digital E2E'] == 'NO')]

    return {
        "SELECCIÓN": get_stats(df_filtrado),
        "DIGITALIZADOS 2024": get_stats(df_digitalizados),
        "ATDT": get_stats(df_atdt),
        "PUNTA_A_PUNTA": get_stats(df_punta_a_punta),
        "SOLO_ENTRADA": get_stats(df_solo_entrada)
    }

def crear_tarjeta_kpi(titulo, stats, color_header='#1a3e35'):
    filas = [
        ("DEPENDENCIAS", stats['dep'], stats['p_dep']),
        ("TOTAL DE TRÁMITES", stats['tra'], stats['p_tra']),
        ("TOTAL DE ACTOS", stats['freq'], stats['p_freq'])
    ]
    
    contenido_filas = []
    for i, (label, valor, porcentaje) in enumerate(filas):
        contenido_filas.append(
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'margin': '15px 0'}, children=[
                html.Div([
                    html.P(label, style={'margin': '0', 'fontSize': '11px', 'fontWeight': 'bold', 'color': '#555'}),
                    html.H3(format_number(valor), style={'margin': '0', 'fontSize': '20px', 'color': '#1a1a1a'})
                ]),
                html.Div(f"{porcentaje:.1f}%", style={
                    'backgroundColor': '#e8eceb', 'padding': '4px 10px', 'borderRadius': '15px', 
                    'fontSize': '11px', 'fontWeight': 'bold', 'color': '#1a3e35'
                })
            ])
        )
        if i < 2: contenido_filas.append(html.Hr(style={'border': '0', 'borderTop': '1px solid #eee', 'margin': '0'}))

    return html.Div(style={
        'flex': '1', 'backgroundColor': 'white', 'borderRadius': '12px', 
        'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'margin': '0 10px', 'overflow': 'hidden'
    }, children=[
        html.Div(titulo, style={
            'backgroundColor': color_header, 'color': 'white', 'padding': '12px', 
            'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '14px', 'letterSpacing': '1px'
        }),
        html.Div(contenido_filas, style={'padding': '10px 20px'})
    ])

# --- 3. DASHBOARD LAYOUT ---
app = dash.Dash(__name__)
server = app.server
filtros_principales = ['Sector', 'Dependencia']
filtros_sankey = ['ID_tablero', 'Responsable digitalización', 'Solución tecnológica']
todos_los_filtros = filtros_principales + filtros_sankey

app.layout = html.Div(style={'backgroundColor': '#f8faf9', 'padding': '20px', 'fontFamily': 'Segoe UI, Arial'}, children=[
    
    # BARRA SUPERIOR DE FILTROS
    html.Div(style={
        'display': 'flex', 'gap': '20px', 'padding': '20px', 'backgroundColor': '#1a3e35', 
        'borderRadius': '12px', 'marginBottom': '20px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'
    }, children=[
        html.Div([
            html.Label(f"Filtrar {col}", style={'color': 'white', 'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
            dcc.Dropdown(
                id=f'filter-{col}',
                options=[{'label': str(v), 'value': v} for v in df_seguimiento[col].unique()],
                multi=True, placeholder=f"Seleccionar {col}..."
            )
        ], style={'flex': '1'}) for col in filtros_principales
    ]),

    # FILA 1 DE TARJETAS (Métricas Generales)
    html.Div(id='contenedor-tarjetas-1', style={'display': 'flex', 'margin': '0 -10px 20px -10px'}),

    # FILA 2 DE TARJETAS (Métricas E2E vs Entrada)
    html.Div(id='contenedor-tarjetas-2', style={'display': 'flex', 'margin': '0 -10px 20px -10px'}),

    # TREEMAP
    html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '12px', 'marginBottom': '20px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)'}, children=[
        html.H4("Distribución de Trámites por Sector", style={'marginTop': '0', 'color': '#1a3e35'}),
        dcc.Graph(id='treemap-sectores', style={'height': '400px'})
    ]),

    # FILTROS SANKEY
    html.Div(style={
        'display': 'flex', 'gap': '15px', 'padding': '15px', 'backgroundColor': 'white', 
        'borderRadius': '12px', 'marginBottom': '20px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.05)'
    }, children=[
        html.Div([
            html.Label(col, style={'fontSize': '12px', 'fontWeight': 'bold', 'color': '#333'}),
            dcc.Dropdown(
                id=f'filter-{col}',
                options=[{'label': str(v), 'value': v} for v in df_seguimiento[col].unique()],
                multi=True, placeholder="Todos..."
            )
        ], style={'flex': '1'}) for col in filtros_sankey
    ]),

    # GRÁFICO SANKEY
    html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '12px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)'}, children=[
        dcc.Graph(id='sankey-interactivo', style={'height': '650px'})
    ])
])

# --- 4. CALLBACK UNIFICADO ---
@app.callback(
    [Output('contenedor-tarjetas-1', 'children'),
     Output('contenedor-tarjetas-2', 'children'),
     Output('sankey-interactivo', 'figure'),
     Output('treemap-sectores', 'figure')] + 
    [Output(f'filter-{col}', 'options') for col in todos_los_filtros],
    [Input(f'filter-{col}', 'value') for col in todos_los_filtros]
)
def update_dashboard(*filter_values):
    filtros_activos = {todos_los_filtros[i]: val for i, val in enumerate(filter_values) if val}

    # FILTRADO DE DATOS
    dff = df_seguimiento.copy()
    for col, val in filtros_activos.items():
        dff = dff[dff[col].isin(val)]

    # ACTUALIZACIÓN OPCIONES FILTROS EN CASCADA
    nuevas_opciones = []
    for col_objetivo in todos_los_filtros:
        dff_opciones = df_seguimiento.copy()
        for col_filtro, val_filtro in filtros_activos.items():
            if col_filtro != col_objetivo:
                dff_opciones = dff_opciones[dff_opciones[col_filtro].isin(val_filtro)]
        opciones = [{'label': str(v), 'value': v} for v in sorted(dff_opciones[col_objetivo].unique())]
        nuevas_opciones.append(opciones)

    # GENERACIÓN DE MÉTRICAS Y TARJETAS
    stats = calcular_metricas_tarjetas(dff, df_seguimiento)
    
    tarjetas_1 = [
        crear_tarjeta_kpi("TOTAL DE TRÁMITES", stats["SELECCIÓN"]),
        crear_tarjeta_kpi("DIGITALIZADOS 2024", stats["DIGITALIZADOS 2024"]),
        crear_tarjeta_kpi("ATDT", stats["ATDT"]),
    ]
    
    tarjetas_2 = [
        crear_tarjeta_kpi("PUNTA A PUNTA", stats["PUNTA_A_PUNTA"], color_header='#2c5d50'),
        crear_tarjeta_kpi("SOLO ENTRADA", stats["SOLO_ENTRADA"], color_header='#2c5d50'),
        # Espaciador para mantener simetría
        html.Div(style={'flex': '1', 'margin': '0 10px'})
    ]

    # GENERACIÓN DEL TREEMAP
    if dff.empty:
        fig_tree = go.Figure().update_layout(title="Sin datos")
    else:
        df_tree = dff.groupby('Sector').size().reset_index(name='cuenta')
        fig_tree = px.treemap(
            df_tree,
            path=[px.Constant("TODOS LOS SECTORES"), 'Sector'],
            values='cuenta',
            color='cuenta',
            color_continuous_scale=['#d1dbd9', '#2c5d50', '#1a3e35']
        )
        fig_tree.update_layout(margin=dict(t=0, l=0, r=0, b=0), coloraxis_showscale=False)

    # GENERACIÓN DEL SANKEY
    if dff.empty:
        fig_sankey = go.Figure().update_layout(title="Sin datos")
    else:
        all_labels, label_map, node_index = [], {}, 0
        for col in filtros_sankey:
            conteos = dff[col].fillna("Sin información").value_counts()
            for nombre, total in conteos.items():
                if (col, nombre) not in label_map:
                    all_labels.append(f"<b>{nombre}</b><br>{total} trámites")
                    label_map[(col, nombre)] = node_index
                    node_index += 1

        sources, targets, values = [], [], []
        for i in range(len(filtros_sankey) - 1):
            orig, dest = filtros_sankey[i], filtros_sankey[i+1]
            counts = dff.groupby([orig, dest]).size().reset_index(name='count')
            for _, row in counts.iterrows():
                sources.append(label_map[(orig, row[orig])])
                targets.append(label_map[(dest, row[dest])])
                values.append(row['count'])

        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=20, thickness=25, label=all_labels, color="#2c5d50"),
            link=dict(source=sources, target=targets, value=values, color="rgba(44, 93, 80, 0.15)")
        )])
        fig_sankey.update_layout(title_text="Flujo de Digitalización", font_size=11)

    return [tarjetas_1, tarjetas_2, fig_sankey, fig_tree] + nuevas_opciones

if __name__ == '__main__':
    app.run(debug=True)


