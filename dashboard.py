#from utils import *
import pandas as pd
import numpy as np
import streamlit as st
import subprocess
from PIL import Image

from causalimpact import CausalImpact
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
import json
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from causalimpact import CausalImpact
from dtaidistance import dtw

st.set_option('deprecation.showPyplotGlobalUse', False)

COLOR_MAP = {"default": "#262730", "pink": "#E22A5B"}
red = "#E07182"
green = "#96D6B4"
blue = "#4487D3"
grey = "#87878A"

tfd = tfp.distributions


def main():
    max_width_(width=1200)
    
    image = Image.open('causal_impact_explainer_logo.png')
    st.sidebar.image(image, caption='', use_column_width=True)
    st.title("Mega sequía: midiendo el impacto económico :volcano: :earth_americas:")
    texto("""Esta aplicación ayuda a explorar los resultados de la librería Causal Impact para medir el impacto económico generado por la mega sequía que ocurre en Chile. Estos resultados son generados usando datos del banco central, donde usamos como control series económicas no afectadas por la sequía que ayudan a reconstruir un escenario contrafactual donde respondemos: qué hubiese pasado en la economía si no hubiese habido mega sequía.""",
   nfont=17)
    disclaimer()

    link_libreria()
    st.markdown('## Los datos provienen del  Producto Interno Bruto de Chile entre los años 2013-19')
    
    st.markdown('### Choose Causal Impact parameters')

    chosen_df = load_dataframe()
    col1, col2 = st.beta_columns(2)
    with col1:
        time_var = st.selectbox("Choose the time variable",
                               chosen_df.columns,
                               index=0,
                               key='time_variable') #date
        alpha = st.number_input("Significance level", 0.01, 0.5, value=0.05, 
                                    step=0.01, 
                                    key='significance_level')


    with col2:
        y_var = st.selectbox("Choose the outcome variable (y)",
                             chosen_df.columns,
                             index=2,
                             key='analysis_variable')
        
        df_experiment = chosen_df.copy()
        df_experiment[time_var] = df_experiment[time_var].apply(pd.to_datetime)
        df_experiment.sort_values(time_var, inplace=True)
        df_experiment.index = range(len(df_experiment))
            
        min_date = df_experiment[time_var].min().date()
        last_date = df_experiment[time_var].max().date()
        mid_point = int(len(df_experiment) / 2)
        intervention_time = st.slider('Fecha del inicio de la sequía', 
                                              min_date, 
                                              last_date, 
                                              value=df_experiment.loc[mid_point + 20, time_var].date(),
                                              key='training_period')
        
        
    beg_pre_period, end_pre_period = min_date, intervention_time
    beg_eval_period, end_eval_period = intervention_time, last_date    
    beg_eval_period = beg_eval_period + pd.DateOffset(months=1)


    st.sidebar.markdown("#### Select the control variables")
    x_vars = sorted(list([col for col in chosen_df.columns if col != y_var and col != time_var and col!='group']),  key=len)
    selected_x_vars = st.sidebar.multiselect("Las variables de control no deben haber sido afectadas por la sequía", x_vars,
                        default=x_vars)


    
    strftime_format="%Y-%m-%d"
    parameters = {
                  "alpha": alpha, 
                  "beg_pre_period": beg_pre_period.strftime(strftime_format),
                  "end_pre_period": end_pre_period.strftime(strftime_format),
                  "beg_eval_period": beg_eval_period.strftime(strftime_format),
                  "end_eval_period": end_eval_period.strftime(strftime_format),
                  "selected_x_vars": selected_x_vars,
                  "y_var": y_var,
                  "time_var": time_var,
                 }

    with st.beta_expander('Show dataframe'):
        st.write(df_experiment.head(5))
        
    with st.beta_expander('Ploting variables'):
        vars_to_plot = st.multiselect("Variables to plot",
                                  list(df_experiment.columns),
                                  default=y_var)

        plot_vars(df_experiment, vars_to_plot, time_var, 
                  beg_pre_period=parameters['beg_pre_period'],
                  end_pre_period=parameters['end_pre_period'],
                  beg_eval_period=parameters['beg_eval_period'], 
                  end_eval_period=parameters['end_eval_period'],
                  )

        col_mutual_info, col_precip, col_tdw = st.beta_columns(3)
        with col_mutual_info:
            find_mutual_info(df_experiment, time_var, y_var, parameters['end_pre_period'])            
        with col_tdw:
            find_dynamic_time_warp(df_experiment, time_var, y_var, parameters['end_pre_period'])
        with col_precip:
            find_mutual_info_precipitaciones(df_experiment, time_var, y_var, parameters['end_pre_period'])

        
        texto(' ')

    with st.beta_expander("Estimate Causal Impact model"):
        
        beg_pre_period = parameters['beg_pre_period'].split('-')[0] + parameters['beg_pre_period'].split('-')[1] + '01'
        end_pre_period = parameters['end_pre_period'].split('-')[0] + parameters['end_pre_period'].split('-')[1] + '01'
        
        beg_eval_period = parameters['beg_eval_period'].split('-')[0] + parameters['beg_eval_period'].split('-')[1] + '01'
        end_eval_period = parameters['end_eval_period'].split('-')[0] + parameters['end_eval_period'].split('-')[1] + '01'
        
        pre_period = [beg_pre_period, end_pre_period]
        post_period = [beg_eval_period, end_eval_period]

        df_experiment[time_var] = df_experiment[time_var].dt.strftime('%m/%d/%Y')
        df_experiment[y_var] = df_experiment[y_var].astype(float)

        mapa_nombres = {var_x: f'x_{idx}' for idx, var_x in enumerate(parameters['selected_x_vars'])}
        mapa_nombres[y_var]= 'y'
        df_toci = df_experiment[[time_var, y_var] + parameters['selected_x_vars']].set_index(time_var).copy()
        df_toci.rename(columns=mapa_nombres, inplace=True)
        texto('Puede tardar varios segundos', 11)
        if st.checkbox('Run Causal Impact', value=False):
            
            ci = CausalImpact(df_toci, 
                              pre_period, 
                              post_period,
                              alpha=alpha,
                              model_args={'nseasons': 12})
            
            results = ci.inferences.copy()
            results.reset_index(inplace=True)
            results.rename(columns={'index': time_var}, inplace=True)
            

            print_column_description(results, 
                                     time_var,
                                     min_date=parameters["beg_eval_period"])
            ci.plot()
            st.pyplot()

            efecto_acumulado_total = results['post_cum_effects_means'].values[-1]
            valor_promedio = df_toci['y'].mean()
            porcentaje = 100*efecto_acumulado_total/valor_promedio
            col1, col2 = st.beta_columns([1,2])
            with col1:
                texto(' ', 40)
                texto('      El efecto acumulado:', 25)
                texto('Corresponde al efecto acumulado desde la intervención en adelante')
            with col2:
                estadisticos(efecto_acumulado_total, porcentaje)

            df_toci_plot = df_toci.copy()
            df_toci_plot.reset_index(inplace=True)
            df_toci_plot.rename(columns={'index': time_var}, inplace=True)
#             plot_two_df(data_1=results, 
#                         mean_col_1='preds', 
#                         data_2=df_toci_plot, 
#                         mean_col_2='y',
#                         index_col_1=time_var, 
#                         index_col_2=time_var,
#                         eval_date=parameters['beg_eval_period'],
#                         title='Predicción',
#                         min_date=parameters['beg_pre_period'])
#             plot_statistics(results, 
#                              index_col=time_var, 
#                              lower_col="point_effects_lower",
#                              upper_col='point_effects_upper', 
#                              mean_col='point_effects',
#                              title='Efecto puntual',
#                              min_date=parameters['beg_pre_period'],
#                              eval_date=parameters['beg_eval_period'])
#             plot_statistics(results, 
#                              index_col=time_var, 
#                              lower_col="post_cum_effects_lower",
#                              upper_col='post_cum_effects_upper', 
#                              mean_col='post_cum_effects',
#                              title='Efecto acumulado',
#                              min_date=parameters['beg_pre_period'],
#                              eval_date=parameters['beg_eval_period'])
                
#             plot_two_series(results_from_r,
#                            index_col=parameters['time_var'],
#                            first_serie='response',
#                            second_serie='point.pred',
#                            initial_time=parameters['beg_eval_period'])
#         if st.checkbox("Plot results"):
#             plot_statistics(results, 
#                             index_col=parameters["time_var"], 
#                             lower_col="cum.effect.lower",
#                             upper_col='cum.effect.upper', 
#                             mean_col='cum.effect',
#                             title='Efecto acumulado',
#                             min_date=parameters['end_pre_period'])


#             plot_statistics(results, 
#                             index_col=parameters["time_var"],
#                             lower_col="rel_cum_effect_lower",
#                             upper_col='rel_cum_effect_upper', 
#                             mean_col='rel_cum_effect',
#                             title='Efecto relativo acumulado',
#                             min_date=parameters['end_pre_period'])

#             get_statistics(results, 
#                            time_var=time_var, 
#                            min_date=parameters['beg_eval_period'])


def estadisticos(platita, platita_relativa):
    _border_color = "light-gray"
    _number_format = "font-size:35px; font-style:bold;"
    _cell_style = f" border: 2px solid {_border_color}; border-bottom:2px solid white; margin:10px"
    st.markdown(
        f"<table style='width: 100%; font-size:14px;  border: 0px solid gray; border-spacing: 10px;  border-collapse: collapse;'> "
        f"<tr> "
        f"<td style='{_cell_style}'> Efecto acumulado</td> "
        f"<td style='{_cell_style}'> Efecto acumulado relativo </td>"
        "</tr>"
        f"<tr style='border: 2px solid {_border_color}'> "
        f"<td style='border-right: 2px solid {_border_color}; border-spacing: 10px; {_number_format + 'font-color:red'} miles de millones' > {int(platita):,}</td> "
        f"<td style='{_number_format + 'color:red'}'> {int(platita_relativa):,} %</td>"
        "</tr>"
        "</table>"
        "<br>",
        unsafe_allow_html=True,
    )




@st.cache
def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv("data/datos_pib.csv")
    df.columns = [m.replace(' ','_') for m in df.columns]
    mapa_nombres = {k : k.replace('PIB_','') for k in df.columns}
    df.rename(columns=mapa_nombres, inplace=True)
    df['Periodo'] = df['Periodo'].apply(lambda x: x.split(' ')[0])
    df.rename(columns={'Minerales_no_metalicos_y_metalica_basica': 'Minerales_no_metalicos'}, inplace=True)
    
    return df


def find_dynamic_time_warp(df, time_var, y_var, end_training_period):
    
    texto(' ')
    st.markdown('---')
    texto(f'Dynamic time warp con {y_var}', 17)
    texto('Sugerencia: series con mayor distancia son diferentes entre sí.', 12, 'grey')
    df_filtered = df.query(f'{time_var} <= "{end_training_period}"')
    new_frame = pd.DataFrame(index=[m for m in df.columns if m != time_var and 'scaled' not in m])
    for col in df.columns:
        if 'scaled' not in col and col != time_var:
            new_frame.loc[col, 'dtw distance'] = dtw.distance(df_filtered[y_var].values, 
                                                           df_filtered[col].values)
    new_frame.sort_values(by='dtw distance', ascending=False, inplace=True)

    st.write(new_frame)


from sklearn.feature_selection import mutual_info_regression
def find_mutual_info(df, time_var, y_var, end_training_period):
    texto(' ')
    st.markdown('---')
    texto(f'Información mutua con {y_var}', 17)
    texto('Sugerencia: series con información mutua igual a 0 son independientes.', 12, 'grey')

    df_mutualinfo = df.query(f'{time_var} <= "{end_training_period}"')

    df_mutualinfo.drop(columns=[m for m in df.columns if 'scaled' in m], inplace=True)
    informacion_mutua = mutual_info_regression(df_mutualinfo[[m for m in df_mutualinfo.columns if time_var not in m]].values, 
                                            df_mutualinfo[y_var].values)
    new_frame = pd.DataFrame(index = df_mutualinfo[[m for m in df_mutualinfo.columns if time_var not in m]].columns)
    new_frame[f'mutual info'] = informacion_mutua
    new_frame.sort_values(by='mutual info', inplace=True)
    st.write(new_frame)
    
    
    
def find_mutual_info_precipitaciones(df, time_var, y_var, end_training_period):

    precipitaciones = pd.read_csv('data/precipitaciones.csv')
    precipitaciones['Periodo'] = precipitaciones['Periodo'].apply(pd.to_datetime)
#     texto(' ')
#     st.markdown('---')
#     texto(f'Información mutua entre variables con precipitaciones', 17)
#     texto('Sugerencia: series con información mutua igual a 0 son independientes.', 12, 'grey')

    full_frame = df.merge(precipitaciones, on='Periodo')
    find_mutual_info(full_frame, time_var, 'precipitaciones', end_training_period)
#     df_mutualinfo = df.query(f'{time_var} <= "{end_training_period}"')

#     df_mutualinfo.drop(columns=[m for m in df.columns if 'scaled' in m], inplace=True)
#     informacion_mutua = mutual_info_regression(df_mutualinfo[[m for m in df_mutualinfo.columns if time_var not in m]].values, 
#                                             df_mutualinfo[y_var].values)
#     new_frame = pd.DataFrame(index = df_mutualinfo[[m for m in df_mutualinfo.columns if time_var not in m]].columns)
#     new_frame[f'mutual info'] = informacion_mutua
#     new_frame.sort_values(by='mutual info', inplace=True)
#     st.write(new_frame)
    
    

def plot_vars(df_experiment, vars_to_plot, time_var, beg_pre_period=None, end_pre_period=None,
                beg_eval_period=None, end_eval_period=None):

    
    for var in vars_to_plot:
        df_experiment[f'{var}_scaled'] = (df_experiment[var] - df_experiment[var].mean())/df_experiment[var].std()

    scalled = st.checkbox('Plot scaled variables', value=False)
    if scalled:
        vars_to_plot = [f'{var}_scaled' for var in vars_to_plot]
    plotly_time_series(df_experiment, time_var, vars_to_plot, beg_pre_period, end_pre_period, beg_eval_period, end_eval_period)


def plotly_time_series(df, time_var, vars_to_plot, beg_pre_period, end_pre_period, beg_eval_period, end_eval_period):

    df_toplot = df.melt(id_vars=time_var, value_vars=vars_to_plot)
    df_toplot.sort_values(by=time_var, inplace=True)

    fig = px.line(df_toplot,
                  x=time_var,
                  y='value',
                  color='variable')

    max_y = df_toplot.value.max()
    min_y = df_toplot.value.min()
    d1_eval = datetime(int(beg_eval_period.split('-')[0]), int(beg_eval_period.split('-')[1]), int(beg_eval_period.split('-')[2]))
    d2_eval = datetime(int(end_eval_period.split('-')[0]), int(end_eval_period.split('-')[1]), int(end_eval_period.split('-')[2]))
    fecha_media_eval = d1_eval + (d2_eval-d1_eval)/2
    fig.add_shape(type="rect",
                 xref="x", yref="y",
                 x0=beg_eval_period, y0=min_y,
                 x1=end_eval_period, y1=max_y,
                 line=dict(
                     color="#D62728",
                     width=3,
                          ),
                 fillcolor=red,
                 opacity=0.2
                )

    fig.add_annotation(dict(
                            x=fecha_media_eval,
                            y=0.93*max_y,
                            showarrow=False,
                            text='Evaluation period',
                            textangle=0,
                            xref="x",
                            yref="y", opacity=0.8
                           ))

    d1_pre = datetime(int(beg_pre_period.split('-')[0]), int(beg_pre_period.split('-')[1]), int(beg_pre_period.split('-')[2]))
    d2_pre = datetime(int(end_pre_period.split('-')[0]), int(end_pre_period.split('-')[1]), int(end_pre_period.split('-')[2]))
    fecha_media_pre = d1_pre + (d2_pre-d1_pre)/2
    fig.add_shape(type="rect",
                 xref="x", yref="y",
                 x0=beg_pre_period, y0=min_y,
                 x1=end_pre_period, y1=max_y,
                 line=dict(
                     color="LightSeaGreen",
                     width=3,
                          ),
                 fillcolor="PaleTurquoise",
                 opacity=0.2
                )

    fig.add_annotation(dict(
                            x=fecha_media_pre,
                            y=0.93*max_y,
                            showarrow=False,
                            text='Pre period',
                            textangle=0,
                            xref="x",
                            yref="y", opacity=0.8
                           ))

    fig.update_layout(height=400,
                      width=1200,
                      xaxis_title="",
                      yaxis_title='Value')

    fig.update_layout(
                    #showlegend=False,
                    plot_bgcolor="white",
                    margin=dict(t=10,l=10,b=10,r=10))
    
    
    #fig.update_xaxes(visible=False, fixedrange=True)
    #fig.update_yaxes(visible=False, fixedrange=True)
    st.plotly_chart(fig)
    texto('<b>Pre period </b> corresponde al período de entrenamiento del modelo, donde aprende a reconstruir la señal y a partir de las series de control', nfont=14,
    color="grey")
    texto("""<b>Evaluation period </b> corresponde al período donde evaluamos el efecto de la intervención, propagando a futuro un escenario contrafactual a partir de las series de control""", nfont=14,
    color="grey")
    texto(' ')
    #return fig



def plot_statistics(data, 
                    lower_col="cum.effect.lower",
                    upper_col='cum.effect.upper', 
                    mean_col='cum.effect',
                    index_col="date", 
                    min_date='2019-01-01',
                    dashed_col=None, 
                    show_legend=False,
                    xaxis_title='Date', 
                    yaxis_title='Value', 
                    title=None,
                    name='Mean effect'):
    
    color_lower_upper_marker = "#C7405A"
    color_fillbetween = 'rgba(88, 44, 51, 0.3)'
    color_lower_upper_marker = color_fillbetween  # "#C7405A"
    color_median = red
    
    
    data_toplot = data.query(f'{index_col} >= "{min_date}"').copy()
    fig_list = [
        go.Scatter(
            name=name,
            x=data_toplot[index_col],
            y=data_toplot[mean_col],
            mode='lines',
            line=dict(color=color_median),
            showlegend=show_legend,
        ),
        go.Scatter(
            name=f'Upper effect',
            x=data_toplot[index_col],
            y=data_toplot[upper_col],
            mode='lines',
            marker=dict(color=color_lower_upper_marker),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name=f'Lower effect',
            x=data_toplot[index_col],
            y=data_toplot[lower_col],
            marker=dict(color=color_lower_upper_marker),
            line=dict(width=0),
            mode='lines',
            fillcolor=color_fillbetween,
            fill='tonexty',
            showlegend=False
        )
    ]
    if dashed_col is not None:
        dashed_fig = go.Scatter(
            name='Actual',
            x=data_toplot[index_col],
            y=data_toplot[dashed_col],
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=show_legend,
        )
        fig_list = [dashed_fig] + fig_list
    fig = go.Figure(fig_list)
    fig.update_layout(
        yaxis=dict(title=yaxis_title, showgrid=False),
        xaxis=dict(title=xaxis_title, showgrid=False),
        height=400, width=950,
        title=title,
        hovermode="x",
        paper_bgcolor='white',
        plot_bgcolor='white',
        hoverlabel_align='right',
        #margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig)

    
def plot_two_series(df: pd.DataFrame,
                    index_col: str = 'Periodo',
                    first_serie: str = 'response',
                    second_serie: str = 'point.pred',
                    initial_time: str = '2020-01-01'):
    
    df_toplot = df.melt(id_vars=index_col, 
                        value_vars=[first_serie, second_serie])
    df_toplot['Periodo'] = pd.to_datetime(df_toplot['Periodo']).dt.date
    
    st.write(initial_time)
    #st.write(df_toplot.dtypesr)

    df_toplot.sort_values(by=index_col, inplace=True)
    fig = px.line(df_toplot.query(f'{index_col} >= "{initial_time}"'), 
                  x=index_col, 
                  y='value', 
                  color='variable')
    st.plotly_chart(fig)




def texto(texto : str = 'jeje',
          nfont : int = 16,
          color : str = 'black',
          line_height : float =None,
          sidebar: bool = False):
    
    if sidebar:
        st.sidebar.markdown(
                body=generate_html(
                    text=texto,
                    color=color,
                    font_size=f"{nfont}px",
                    line_height=line_height
                ),
                unsafe_allow_html=True,
                )
    else:
        st.markdown(
        body=generate_html(
            text=texto,
            color=color,
            font_size=f"{nfont}px",
            line_height=line_height
        ),
        unsafe_allow_html=True,
        )
    

def generate_html(
    text,
    color=COLOR_MAP["default"],
    bold=False,
    font_family=None,
    font_size=None,
    line_height=None,
    tag="div",
):
    if bold:
        text = f"<strong>{text}</strong>"
    css_style = f"color:{color};"
    if font_family:
        css_style += f"font-family:{font_family};"
    if font_size:
        css_style += f"font-size:{font_size};"
    if line_height:
        css_style += f"line-height:{line_height};"

    return f"<{tag} style={css_style}>{text}</{tag}>"


def max_width_(width=1000):
    max_width_str = f"max-width: {width}px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


def link_libreria():
    link_libreria = (
    "https://github.com/WillianFuks/tfcausalimpact"
)
    st.markdown(
        body=generate_html(
            tag="h4",
            text=f"<u><a href=\"{link_libreria}\" target=\"_blank\" style=color:{COLOR_MAP['pink']};>"
            "Metodología</a></u> <span> &nbsp;&nbsp;&nbsp;&nbsp</span>"
            "<hr>",
        ),
        unsafe_allow_html=True,
    )
    
def disclaimer():
    st.markdown(
        body=generate_html(
            text="<strong>Disclaimer:</strong> <em>The creators of this application are not climate professionals. "
            "The illustrations provided were estimated using best available data but might not accurately reflect reality.</em>",
            color="gray",
            font_size="12px",
        ),
        unsafe_allow_html=True,
    )
    
def print_column_description(df, time_var, min_date):
    
    
    st.write(df)

    col1, col2 = st.beta_columns(2)
    with col1:
        texto('<b> complete_preds </b>: Todas las prediccciones.', nfont=11, color='gray')
        texto('<b> post_preds </b>: Predicciones post intervención.', nfont=11, color='gray')
        texto('<b> post_cum_y </b>: Variable observada acumulada post intervención.', nfont=11, color='gray')
        texto('<b> post_cum_preds </b>: Predicciones acumuladas post intervención.', nfont=11, color='gray')
    with col2:
        texto('<b> point_effect </b>: Efecto puntual de la intervención.', nfont=11, color='gray')
        texto('<b> post_cum_effects </b>: Efecto acumulado post intervención.', nfont=11, color='gray')
        texto('<b> sufijo _lower </b>: Límite inferior del intervalo de la variable.', nfont=11, color='gray')
        texto('<b> sufijo _upper </b>: Límite superior del intervalo de la variable.', nfont=11, color='gray')

        
def plot_two_df(data_1='results', 
                mean_col_1='cum.effect',
                data_2='results',
                mean_col_2='cum.effect',
                index_col_1="date", 
                index_col_2="date",
                min_date='2019-01-01',
                eval_date='2020-01-01', 
                show_legend=False,
                xaxis_title='Date', 
                yaxis_title='Value', 
                title=None,
                name='Mean effect'):
    color_lower_upper_marker = "#C7405A"
    color_fillbetween = 'rgba(88, 44, 51, 0.3)'
    color_lower_upper_marker = color_fillbetween  # "#C7405A"
    color_median = red
    data_toplot = data_1.query(f'{index_col_1} >= "{min_date}"').copy()
    data_toplot_2 = data_2.query(f'{index_col_2} >= "{min_date}"').copy()
    fig_list = [
        go.Scatter(
            name=name,
            x=data_toplot[index_col_1],
            y=data_toplot[mean_col_1],
            mode='lines',
            line=dict(color=color_median),
            showlegend=show_legend,
        ),
        go.Scatter(
            name=name,
            x=data_toplot_2[index_col_2],
            y=data_toplot_2[mean_col_2],
            mode='lines',
            line=dict(color='black', width=1), #, dash='dash'
            showlegend=show_legend,
        ),        
    ]
    fig_list = fig_list
    fig = go.Figure(fig_list)
    fig.add_vline(x=eval_date, line=dict(color="black",width=1,dash='dash'))
    fig.add_hline(y=0, line=dict(color="gray",width=0.5,dash='dash'))
    fig.update_layout(
        yaxis=dict(title=yaxis_title, showgrid=False),
        xaxis=dict(title=xaxis_title, showgrid=False),
        height=400, width=950,
        title=title,
        hovermode="x",
        paper_bgcolor='white',
        plot_bgcolor='white',
        hoverlabel_align='right',
        #margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig)
   



main()
