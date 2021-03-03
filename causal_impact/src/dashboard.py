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


tfd = tfp.distributions
plt.rcParams['figure.figsize'] = [15, 10]

# from utils import (plotly_time_series, estimate_model,
#                    get_n_most_important_vars, plot_top_n_relevant_vars,
#                    plot_statistics, send_parameters_to_r, texto, max_width_,
#                    link_libreria, disclaimer, print_column_description,
#                    sidebar, plot_two_series, find_correlations)


def main():
    max_width_(width=1200)
    
    
    st.title("Mega sequía: midiendo el impacto económico :volcano: :sun_with_face:")
    texto("""Esta aplicación ayuda a explorar los resultados generados por paquetes de la librería Causal Impact para medir el impacto económico de la mega sequía usando como control series no afectadas""",
   nfont=17)
    disclaimer()

    link_libreria()
    st.markdown('## Datos del PIB')
    texto('Usamos los datos mensuales del PIB obtenidos desde la página del banco central, con datos desde 2013.')
    st.markdown('### Choose the following parameters')

    chosen_df = load_dataframe()
    col1, col2 = st.beta_columns(2)
    with col1:
        time_var = st.selectbox("Choose the time variable",
                               chosen_df.columns,
                               index=0,
                               key='time_variable') #date


    with col2:
        y_var = st.selectbox("Choose the outcome variable (y)",
                             chosen_df.columns,
                             index=2,
                             key='analysis_variable') #sales

    
    df_experiment = chosen_df.copy()
    df_experiment[time_var] = df_experiment[time_var].apply(pd.to_datetime)
    df_experiment.sort_values(time_var, inplace=True)
    df_experiment.index = range(len(df_experiment))

    parameters = sidebar(df_experiment, chosen_df, time_var, y_var)
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
        
        find_correlations(df_experiment, time_var, y_var, parameters['end_pre_period'])
        

        texto(' ')
    
    
    
    

    pre_period = [parameters['beg_pre_period'], parameters['end_pre_period']]
    post_period = [parameters['beg_eval_period'], parameters['end_eval_period']]
    
    st.write(parameters['beg_pre_period'])
    st.write(df_experiment.set_index(time_var).tail())
    #df_experiment[time_var] = pd.to_datetime(df_experiment[time_var])
    df_experiment[time_var] = df_experiment[time_var].dt.strftime('%m/%d/%Y')
    st.write(df_experiment.set_index(time_var).dtypes)
    df_experiment[y_var] = df_experiment[y_var].astype(float)
    
    mapa_nombres = {var_x: f'x_{idx}' for idx, var_x in enumerate(parameters['selected_x_vars'])}
    mapa_nombres[y_var]= 'y'
    df_toci = df_experiment.set_index(time_var).copy()
    df_toci.rename(columns=mapa_nombres, inplace=True)

    ci = CausalImpact(df_toci, pre_period, post_period)
    st.write(ci.inferences)

    
    with st.beta_expander("Estimate Causal Impact model with R"):
        if st.checkbox("Run R", False):
            #Save and run R
            df_experiment.to_csv("example_data/input_causal_impact_one_experiment.csv")
            #bag bug fix
            selected_experiment = ''
            send_parameters_to_r("example_data/parameters_for_r.json", parameters, selected_experiment)

            completed = subprocess.run(["Rscript", "causal_impact_one_experiment.R"],)
                            #capture_output=True)
            if st.checkbox("Show output from R (for debugging)"):
                st.write(completed)

            #Bring results from R
            results_from_r = pd.read_feather("example_data/results_causal_impact_from_r.feather")
            results_from_r[time_var] = pd.to_datetime(results_from_r[time_var]).dt.date
            results_from_r[time_var] = results_from_r[time_var].apply(pd.to_datetime)
            
            results_from_r['rel_cum_effect'] = results_from_r['cum.effect']/results_from_r['point.pred']
            results_from_r['rel_cum_effect_upper'] = results_from_r['cum.effect.upper']/results_from_r['point.pred']
            results_from_r['rel_cum_effect_lower'] = results_from_r['cum.effect.lower']/results_from_r['point.pred']
            
            
            if st.checkbox("Show results dataframe"):

                #st.write(results_from_r.query(f'{time_var} >= "{parameters["beg_eval_period"]}"'))

                print_column_description(results_from_r, 
                                         time_var,
                                         min_date=parameters["beg_eval_period"])

                
                
#             plot_two_series(results_from_r,
#                            index_col=parameters['time_var'],
#                            first_serie='response',
#                            second_serie='point.pred',
#                            initial_time=parameters['beg_eval_period'])
            if st.checkbox("Plot results"):
                plot_statistics(results_from_r, 
                                index_col=parameters["time_var"], 
                                lower_col="cum.effect.lower",
                                upper_col='cum.effect.upper', 
                                mean_col='cum.effect',
                                title='Efecto acumulado',
                                min_date=parameters['end_pre_period'])


                plot_statistics(results_from_r, 
                                index_col=parameters["time_var"],
                                lower_col="rel_cum_effect_lower",
                                upper_col='rel_cum_effect_upper', 
                                mean_col='rel_cum_effect',
                                title='Efecto relativo acumulado',
                                min_date=parameters['end_pre_period'])

            get_statistics(results_from_r, 
                           time_var=time_var, 
                           min_date=parameters['beg_eval_period'])
#     with st.beta_expander('Estimate Causal Impact model with python'):
#         if st.checkbox("Run Python", False):
#             #Where is the input for time_var
#             x_vars = [parameters["y_var"], "x0", "x1"]
#             df__ = df_experiment[x_vars]
#             ci = estimate_model(df__, parameters["y_var"],
#                                 x_vars,
#                                 3,
#                                 170,
#                                 171,
#                                 220)
#             fig, _ = ci.plot()
#             st.pyplot(fig)
#             st.text(ci.summary())
# #
#             st.text(ci.summary('report'))
#     #
#             st.markdown("### Most important variables (according to coefficients)")
#     #
#             top_n_vars = get_n_most_important_vars(ci, 1)
#     #
#             y_and_top_vars = [parameters["y_var"]] + top_n_vars
#             fig, _ = plot_top_n_relevant_vars(df__,
#                      parameters["time_var"], y_and_top_vars,
#                                               171)
#             st.pyplot(fig)




COLOR_MAP = {"default": "#262730", "pink": "#E22A5B"}
red = "#E07182"
green = "#96D6B4"
blue = "#4487D3"
grey = "#87878A"

@st.cache
def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv("example_data/datos_pib.csv")
    df.columns = [m.replace(' ','_') for m in df.columns]
    mapa_nombres = {k : k.replace('PIB_','') for k in df.columns}
    df.rename(columns=mapa_nombres, inplace=True)
    df['Periodo'] = df['Periodo'].apply(lambda x: x.split(' ')[0])
    
    return df


def plot_vars(df_experiment, vars_to_plot, time_var, beg_pre_period=None, end_pre_period=None,
                beg_eval_period=None, end_eval_period=None):

    
    for var in vars_to_plot:
        df_experiment[f'{var}_scaled'] = (df_experiment[var] - df_experiment[var].mean())/df_experiment[var].std()

    scalled = st.checkbox('Plot scaled variables', value=False)
    if scalled:
        vars_to_plot = [f'{var}_scaled' for var in vars_to_plot]
    plotly_time_series(df_experiment, time_var, vars_to_plot, beg_pre_period, end_pre_period, beg_eval_period, end_eval_period)
    

def get_statistics(df, time_var: str, min_date: str):
    
    st.write('jejeje')

    
def find_correlations(df, time_var, y_var, end_training_period):
    texto(' ')
    st.markdown('---')
    texto('Análisis de las correlaciones entre variables', 17)
    texto('Sugerencia: series con baja correlación pueden introducir ruido o sesgo en los resultados', 12, 'grey')
    df_tocorr = df.query(f'{time_var} <= "{end_training_period}"')
    df_tocorr.drop(columns=[m for m in df.columns if 'scaled' in m], inplace=True)
    correlaciones_absolutas = df_tocorr.corr().abs()
    new_var_name =  f'correlación absoluta con {y_var}'
    correlaciones_absolutas.rename(columns={y_var: new_var_name}, inplace=True)
    st.dataframe(correlaciones_absolutas.sort_values(new_var_name, ascending=False)[1:][new_var_name])#[y_var])
    


class myCausalImpact(CausalImpact):
    def __init__(self, data, pre_period, post_period, model=None, alpha=0.05, **kwargs):
        super(myCausalImpact, self).__init__(data, pre_period,
                                             post_period, model, alpha, **kwargs)
        #checked_input = self._process_input_data(
        #    data, pre_period, post_period, model, alpha, **kwargs
        #)


    def plot(self, panels=['original', 'pointwise', 'cumulative'], figsize=(15, 12)):
        """Plots inferences results related to causal impact analysis.
            Args
            ----
            panels: list.
                Indicates which plot should be considered in the graphics.
            figsize: tuple.
                Changes the size of the graphics plotted.
            Raises
            ------
            RuntimeError: if inferences were not computed yet.
            """
        
        fig = plt.figure(figsize=figsize)
        if self.summary_data is None:
            raise RuntimeError(
                'Please first run inferences before plotting results')

        valid_panels = ['original', 'pointwise', 'cumulative']
        for panel in panels:
            if panel not in valid_panels:
                raise ValueError(
                    '"{}" is not a valid panel. Valid panels are: {}.'.format(
                        panel, ', '.join(['"{}"'.format(e)
                                            for e in valid_panels])
                    )
                )

        # First points can be noisy due approximation techniques used in the likelihood
        # optimizaion process. We remove those points from the plots.
        llb = self.trained_model.filter_results.loglikelihood_burn #type: ignore
        inferences = self.inferences.iloc[llb:]

        intervention_idx = inferences.index.get_loc(self.post_period[0])
        n_panels = len(panels)
        ax = plt.subplot(n_panels, 1, 1)
        idx = 1

        if 'original' in panels:
            ax.plot(pd.concat([self.pre_data.iloc[llb:, 0], self.post_data.iloc[:, 0]]),  # type: ignore
                    'k', label='y') 
            ax.plot(inferences['preds'], 'b--',
                    label='Predicted')  # type: ignore
            ax.axvline(
                inferences.index[intervention_idx - 1], c='k', linestyle='--')
            ax.fill_between(
                self.pre_data.index[llb:].union(self.post_data.index),
                inferences['preds_lower'],
                inferences['preds_upper'],
                facecolor='blue',
                interpolate=True,
                alpha=0.25
            )
            ax.grid(True, linestyle='--')
            ax.legend()
            if idx != n_panels:
                plt.setp(ax.get_xticklabels(), visible=False)
            idx += 1

        if 'pointwise' in panels:
            ax = plt.subplot(n_panels, 1, idx, sharex=ax)
            ax.plot(inferences['point_effects'], 'b--', label='Point Effects')
            ax.axvline(
                inferences.index[intervention_idx - 1], c='k', linestyle='--')
            ax.fill_between(
                inferences['point_effects'].index,
                inferences['point_effects_lower'],
                inferences['point_effects_upper'],
                facecolor='blue',
                interpolate=True,
                alpha=0.25
            )
            ax.axhline(y=0, color='k', linestyle='--')
            ax.grid(True, linestyle='--')
            ax.legend()
            if idx != n_panels:
                plt.setp(ax.get_xticklabels(), visible=False)  # type: ignore
            idx += 1

        if 'cumulative' in panels:
            ax = plt.subplot(n_panels, 1, idx, sharex=ax)
            ax.plot(inferences['post_cum_effects'], 'b--',
                    label='Cumulative Effect')
            ax.axvline(
                inferences.index[intervention_idx - 1], c='k', linestyle='--')
            ax.fill_between(
                inferences['post_cum_effects'].index,
                inferences['post_cum_effects_lower'],
                inferences['post_cum_effects_upper'],
                facecolor='blue',
                interpolate=True,
                alpha=0.25
            )
            ax.grid(True, linestyle='--')  # type: ignore
            ax.axhline(y=0, color='k', linestyle='--')  # type: ignore
            ax.legend()  # type: ignore

        # Alert if points were removed due to loglikelihood burning data
        if llb > 0:
            text = ('Note: The first {} observations were removed due to approximate '
                    'diffuse initialization.'.format(llb))
            fig.text(0.1, 0.01, text, fontsize='large')  # type: ignore

        return fig, fig.axes  # type: ignore


def send_parameters_to_r(file_name: str, parameters: dict, selected_experiment: str) -> None:
    """
    Collects relevant parameters and sends them to r as a json
    """
    parameters["experiment"] = selected_experiment

    with open(file_name, "w") as outfile:
        json.dump(parameters, outfile)


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
                      width=950,
                      xaxis_title="",
                      yaxis_title='Value')

    fig.update_layout(
                    #showlegend=False,
                    plot_bgcolor="white",
                    margin=dict(t=10,l=10,b=10,r=10))
    
    
    #fig.update_xaxes(visible=False, fixedrange=True)
    #fig.update_yaxes(visible=False, fixedrange=True)
    st.plotly_chart(fig)
    texto('<b>Pre period </b> the model uses this period to learn patterns', nfont=14,
    color="grey")
    texto("""<b>Evaluation period </b> the model uses this period to evaluate the 
    intervention. It doesn't use it for training""", nfont=14,
    color="grey")
    texto(' ')
    #return fig

#Not sure if this speeds up anything
@st.cache(hash_funcs={myCausalImpact: id})
def estimate_model(df: pd.DataFrame, y_var_name: str, x_vars: list,
                 beg_pre_period, end_pre_period, beg_eval_period,
                   end_eval_period) -> myCausalImpact:
    st.write("caching didn't work")
    pre_period = [beg_pre_period, end_pre_period]
    eval_period = [beg_eval_period, end_eval_period]
    selected_x_vars_plus_target = [y_var_name] + x_vars
    ci = myCausalImpact(
        df[selected_x_vars_plus_target], pre_period, eval_period)
    return ci


def get_n_most_important_vars(trained_c_impact: myCausalImpact, top_n: int):
    """
    Get the names of the n most important variables in the training of the causal impact
    model.
    Most important is given by the absolute value of the coefficient
    (I THINK that data is standardized beforehand so scale of X shouldn't matter)
    """
    params: pd.Series = trained_c_impact.trained_model.params #type: ignore
    contains_beta = params.index.str.contains("beta")
    does_not_contain_t = params.index != "beta.t"
    params = params[contains_beta & does_not_contain_t]
    params = np.abs(params)

    top_n_vars = params.sort_values(ascending=False).index.values[:top_n]

    top_n_vars = [var.split(".")[1] for var in top_n_vars]
    return top_n_vars


def plot_top_n_relevant_vars(df, time_var, y_and_top_vars,
                            beg_eval_period):
    n_total_vars = len(y_and_top_vars)
    fig, axes = plt.subplots(n_total_vars, 1,
                             figsize=(6, 1.5*n_total_vars))
    for ax, var_name in zip(axes, y_and_top_vars):  # type: ignore
        ax.plot(df[time_var], df[var_name])
        ax.set_title(var_name)
        ax.axvline(beg_eval_period, c='k', linestyle='--')

    return fig, axes


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

def plot_logo_spike():
    st.sidebar.markdown(
        "<br>"
        '<div style="text-align: center;">'
        '<a href="http://www.spikelab.xyz/"> '
        '<img src="https://raw.githubusercontent.com/SpikeLab-CL/calidad_aire_2050_cr2/master/logo/logo_con_caption.png" width=150>'
        " </img>"
        "</a> </div>",
        unsafe_allow_html=True,
    )
    
def link_libreria():
    link_libreria = (
    "https://google.github.io/CausalImpact/CausalImpact.html"
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
    
    
    st.write(df.query(f'{time_var} >= "{min_date}"')[[m for m in df.columns if 'upper' not in m and 'lower' not in m]])
    col1, col2 = st.beta_columns(2)
    with col1:
        texto('<b> response </b>: Observed response as supplied to CausalImpact().', nfont=11, color='gray')
        texto('<b> cum.response </b>: Cumulative response during the modeling period.', nfont=11, color='gray')
        texto('<b> point.pred </b>: Posterior mean of counterfactual predictions.', nfont=11, color='gray')
        texto('<b> point.pred.lower </b>: Lower limit of a (1 - alpha) posterior interval.', nfont=11, color='gray')
        texto('<b> point.pred.upper </b>: Upper limit of a (1 - alpha) posterior interval.', nfont=11, color='gray')
        texto('<b> cum.pred </b>: Posterior cumulative counterfactual predictions.', nfont=11, color='gray')
        texto('<b> cum.pred.lower </b>: Lower limit of a (1 - alpha) posterior interval.', nfont=11, color='gray')
    with col2:
        texto('<b> cum.pred.lower </b>: Upper limit of a (1 - alpha) posterior interval.', nfont=11, color='gray')
        texto('<b> point.effect </b>: Point-wise posterior causal effect.', nfont=11, color='gray')
        texto('<b> point.effect.lower </b>: Lower limit of the posterior interval (as above).', nfont=11, color='gray')
        texto('<b> point.effect.lower </b>: LUpperower limit of the posterior interval (as above).', nfont=11, color='gray')
        texto('<b> cum.effect </b>: Posterior cumulative effect.', nfont=11, color='gray')
        texto('<b> cum.effect.lower </b>: Lower limit of the posterior cumulative interval (as above).', nfont=11, color='gray')
        texto('<b> cum.effect.lower </b>: Upper limit of the posterior cumulative interval (as above).', nfont=11, color='gray')

        
def sidebar(df_experiment : pd.DataFrame, 
            chosen_df : pd.DataFrame,
            time_var,
            y_var):

    image = Image.open('causal_impact_explainer_logo.png')
    st.sidebar.image(image, caption='', use_column_width=True)

    st.sidebar.markdown("#### Select the variables you will use as control")

    x_vars = [col for col in chosen_df.columns if col != y_var and col != time_var and col!='group']
    selected_x_vars = st.sidebar.multiselect("Better less than more", x_vars,
                        default=x_vars)

    st.sidebar.markdown("## Experiment setting")

    alpha = st.sidebar.number_input("Significance level", 0.01, 0.5, value=0.05, 
                                    step=0.01, 
                                    key='significance_level')


    min_date = df_experiment[time_var].min().date()
    last_date = df_experiment[time_var].max().date()
    mid_point = int(len(df_experiment) / 2)

    st.sidebar.markdown("### Beginning and end pre period")

    beg_pre_period, end_pre_period = st.sidebar.slider('', min_date, last_date, 
                                                       value=(min_date,
                                                       df_experiment.loc[mid_point + 20, time_var].date()),
                                                      key='training_period')

    st.sidebar.markdown("### Beginning and end evaluation period")
    beg_eval_period, end_eval_period = st.sidebar.slider('',
                                                         end_pre_period, last_date,
                                                         value=(df_experiment.loc[mid_point + 26, time_var].date(),
                                                         last_date),
                                                         key='evaluation_period')
    
    strftime_format="%Y-%m-%d"
    parameters = {"alpha": alpha, 
                  "beg_pre_period": beg_pre_period.strftime(strftime_format),
                  "end_pre_period": end_pre_period.strftime(strftime_format),
                  "beg_eval_period": beg_eval_period.strftime(strftime_format),
                  "end_eval_period": end_eval_period.strftime(strftime_format),
                  "selected_x_vars": selected_x_vars,
                  "y_var": y_var,
                  "time_var": time_var,
                 }

    return parameters






main()
