# Librerías
from ast import Try
from pandas import DataFrame
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Valores faltantes
# Función para observar variables con datos nulos y su porcentaje
def nan_values(data: DataFrame, variables: list, variable_type:str):
    """
    Function to observe variables with nan values and their percentages

    Args:
        data: DataFrame
        variables: list
        variable_type: str
    Returns:
        print: Variables that meet this condition
    """
    print('Variables ' + variable_type)

    for var in variables:
        if data[var].isnull().sum() > 0:
            print(f'{var} {data[var].isnull().mean()*100:0.2f}%')


# Downcasting
def downcast_dtypes(dataframe: DataFrame) -> DataFrame:

    """
    Function to downcast any type variable

    Args:
        dataframe: DataFrame
    Returns
        DataFrame: Downcasted DataFrame
    """

    start = dataframe.memory_usage(deep=True).sum() / 1024 ** 2
    float_cols = [col for col in dataframe if dataframe[col].dtype == 'float64']
    int_cols = [col for col in dataframe if dataframe[col].dtype in ['int64', 'int32']]
    object_cols = [col for col in dataframe if dataframe[col].dtype in ['object', 'bool']]

    dataframe[float_cols] = dataframe[float_cols].astype(np.float32)
    dataframe[int_cols] = dataframe[int_cols].astype(np.int16)
    dataframe[object_cols] = dataframe[object_cols].astype('category')

    end = dataframe.memory_usage(deep=True).sum() / 1024 ** 2
    saved = (start - end) / start * 100
    print(f'Memory Saved: {saved:0.2f}%', '\n')
    dataframe.info()

    return dataframe

# Variables estratificadas por clases
# Función para obtener la estratificación de clases/target
def get_estratified_classes(data:DataFrame, target:str):

    """
    Function to get estratified by classes

    Args:
        data: DataFrame
        target: str
    Returns:
        tmp: dict
    """

    tmp = (data.groupby(target).size().sort_values(ascending=False))/len(data)
    tmp = dict(tmp)
    
    print('Clases estratificadas')
    for key, value in tmp.items():
        print(f'{key}: {value*100:0.2f}%')


# Diagnóstico de variables
# Función para observar el comportamiento de ciertas variables
def diagnostic_plots(dataframe, variable):

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle('Diagnostic Plots', fontsize=18)

    plt.subplot(1, 4, 1)
    sns.histplot(dataframe[variable], bins=20, color='gold')
    plt.grid(which='major')

    plt.subplot(1, 4, 2)
    stats.probplot(dataframe[variable], dist='norm', plot=plt)
    plt.grid()

    plt.subplot(1, 4, 3)
    sns.kdeplot(dataframe[variable], shade=True, color='red')
    plt.grid()

    plt.subplot(1, 4, 4)
    sns.boxplot(y=dataframe[variable], color='floralwhite', linewidth=2)

    plt.xlabel(variable)
    fig.tight_layout()


# Revisar la cardinalidad de variables categóricas
# Función para graficar variables categóricas
def categoricals_plot(dataframe, variables: list, ylabel: str):

    plt.style.use('dark_background')
    for var in variables:

        temp_dataframe = pd.Series(dataframe[var].value_counts() / len(dataframe))

        # Graficar con los porcentajes
        fig = temp_dataframe.sort_values(ascending=False).plot.bar(color='lavender')
        fig.set_xlabel(var)

        # Añadir una línea horizontal a 5% para resaltar categorías poco comunes
        fig.axhline(y=0.05, color='#e51a4c')
        fig.set_ylabel(ylabel)

        plt.grid()
        plt.show()
