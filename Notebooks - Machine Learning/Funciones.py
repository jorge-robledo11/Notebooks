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
def nan_values(data:DataFrame, variables:list, variable_type:str):
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
# Función para reducir el peso en memoria de un DataFrame
def downcast_dtypes(dataframe:DataFrame) -> DataFrame:

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
def get_estratified_classes(data:DataFrame, target:str) -> any:

    """
    Function to get estratified by classes

    Args:
        data: DataFrame
        target: str
    Returns:
        tmp: print
    """

    tmp = (data.groupby(target).size().sort_values(ascending=False))/len(data)
    tmp = dict(tmp)
    
    print('\t\tDistribución de clases')
    for key, value in tmp.items():
        print(f'{key}: {value*100:0.2f}%')


# Asimetría entre predictores
# Función para obtener la asimetría de los predictores
def get_skew(data:DataFrame) -> any:

    """
    Function to get skew by classes

    Args:
        data: DataFrame
    Returns:
        print
    """

    tmp = data.skew().sort_values(ascending=False)
    tmp = dict(tmp)
    
    print('\t\tAsimetría entre predictores')
    for key, value in tmp.items():
        print(f'{key}: {value:0.2f}')


# Función para detectar outliers
def get_outliers(data:DataFrame) -> list:

    """
    Returns a list of rows with outliers, 
    we define the upper and lower limit to 1.5 std
    Args:
        data: DataFrame
    Returns:
        list: outliers
    """

    outliers = list()

    # Std & Mean
    data_std = data.std()
    data_mean = data.mean()

    # Cotas
    anomaly_cut_off = data_std * 1.5
    # Inferior
    lower_limit = data_mean - anomaly_cut_off
    # Superior
    upper_limit = data_mean + anomaly_cut_off

    # Generamos los outliers
    for index, row in data.iterrows():     
        outlier = row
        if (outlier.iloc[0] > upper_limit[0]) or (outlier.iloc[0] < lower_limit[0]):
            outliers.append(index)

    return outliers


# Diagnóstico de variables
# Función para observar el comportamiento de ciertas variables
def diagnostic_plots(dataframe:DataFrame, variable:list):

    dataframe = dataframe[variable]
    for var in dataframe:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 4, figsize=(18, 6))
        fig.suptitle('Diagnostic Plots', fontsize=18)

        plt.subplot(1, 4, 1)
        sns.histplot(dataframe[var], bins=20, color='gold')
        plt.grid(which='major')

        plt.subplot(1, 4, 2)
        stats.probplot(dataframe[var], dist='norm', plot=plt)
        plt.grid()

        plt.subplot(1, 4, 3)
        sns.kdeplot(dataframe[var], shade=True, color='red')
        plt.grid()

        plt.subplot(1, 4, 4)
        sns.boxplot(y=dataframe[var], color='floralwhite', linewidth=2)

        plt.xlabel(var)
        fig.tight_layout()


# Revisar la cardinalidad de variables categóricas
# Función para graficar variables categóricas
def categoricals_plot(dataframe:DataFrame, variables: list, ylabel: str):

    for var in variables:
        plt.style.use('dark_background')
        temp_dataframe = pd.Series(dataframe[var].value_counts() / len(dataframe))

        # Graficar con los porcentajes
        fig = temp_dataframe.sort_values(ascending=False).plot.bar(color='lavender')
        fig.set_xlabel(var)

        # Añadir una línea horizontal a 5% para resaltar categorías poco comunes
        fig.axhline(y=0.05, color='#e51a4c')
        fig.set_ylabel(ylabel)

        plt.xticks(rotation=25)
        plt.grid()
        plt.show()
