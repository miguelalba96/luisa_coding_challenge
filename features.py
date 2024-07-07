from datetime import date
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier


CATEGORICAL_COLUMNS = [
    "estado_por_periodo",
    "estado_actual_credito",
    "actividad_cliente",
    "oficina",
    "tipo_crédito",
    "linea_credito",
    "ciudad_cliente",
    "causal_incorporacion",
    "calificacion_deterioro",
    "departamento_cliente",
    "tipo_contrato",
    "estado_civil",
    "tipo_vivienda",
    "estado_incorporacion",
    "alturamora_act",
    "alturamora_ant",
    "rangoantlab",
    "rangoingresos",
    "rangomontoaprobado",
    "marca_incorporado",
    "sfc",
    "regional",
    "tipo_producto",
    "venta_en_firme",
    "nombre_oficina",
    "propiedad_cartera",
    "nombre_oficina",
    "marca_capitalizado",
    "fondeador",
    "pagaduría",
]

NO_USE_COLUMNS = [
    "periodo_cerrado",
    "fecha_primer_descuento",  # weird format,
    "periodo_primer_recaudo_recibido",  # weird format
    "marca_unica",
    "marca_cédula",
]

DATE_COLUMNS = [
    "fecha_nacimiento",
    "fecha_activacion_contable",
    "fecha_corte",
    "fecha_terminacion_credito",
]


@dataclass
class FeatureBuilder:
    """
    Builds a features table with all relevant information to do standard classification
    :param file_path: path to the csv file
    :param encodings: dictionary of encodings for categorical columns, in case of need for further processing
        ex. {"F": 0, "M": 1}
    """
    file_path: str
    encodings: defaultdict = None

    def __post_init__(self):
        self.encodings = defaultdict(dict)

    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path, sep=";", encoding="latin-1")
        df.drop("Unnamed: 0", axis=1, inplace=True)
        df.drop("Unnamed: 2", axis=1, inplace=True)
        df.drop("Unnamed: 20", axis=1, inplace=True)
        return df

    @staticmethod
    def preprocess_column_names(df: pd.DataFrame) -> pd.DataFrame:
        column_names = df.columns.str.replace(" ", "_")
        df.columns = column_names.str.lower()
        return df

    @staticmethod
    def create_target(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a binary classification target variable based on mora_maxima > 90
        """
        df["BM"] = (df["mora_máxima"] > 90) * 1
        df["BM"] = df["BM"].astype("int64")
        return df

    @staticmethod
    def convert_to_datetime(df: pd.DataFrame, column) -> pd.DataFrame:
        df[column] = pd.to_datetime(df[column], format='%d/%m/%Y')
        return df

    @staticmethod
    def fix_age_feature(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix fecha_nacimiento and edad_cliente
        """
        df["fecha_nacimiento"] = pd.to_datetime(df["fecha_nacimiento"], format='%d/%m/%Y')
        df["edad_cliente"] = date.today().year - df["fecha_nacimiento"].dt.year
        df["edad_cliente"] = df["edad_cliente"] - 1  # match with the original column values
        return df

    @staticmethod
    def fix_estrato_feature(df: pd.DataFrame) -> pd.DataFrame:
        # estrato column UNO, DOS, TRES, CUATRO, CINCO, SEIS
        mapping = {
            'DOS': 2,
            'TRES': 3,
            'UNO': 1,
            'CUATRO': 4,
            'CINCO': 5,
            'SEIS': 6,
            'PIRATA': -1,
            '2': -1
        }
        df["estrato"] = df["estrato"].map(mapping)
        return df

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert natural language or categorical features into numerical features
        """
        df["genero"] = df["genero"].apply(lambda x: 1 if x == "M" else 0)
        # Create age from fecha_de_nacimiento
        df = self.fix_age_feature(df)
        df = self.fix_estrato_feature(df)

        # convert categorical columns into integers
        for column in CATEGORICAL_COLUMNS:
            # replace NaN first with "unknown"
            df[column] = df[column].fillna("unknown")

            column_as_category = df[column].astype("category")
            df[column] = column_as_category.cat.codes
            # save the cat -> code relation in encodings defaultdict
            self.encodings[column] = dict(
                zip(column_as_category.cat.categories, column_as_category.cat.codes)
            )
            # for "unknown" code revert back to NaN for imputation
            unknown_code = self.encodings[column].get("unknown")
            df.loc[df[column] == unknown_code, column] = np.nan

        # delete columns with no use
        # This is performed because I don't know the business logic behind
        # those and the meaning of their values
        df.drop(NO_USE_COLUMNS, axis=1, inplace=True)

        # convert columns to datetime
        for column in DATE_COLUMNS:
            df = self.convert_to_datetime(df, column)

        # all remaining object columns make them numerical
        for column in df.select_dtypes(include=[object]).columns:
            df[column] = df[column].str.replace(",", ".")
            df[column] = df[column].astype("float64")
        return df


def get_imputed_dataframe(df: pd.DataFrame, num_neighbors: int=5) -> pd.DataFrame:
    """
    Apply KNN imputation for missing values
    :param df: features
    :param num_neighbors: number of neighbors to consider for KNN imputation
    """
    imputer = KNNImputer(n_neighbors=num_neighbors)

    # select all numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    inputed_data = imputer.fit_transform(df[numerical_columns])
    df = pd.DataFrame(inputed_data, columns=df.columns)
    return df


def get_minus_one_imputed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all -1 values with NaN
    """
    df = df.copy().fillna(-1)
    return df


def apply_recursive_feature_elimination(
        df: pd.DataFrame, target: str, num_features: int = 20
) -> pd.DataFrame:
    """
    Apply Recursive Feature Elimination to select the most important features
    :param df: features
    :param target: target variable
    :param num_features: number of features to select
    """
    # select all numerical columns
    numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
    numerical_columns = [col for col in numerical_columns if col != target]
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=num_features)
    rfe.fit(df[numerical_columns], df[target])
    selected_columns = df[numerical_columns].columns[rfe.support_]
    return df[selected_columns]

