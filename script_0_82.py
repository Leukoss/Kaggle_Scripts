import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def load_test_set() -> pd.DataFrame:
    """
    Charge le jeu de données test complet
    :return: un dataframe test complet
    """
    files = ['USAGERS', 'VEHICULES', 'CARACTERISTIQUES', 'LIEUX']
    data_path = '../Data/TEST/TEST_FILES/'
    full_data = pd.DataFrame()

    for file_name in files:
        path_to_file = data_path + file_name + '.csv'
        data = pd.read_csv(path_to_file, encoding="latin1", sep=",",
                           low_memory=False)

        if file_name == 'USAGERS':
            full_data = pd.concat([full_data, data], axis=1)
        else:
            if file_name == 'VEHICULES':
                full_data = pd.merge(full_data, data, on=['Num_Acc', 'num_veh'],
                                     how='left')
            else:
                full_data = pd.merge(full_data, data, on=['Num_Acc'],
                                     how='left')

    return full_data


def load_train_set() -> pd.DataFrame:
    """
    Charge le jeu de données train complet
    :return: un dataframe train complet
    """
    years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
             '2020', '2021', '2022']
    files = ['usagers_', 'vehicules_', 'caracteristiques_', 'lieux_', ]
    data_path = '../Data/TRAIN/TRAIN_FILES/BAAC-Annee-'
    full_data = pd.DataFrame()

    for year in years:
        path_to_baac = data_path + year + '/'
        year_file = pd.DataFrame()

        for file_name in files:
            path_to_file = path_to_baac + file_name + year + '_.csv'
            data = pd.read_csv(path_to_file,
                               encoding="latin1",
                               sep=";",
                               low_memory=False)
            data = data.drop(data.columns[0], axis=1)

            if file_name == 'usagers_':
                year_file = pd.concat([year_file, data], axis=1)
            else:
                if file_name == 'vehicules_':
                    year_file = pd.merge(year_file, data,
                                         on=['Num_Acc', 'num_veh'], how='left')
                else:
                    year_file = pd.merge(year_file, data, on=['Num_Acc'],
                                         how='left')

        full_data = pd.concat([full_data, year_file], axis=0)

    return full_data


def divide_dataset(full_data: pd.DataFrame, train=True) -> (pd.DataFrame,
                                                            pd.DataFrame):
    """
    Permet de diviser un jeu de données complet en deux (2012-2018 et 2019-2022)
    :param full_data: jeu de données complet
    :param train: de base, on considère que nous traitons le cas d'un jeu de
    train, dans le cas contraire, on traite un jeu test
    :return:
    """
    # Harmonisation des années pour le tri
    full_data.loc[(full_data['an'] >= 12) & (full_data['an'] <= 18), 'an'] \
        += 2000

    if train:
        # Binarisation de la variable à prédire pour l'entraînement
        full_data.replace({'grav': {1: 0, 4: 0, 2: 1, 3: 1}}, inplace=True)

    # Jeu de 2012 à 2018
    data_2012_to_2018 = full_data[(full_data['an'] >= 2012) &
                                  (full_data['an'] <= 2018)]

    # Jeu de 2019 à 2022
    data_2019_to_2022 = full_data[(full_data['an'] >= 2019) &
                                  (full_data['an'] <= 2022)]

    return data_2012_to_2018, data_2019_to_2022


def drop_feature(columns: list, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes non pertinentes pour l'étude, car trop vides
    :param columns: liste des variables à supprimer
    :param dataframe: jeu de données où supprimer les variables
    :return: un dataframe filtré
    """
    dataframe = dataframe.drop(columns=columns)
    return dataframe


def treatment_2012_2018(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permet d'appliquer des changements pour harmoniser les valeurs de l'ensemble
    :param df: dataframe à harmoniser
    :return: un dataframe harmonisé
    """
    df = df.fillna(pd.NA)

    df = df.replace({'sexe': {2: 0}, 'agg': {2: 0}, 'env1': {99: 2},
                     'secu': {1: 13, 2: 23, 3: 33}})

    df['senc'] = df['senc'].fillna(0)
    df['place'] = df['place'].fillna(0)

    feat_0_to_na = ['trajet', 'locp', 'surf', 'prof', 'plan', 'situ', 'nbv']
    df[feat_0_to_na] = df[feat_0_to_na].replace(0, pd.NA)

    df['voie'] = df['voie'].str.split('.').str[0]

    # ZONE DANGER
    df['lat'] = df['lat'].astype(float) / 100000
    df['long'] = df['long'].astype(float) / 100000
    df['lartpc'] = df['lartpc'] / 100

    return df


def treatment_2019_2022(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permet d'appliquer des changements pour harmoniser les valeurs de l'ensemble
    :param df: dataframe à harmoniser
    :return: un dataframe harmonisé
    """
    df = df.replace({'sexe': {2: 0}, 'agg': {2: 0}, 'nbv': {0: pd.NA, -1: pd.NA}
                        , 'vma': {900: 90, 700: 70, 502: 50, 500: 50, 300: 30,
                                  42: 40, -1: pd.NA, 1: pd.NA, 2: pd.NA, 3:
                                      pd.NA, 5: pd.NA, 6: pd.NA, 7: pd.NA, 8:
                                      pd.NA, 10: pd.NA, 15: pd.NA}})

    return df


def drop_na_row(df: pd.DataFrame, is_2012_2018=True) -> (
        pd.DataFrame):
    """
    Permet de supprimer les lignes contenant des valeurs vides dans le train set
    en cas de petites quantités
    :param is_2012_2018: en fonction de ce paramètre, on ajustera les valeurs à
    supprimer
    :param df: jeu de données d'entraînement
    :return: un nouveau jeu de données plus court
    """
    if is_2012_2018:
        cols_to_drop = ['an_nais', 'obs', 'obsm', 'choc', 'manv', 'atm', 'col',
                        'circ', 'vosp', 'surf', 'infra', 'env1', 'secu', 'prof',
                        'plan', 'situ']
    else:
        cols_to_drop = ['an_nais', 'adr', 'vma']

    df = df.dropna(subset=cols_to_drop)

    return df


def parse_to_categorical(df_train_2012_2018: pd.DataFrame,
                         df_train_2019_2022: pd.DataFrame,
                         df_test_2012_2018: pd.DataFrame,
                         df_test_2019_2022: pd.DataFrame) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Permet de parser les variables afin de construire nos modèles
    :param df_train_2012_2018: train set 2012/2018
    :param df_train_2019_2022: train set 2019/2022
    :param df_test_2012_2018: test set 2012/2018
    :param df_test_2019_2022: test set 2019/2022
    :return: les 4 jeux de données dans le même ordre mais parsés
    """
    feat_numerical = ['occutc', 'lat', 'long', 'lartpc', 'larrout']

    feat_cat_2012_2018 = [col for col in df_train_2012_2018.columns if
                          col not in feat_numerical]

    feat_cat_2019_2022 = [col for col in df_train_2019_2022.columns if
                          col not in feat_numerical]

    def feat_cat(df_test: pd.DataFrame, df_train: pd.DataFrame, list_feat: list
                 ) -> (pd.DataFrame, pd.DataFrame):
        """
        Evite la duplication de code pour les variables catégorielles
        :param df_test: jeu de test
        :param df_train: jeu de train
        :param list_feat: liste des variables catégorielles
        :return: les deux jeux de données avec des variables catégorielles
        """
        for col in list_feat:
            df_train[col] = df_train[col].astype('category')
            if col in df_test.columns:
                df_test[col] = df_test[col].astype('category')

        return df_test, df_train

    df_test_2012_2018, df_train_2012_2018 = feat_cat(
        df_test_2012_2018, df_train_2012_2018, feat_cat_2012_2018)

    df_test_2019_2022, df_train_2019_2022 = feat_cat(
        df_test_2019_2022, df_train_2019_2022, feat_cat_2019_2022)

    return (df_train_2012_2018, df_train_2019_2022, df_test_2012_2018,
            df_test_2019_2022)


def lightgbm(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Permet de réaliser un modèle et de prédire un jeu test
    :param df_train: jeu d'entraînement
    :param df_test: jeu de test
    :return: prédiction (Num_Acc, GRAVE)
    """

    x = df_train.drop(columns=['grav'], axis=1)
    y = df_train['grav']

    x_pred = df_test.drop(columns=['Num_Acc'], axis=1)
    num_acc = df_test['Num_Acc']

    common_columns = x.columns.intersection(x_pred.columns)
    x_pred = x_pred[x.columns]
    x_pred = x_pred[common_columns]

    print(f'x.shape: {x.shape} - y.shape: {y.shape}')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=42)

    lgb_classifier = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        objective='binary',
        random_state=42
    )

    lgb_classifier.fit(x_train, y_train)

    # Feature importances
    feature_importances = pd.DataFrame({'Feature': x.columns,
                                        'Importance':
                                            lgb_classifier.feature_importances_}
                                       )
    print(feature_importances)

    accuracy = lgb_classifier.score(x_test, y_test)
    print("Validation Accuracy:", accuracy)

    y_pred = lgb_classifier.predict_proba(x_pred)[:, 1]
    df_pred = pd.DataFrame({'Num_Acc': num_acc, 'GRAVE': y_pred})

    return df_pred


# Charge les données
full_data_test = load_test_set()
full_data_train = load_train_set()

# On sauvergarde les fichiers
full_data_test.to_csv('TEST/FULL_TEST.csv', index=False)
full_data_train.to_csv('TRAIN/FULL_TRAIN.csv', index=False)

full_data_test = pd.read_csv('TEST/FULL_TEST.csv', sep=',', low_memory=False)
full_data_train = pd.read_csv('TRAIN/FULL_TRAIN.csv', sep=',', low_memory=False)

# On sauvergarde les fichiers
full_data_test.to_csv('TEST/FULL_TEST.csv', index=False)
full_data_train.to_csv('TRAIN/FULL_TRAIN.csv', index=False)

full_data_test = pd.read_csv('TEST/FULL_TEST.csv', sep=',', low_memory=False)
full_data_train = pd.read_csv('TRAIN/FULL_TRAIN.csv', sep=',', low_memory=False)

# Divises-en deux les années (2012-2018 et 2019-2022)
train_data_2012_2018, train_data_2019_2022 = divide_dataset(full_data_train)
test_data_2012_2018, test_data_2019_2022 = divide_dataset(full_data_test, False)

# On sauvergarde les fichiers
train_data_2012_2018.to_csv('TRAIN/train_data_2012_2018.csv', index=False)
train_data_2019_2022.to_csv('TRAIN/train_data_2019_2022.csv', index=False)
test_data_2012_2018.to_csv('TEST/test_data_2012_2018.csv', index=False)
test_data_2019_2022.to_csv('TEST/test_data_2019_2022.csv', index=False)

train_data_2012_2018 = pd.read_csv('TRAIN/train_data_2012_2018.csv', sep=',', low_memory=False)
train_data_2019_2022 = pd.read_csv('TRAIN/train_data_2019_2022.csv', sep=',', low_memory=False)
test_data_2012_2018 = pd.read_csv('TEST/test_data_2012_2018.csv', sep=',', low_memory=False)
test_data_2019_2022 = pd.read_csv('TEST/test_data_2019_2022.csv', sep=',', low_memory=False)

# Supprime les variables trop vides
feat_to_drop_2012_2018 = ['vma', 'motor', 'id_vehicule_x', 'id_vehicule_y',
                          'id_usager', 'secu3', 'secu2', 'secu1']
feat_to_drop_2019_2022 = ['env1', 'gps', 'secu']

# Drop feature 2012-2018
train_data_2012_2018 = drop_feature(feat_to_drop_2012_2018,
                                    train_data_2012_2018)
test_data_2012_2018 = drop_feature(feat_to_drop_2012_2018,
                                   test_data_2012_2018)

# Drop feature 2019-2022
train_data_2019_2022 = drop_feature(feat_to_drop_2019_2022,
                                    train_data_2019_2022)
test_data_2019_2022 = drop_feature(feat_to_drop_2019_2022,
                                   test_data_2019_2022)

train_data_2012_2018 = drop_feature('Num_Acc', train_data_2012_2018)
train_data_2019_2022 = drop_feature('Num_Acc', train_data_2019_2022)

# Traitement des données
train_data_2012_2018 = treatment_2012_2018(train_data_2012_2018)
test_data_2012_2018 = treatment_2012_2018(test_data_2012_2018)

train_data_2019_2022 = treatment_2019_2022(train_data_2019_2022)
test_data_2019_2022 = treatment_2019_2022(test_data_2019_2022)

# Supprime les lignes vides de certaines colonnes (car vides et très peu
# représentatives)
train_data_2012_2018 = drop_na_row(train_data_2012_2018, True)
train_data_2019_2022 = drop_na_row(train_data_2019_2022, False)

# On sauvegarde les fichiers
train_data_2012_2018.to_csv('TRAIN_FILTERED/train_data_2012_2018.csv', index=False)
train_data_2019_2022.to_csv('TRAIN_FILTERED/train_data_2019_2022.csv', index=False)
test_data_2012_2018.to_csv('TEST_FILTERED/test_data_2012_2018.csv', index=False)
test_data_2019_2022.to_csv('TEST_FILTERED/test_data_2019_2022.csv', index=False)

train_data_2012_2018 = pd.read_csv('TRAIN_FILTERED/train_data_2012_2018.csv', sep=',', low_memory=False)
train_data_2019_2022 = pd.read_csv('TRAIN_FILTERED/train_data_2019_2022.csv', sep=',', low_memory=False)
test_data_2012_2018 = pd.read_csv('TEST_FILTERED/test_data_2012_2018.csv', sep=',', low_memory=False)
test_data_2019_2022 = pd.read_csv('TEST_FILTERED/test_data_2019_2022.csv', sep=',', low_memory=False)

# Gestion des types
(train_data_2012_2018, train_data_2019_2022, test_data_2012_2018,
 test_data_2019_2022) = parse_to_categorical(
    train_data_2012_2018, train_data_2019_2022, test_data_2012_2018,
    test_data_2019_2022)

feat_numerical = ['occutc', 'lat', 'long', 'lartpc', 'larrout']

for feat in feat_numerical:
    if train_data_2019_2022[feat].dtype == 'float64':
        continue
    train_data_2019_2022[feat] = (
        train_data_2019_2022[feat].str.replace(',', '.').astype(float))
    if feat in test_data_2019_2022.columns:
        test_data_2019_2022[feat] = (
            test_data_2019_2022[feat].str.replace(',', '.').astype(float))

pred_2012_2018 = lightgbm(train_data_2012_2018, test_data_2012_2018)
pred_2019_2022 = lightgbm(train_data_2019_2022, test_data_2019_2022)

pred_2012_2022 = pd.concat([pred_2012_2018, pred_2019_2022], axis=0)
pred_2012_2022 = pred_2012_2022.groupby('Num_Acc')['GRAVE'].mean().reset_index()
pred_2012_2022.to_csv('TEST_EVAL/LGBM_27_04_SCRIPT.csv',
                      index=False, header=True)