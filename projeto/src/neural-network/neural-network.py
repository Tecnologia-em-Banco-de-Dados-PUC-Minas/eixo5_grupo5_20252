# %%

import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer
from tensorflow import keras
from tensorflow.keras import callbacks
import keras_tuner as kt

# %%

# Constante de pequeno valor para evitar divisão por zero (termo de estabilidade)
eps = 1e-6

# Lê os dados do csv que contem os caminhos das imagens e resultados de biomassa coletados em campo
df_path_by_sample = pd.read_csv('../data/path_img_by_lab_sample.csv')

# Transforma os caminhos relativos em completos
df_path_by_sample['relative_path'] = df_path_by_sample['relative_path'].map(lambda path: f'../data/img_mobile/{path}')

# Seleciona apenas as colunas de interesse
df_path_by_sample = df_path_by_sample[['relative_path','leaf_mass']]

df_path_by_sample

# %%

def cut_img(crop_size,path_img):
  img = io.imread(path_img)
  center_y, center_x = img.shape[0]//2, img.shape[1]//2
  half = crop_size//2
  return img[center_y - half:center_y + half, center_x - half:center_x + half]


def summarize_by_index(arr, prefix):
    arr = arr.flatten()
    stats = {
        f"{prefix}_mean": np.mean(arr),
        f"{prefix}_std": np.std(arr),
    }
    stats.update({f"{prefix}_quantil_{q}": np.percentile(arr, q) for q in range(20, 100, 20)})
    return stats

def processing_img_and_calc_indexs(row,crop_size):
    # Processamento inicial da imagem
    img_block = cut_img(
       crop_size=crop_size,
       path_img=row['relative_path']
    )

    # Separação das bandas
    R, G, B = [img_block[:, :, i].astype(np.float64) for i in range(3)]

    # Definição dos índices
    indexs = {
        "GLI": (2*G - R - B) / (2*G + R + B + eps),
        "MPRI": (G - R) / (G + R + eps),
        "ExG": (2*G - R - B) / (R + G + B + eps),
        "ExR": (1.4*R - G) / (R + G + B + eps),
        "ExB": (1.4*B - G) / (R + G + B + eps),
    }

    # Juncao dos resultados de cada índice com o calculo das estatisticas
    statistics_by_index = {}
    for index_name, index_list in indexs.items():
        statistics_by_index.update(summarize_by_index(index_list, index_name))

    return pd.Series(statistics_by_index)

def create_df_with_img_statistics(crop_size):
   # Calcula os indices GLI, MPRI, ExG, ExR, ExB
   df_index_statistics = df_path_by_sample.copy()
   df_index_statistics= df_index_statistics.join(
       df_index_statistics.apply(
          processing_img_and_calc_indexs,
          args=(crop_size,),
          axis=1
        )
   )
   
   # Seleciona apenas as colunas de interesse
   df_index_statistics = df_index_statistics[df_index_statistics.columns[1:]]
   return df_index_statistics

crop_size = 240 
df_index_statistics = create_df_with_img_statistics(crop_size)
df_index_statistics

# %%

# Define as features e target
X = df_index_statistics.copy()
y = X.pop('leaf_mass').values.reshape(-1, 1)

# Define as transformacoes que serao realizados nos dados
preprocessor_X = make_column_transformer(
    (StandardScaler(),X.columns.to_list())
)
preprocessor_y = MinMaxScaler()

# Separacao do conjunto de dados de treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                random_state=42,
                                                                test_size=0.20,
                                                                stratify=y
                                                                )

# Aplica as transformacoes nos dados de entrada e saida
X_train = preprocessor_X.fit_transform(X_train)
X_test = preprocessor_X.transform(X_test)
y_train = preprocessor_y.fit_transform(y_train)
y_test = preprocessor_y.transform(y_test)

# %%

def build_model(hp):
    # Define o tipo de modelo
    model = keras.Sequential()

    # Define o número de camadas ocultas
    for i in range(hp.Int('num_layers', min_value=1, max_value=hp_max_layers)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_layer_{i}', min_value=16, max_value=1024, step=32),
            activation='relu'
        ))
    
    # Define o neuronio de saida
    model.add(keras.layers.Dense(1))

    # Define a funcao de otimizacao e a funcao de erro
    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model

results_by_model = []

# Define a condicao de parada do treino do modelo
early_stopping = callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)


# Define o numero de camadas de neuronios de cada treinamento de grupo de modelos
for max_layers in range(1, 11): 
    print(f"\n=== Rodando tuner com max_layers={max_layers} ===")

    # variavel que armazena o numero maximo de camadas
    hp_max_layers = max_layers
    
    # Cria uma busca aleatoria para testar combinações de hiperparâmetros do modelo
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=20,
        executions_per_trial=1,
        directory='/saves/',
        project_name=f'layers_{max_layers}'
    )

    # Define quais os dados e quantas iteracoes seram utilizados na busca aleatoria
    tuner.search(
        X_train, y_train,
        epochs=200,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    # Recupera os modelos treinados
    best_model_list = tuner.get_best_models(num_models=50)
    print(len(best_model_list))

    for model in best_model_list:
        # Previsões com os dados de teste utilizando o modelo
        y_pred = model.predict(X_test, verbose=0)
        y_pred_orig = preprocessor_y.inverse_transform(y_pred)
        y_test_orig = preprocessor_y.inverse_transform(y_test)

        # Métrica de treino do modelo
        r2 = metrics.r2_score(y_test_orig, y_pred_orig)
        
        # Calcula do numero medio de neuronios por camada
        hidden_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense) and layer.units != 1]
        mean_num_neurons = np.mean([layer.units for layer in hidden_layers])

        # Salva os dados de cada modelo
        results_by_model.append({
            "max_layers": max_layers,
            "r2": r2,
            "mean_num_neu": mean_num_neurons
        })

#%%

# Carrega os resultados do treinamento em um dataframe
df_result_by_model = pd.DataFrame(results_by_model)

# Grafico que mostra a perfomace de cada modelo pelo numero de camadas
plt.figure(dpi=400)
plt.scatter(data=df_result_by_model, x="max_layers", y="r2", alpha=0.7)
plt.title("R² por modelo vs numero de camadas")
plt.xlabel("Numero de camadas")
plt.ylabel("R²")
plt.savefig(f"../metrics/neural-network/graphic_regression_scatter_r2_layers.png", dpi=400, bbox_inches="tight")
plt.show()
plt.close()

# Grafico que mostra a perfomace de cada modelo pela media de neuronios
plt.figure(dpi=400)
plt.scatter(data=df_result_by_model, x="mean_num_neu", y="r2", alpha=0.7)
plt.title("R² por modelo vs número médio de neurônios")
plt.xlabel("Numero médio de neurônios")
plt.ylabel("R²")
plt.savefig(f"../metrics/neural-network/graphic_regression_scatter_r2_mean_neu.png", dpi=400, bbox_inches="tight")
plt.show()
plt.close()
