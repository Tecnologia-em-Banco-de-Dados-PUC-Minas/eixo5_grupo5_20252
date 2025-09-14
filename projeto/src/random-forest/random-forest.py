# %%

import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble
from sklearn import pipeline
from sklearn import metrics
from sklearn.preprocessing import label_binarize

# %%

# Constante de pequeno valor para evitar divisão por zero (termo de estabilidade)
eps = 1e-6

# Lê os dados do csv que contem os caminhos das imagens e resultados de biomassa coletados em campo
df_path_by_sample = pd.read_csv('../data/path_img_by_lab_sample.csv')

# Transforma os caminhos relativos em completos
df_path_by_sample['relative_path'] = df_path_by_sample['relative_path'].map(lambda path: f'../data/img_mobile/{path}')

# Seleciona apenas as colunas de interesse
df_path_by_sample = df_path_by_sample[['relative_path','leaf_mass']]

# Seleciona uma amostra de n imagens que possuem um mesmo valor de target 
# df_path_by_sample = df_path_by_sample.groupby('leaf_mass').sample(4).reset_index(drop=True)

df_path_by_sample

# %%

# Grafico da frequencia de imagens por resultado de leaf_mass
plt.figure(dpi=400)
plt.hist(df_path_by_sample['leaf_mass'], bins=15, color="blue", alpha=0.7, edgecolor="black")
plt.title("Distribuição de n imagens por resultado de leaf_mass")
plt.xlabel("leaf_mass")
plt.ylabel("Frequência")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Sumarizacao estatistica da coluna de leaf_mass
df_path_by_sample.describe()

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
    stats.update({f"{prefix}_quantil_{q}": np.percentile(arr, q) for q in range(10, 100, 10)})
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

# Regression

# Definicao das features e target
features = df_index_statistics.columns[1:]
target = 'leaf_mass'
X, y = df_index_statistics[features].copy(), df_index_statistics[target].copy()

# Separacao do conjunto de dados de treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                random_state=42,
                                                                test_size=0.20,
                                                                stratify=y
                                                                )

# Analise das melhores features
tree_features = tree.DecisionTreeRegressor(random_state=42)
tree_features.fit(X_train, y_train)
feature_importances = (pd.Series(tree_features.feature_importances_,index=X_train.columns)
                        .sort_values(ascending=False).reset_index())
feature_importances['acum.']=feature_importances[0].cumsum()
best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index'].tolist())

# Definicao do modelo que sera utilizado
model = ensemble.RandomForestRegressor(
    random_state=42,
    n_jobs=4
)

# Definicao dos hiperparâmetros que seram testados para avaliar a performace do modelo
params = {
    "min_samples_leaf":[15,20,25,30,50],
    "n_estimators":[100,200,500,1000],
    "criterion":['squared_error', 'absolute_error', 'poisson'],
}

# Cria uma busca em grade (Grid Search) para testar combinações de hiperparâmetros do modelo
grid = model_selection.GridSearchCV(model,params,cv=3,scoring='r2',verbose=3)

# Cria um pipeline de processamento para o modelo
model_pipeline = pipeline.Pipeline(
    steps=[('Grid',grid)]
)

# Treina o pipeline usando apenas as melhores features selecionadas
model_pipeline.fit(X_train[best_features], y_train)

# Previsões utilizando o melhor modelo
y_train_pred = model_pipeline.predict(X_train[best_features])
y_test_pred = model_pipeline.predict(X_test[best_features])

# Métricas de treino do melhor modelo
r2_train = metrics.r2_score(y_train, y_train_pred)
rmse_train = metrics.root_mean_squared_error(y_train, y_train_pred)
mae_train = metrics.mean_absolute_error(y_train, y_train_pred)

# Métricas de teste do melhor modelo
r2_test = metrics.r2_score(y_test, y_test_pred)
rmse_test = metrics.root_mean_squared_error(y_test, y_test_pred)
mae_test = metrics.mean_absolute_error(y_test, y_test_pred)

# Grafico com os resultados das metricas de perfomace do melhor modelo
plt.figure(dpi=400)
plt.scatter(y_test, y_test_pred, alpha=0.5, color="blue", label="Teste")
plt.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        '--', color='red')
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title(f'RandomForestRegressor - Previsao x Real')
plt.grid(True)
plt.legend()
textstr = '\n'.join((
    f"Treino - R²: {r2_train:.3f} | RMSE: {rmse_train:.0f} | MAE: {mae_train:.0f}",
    f"Teste  - R²: {r2_test:.3f} | RMSE: {rmse_test:.0f} | MAE: {mae_test:.0f}"
))
plt.text(0.98, 0.02, textstr, transform=plt.gca().transAxes,
        fontsize=8, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

plt.savefig(f"../metrics/random-forest/graphic_regression_scatter_result.png", dpi=400, bbox_inches="tight")
plt.show()
plt.close()

# %%

# Binary Classification
# target < 3000 or target > 4500 -> Não ideal
# target < 4500 and target > 3000 -> Ideal 

# Criacao da coluna com as classes que seram utilizadas pelo modelo
df_index_statistics_classification = df_index_statistics.copy()
df_index_statistics_classification['target_class'] = df_index_statistics_classification['leaf_mass'].apply(
    lambda lf: 1 if (lf > 3000) and (lf < 4500) else 0
)
df_index_statistics_classification = df_index_statistics_classification.drop(columns=['leaf_mass'])
df_index_statistics_classification

# Definicao das features e target 
features = df_index_statistics_classification.columns[0:-1]
target = 'target_class'
X, y = df_index_statistics_classification[features].copy(), df_index_statistics_classification[target].copy()

# Separacao do conjunto de dados de treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                random_state=42,
                                                                test_size=0.20,
                                                                stratify=y,
                                                                )

# Analise das melhores features
tree_features = tree.DecisionTreeClassifier(random_state=42)
tree_features.fit(X_train, y_train)
feature_importances = (pd.Series(tree_features.feature_importances_,index=X_train.columns)
                        .sort_values(ascending=False).reset_index())
feature_importances['acum.']=feature_importances[0].cumsum()
best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index'].tolist())

# Definicao do modelo que sera utilizado
model = ensemble.RandomForestClassifier(
    random_state=42,
    n_jobs=4,
)

# Definicao dos hiperparâmetros que seram testados para avaliar a performace do modelo
params = {
    "min_samples_leaf":[15,20,25,30,50],
    "n_estimators":[100,200,500,1000],
    "criterion":['gini', 'entropy', 'log_loss'],
}

# Cria uma busca em grade (Grid Search) para testar combinações de hiperparâmetros do modelo
grid = model_selection.GridSearchCV(model,
                                    params,
                                    cv=3,
                                    scoring='roc_auc',
                                    verbose=4,
                                    )

# Cria um pipeline de processamento para o modelo
model_pipeline = pipeline.Pipeline(
    steps=[
        ('Grid',grid), 
    ]
)

# Treina o pipeline usando apenas as melhores features selecionadas
model_pipeline.fit(X_train[best_features], y_train)

# Previsões com os dados de treino utilizando o melhor modelo
y_train_predict = model_pipeline.predict(X_train[best_features])
y_train_proba = model_pipeline.predict_proba(X_train[best_features])[:,1]

# Métricas de treino do melhor modelo
acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
roc_train = metrics.roc_curve(y_train, y_train_proba)

# Previsões com os dados de teste utilizando o melhor modelo
y_test_predict = model_pipeline.predict(X_test[best_features])
y_test_proba = model_pipeline.predict_proba(X_test[best_features])[:,1]

# Métricas de teste do melhor modelo
acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
roc_test = metrics.roc_curve(y_test, y_test_proba)

# Grafico com os resultados das metricas de perfomace do melhor modelo em classificao binaria
plt.figure(dpi=400)
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot([0,1], [0,1], '--', color='black')
plt.grid(True)
plt.ylabel("Sensibilidade")
plt.xlabel("1 - Especificidade")
plt.title("Curva ROC")
textstr = '\n'.join((
    f"Treino - AUC: {auc_train:.3f} | Acc: {acc_train:.3f}",
    f"Teste  - AUC: {auc_test:.3f} | Acc: {acc_test:.3f}"
))
plt.text(0.98, 0.17, textstr, transform=plt.gca().transAxes,
         fontsize=9, va="bottom", ha="right",
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
plt.grid(True)
plt.legend([
    f"Treino: {100*auc_train:.2f}",
    f"Teste: {100*auc_test:.2f}"
])
plt.savefig(f"../metrics/random-forest/graphic_binary_classifition_roc_curve_result.png", dpi=400, bbox_inches="tight")
plt.show()
plt.close()

# %%

# Multiclass Classification
# target < 3000 -> Abaixo do ideal
# target < 4500 and target > 3000 -> Ideal 
# target > 4500 -> Acima do ideal

# Criação da coluna com as multiclasses que seram utilizadas pelo modelo
df_index_statistics_classification = df_index_statistics.copy()
df_index_statistics_classification['target_class'] = df_index_statistics_classification['leaf_mass'].apply(
    lambda lf: 0 if lf < 3000 else (1 if lf <= 4500 else 2)
)
df_index_statistics_classification = df_index_statistics_classification.drop(columns=['leaf_mass'])
df_index_statistics_classification['target_class'].value_counts()

# Definicao das features e target 
features = df_index_statistics_classification.columns[0:-1]
target = 'target_class'
X, y = df_index_statistics_classification[features].copy(), df_index_statistics_classification[target].copy()

# Separacao do conjunto de dados de treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, random_state=42, test_size=0.20, stratify=y
)

# Analise das melhores features
tree_features = tree.DecisionTreeClassifier(random_state=42)
tree_features.fit(X_train, y_train)
feature_importances = (pd.Series(tree_features.feature_importances_,index=X_train.columns)
                        .sort_values(ascending=False).reset_index())
feature_importances['acum.']=feature_importances[0].cumsum()
best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index'].tolist())

# Definicao do modelo que sera utilizado
model = ensemble.RandomForestClassifier(
    random_state=42,
    n_jobs=4,
)

# Definicao dos hiperparâmetros que seram testados para avaliar a performace do modelo
params = {
    "min_samples_leaf":[15,20,25,30,50],
    "n_estimators":[100,200,500,1000],
    "criterion":['gini', 'entropy', 'log_loss'],
}

# Cria uma busca em grade (Grid Search) para testar combinações de hiperparâmetros do modelo
grid = model_selection.GridSearchCV(
    model,
    params,
    cv=3,
    scoring="roc_auc_ovr",  # para multiclasse
    verbose=4,
)

# Cria um pipeline de processamento para o modelo
model_pipeline = pipeline.Pipeline(
    steps=[
        ('Grid', grid), 
    ]
)

# Treina o pipeline usando apenas as melhores features selecionadas
model_pipeline.fit(X_train[best_features], y_train)

# Previsões com os dados de treino utilizando o melhor modelo
y_train_predict = model_pipeline.predict(X_train[best_features])
y_train_proba = model_pipeline.predict_proba(X_train[best_features])

# Métricas de treino do melhor modelo
acc_train = metrics.accuracy_score(y_train, y_train_predict)
f1_train = metrics.f1_score(y_train, y_train_predict, average="macro")

# Previsões com os dados de teste utilizando o melhor modelo
y_test_predict = model_pipeline.predict(X_test[best_features])
y_test_proba = model_pipeline.predict_proba(X_test[best_features])

# Métricas de teste do melhor modelo
acc_test = metrics.accuracy_score(y_test, y_test_predict)
f1_test = metrics.f1_score(y_test, y_test_predict, average="macro")

# Realizar uma Binarizacao das classes (necessário para ROC multiclasse)
classes = sorted(y.unique())
y_train_bin = label_binarize(y_train, classes=classes)
y_test_bin = label_binarize(y_test, classes=classes)

# Separacao dos dados da curva roc de cada classe
fpr = dict()
tpr = dict()
roc_auc = dict()
for i, c in enumerate(classes):
    fpr[c], tpr[c], _ = metrics.roc_curve(y_test_bin[:, i], y_test_proba[:, i])
    roc_auc[c] = metrics.auc(fpr[c], tpr[c])

print('frp:',fpr)
print('tpr:',tpr)
print('roc_curve:',roc_auc)

# Grafico com os resultados das metricas de perfomace do melhor modelo em classificao multiclasse
plt.figure(dpi=400)
for c in classes:
    plt.plot(fpr[c], tpr[c], label=f"Classe {c} (AUC={roc_auc[c]:.2f})")
plt.plot([0,1],[0,1], '--', color='black')
plt.xlabel("1 - Especificidade")
plt.ylabel("Sensibilidade")
plt.title("Curvas ROC Multiclasse (One-vs-Rest)")
textstr = '\n'.join((
    f"Treino - ACC: {acc_train:.3f} | F1: {f1_train:.3f}",
    f"Teste  - ACC: {acc_test:.3f} | F1: {f1_test:.3f}"
))
plt.text(0.98, 0.25, textstr, transform=plt.gca().transAxes,
         fontsize=9, va="bottom", ha="right",
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
plt.grid(True)
plt.legend()
plt.savefig(f"../metrics/random-forest/graphic_multiclass_classification_roc_curve_result.png", dpi=400, bbox_inches="tight")
plt.show()
