# Importaciones
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.models import mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from TriAttentionArchitectura import WoodClassifierWithTriAttention
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Metricas estadisticas avanzadas
from scipy.stats import mannwhitneyu, kruskal
import scikit_posthocs as sp

# Rutas
path = "/home/daniel/Imágenes/Camera/TODO"
metadata_file = 'metadata.csv'
consideracion=["Con atencion", "Sin atencion"]
indice = 0 # Controla el modelo a evaluar (modificar entre 0 o 1 según la variable "consideraciones")
model_path="Modelo_E12__CON_aumento_sin_tri.pt"
model_path_attention="Modelo_E12__CON_aumento_con_tri.pt"

# Configuración
CLASS_NAMES = {
    0: 'Cedro',  # Cedro (ajusta el nombre completo)
    1: 'Faique',  # Balsa
    2: 'Guayacán',  # Higuerón
    3: 'Nogal'   # Jacaranda
}
class_names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
# Asegurar reproducibilidad
np.random.seed(42)

# Adquisición de transformaciones
transforms = MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms()

# Lectura del metadatos y creación de df
df = pd.read_csv(os.path.join(path, metadata_file), sep=',') # Path image
df = df.rename(columns={"file_name": "image"})
df['image'] = df['image'].apply(lambda x: os.path.join(path, x))
df = df.reset_index(drop=True)

# Creación de Dataset
class LanscapeDataset(torch.utils.data.Dataset):
    """
    Dataset personalizado para clasificación de madera
    """
    def __init__(self, dataframe, transforms=None):
        """
        Args:
            dataframe: DataFrame con columnas 'image' (ruta) y 'label' (clase)
            transform: Transformaciones de torchvision a aplicar
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transforms = transforms
    
    def __len__(self):
        """Retorna el número total de imágenes"""
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Retorna una muestra del dataset
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            image: Tensor de la imagen transformada
            label: Etiqueta de la clase
        """
        # Obtener la ruta de la imagen y la etiqueta
        img_path = self.dataframe.loc[idx, 'image']
        label = self.dataframe.loc[idx, 'label']
        
        # Cargar la imagen
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error cargando imagen {img_path}: {e}")
            # Retornar una imagen negra en caso de error
            image = Image.new('RGB', (224, 224), color='black')
        
        # Aplicar transformaciones si existen
        if self.transforms:
            image = self.transforms(image)
        
        return image, label


# Creación del Dataset
test_dataset = LanscapeDataset(dataframe=df, transforms=transforms)
# Creación del Dataloader
test_loader = DataLoader(
    test_dataset,
    batch_size=32,       
    shuffle=False,       
    num_workers=2,       
)
print(f"\n✓ Dataset de test creado: {len(test_dataset)} imágenes")
print(f"✓ Test loader: {len(test_loader)} batches de tamaño 32")

# Carga del modelo
def cargar_modelo(path, tri=False):
    try:
        model = WoodClassifierWithTriAttention(num_classes=4, use_tri_attention=tri)

        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        print("✓ Modelo cargado exitosamente")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

model = cargar_modelo(model_path, tri=False) # Sin atención 
# model = cargar_modelo(model_path_attention, tri=True) # Con atención

# Seleccionar device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Test
def calcular_metricas(all_labels, all_preds, matrix=False):
    # Calcular métricas
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro')  # 'macro', 'micro', 'weighted'
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # --- Matriz de confusión ---
    if matrix:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names) 
        disp.plot(cmap="Blues")
        # Guardar grágica
        path = os.path.join("Graficas/", consideracion[indice] + "_Matriz.jpg")
        plt.savefig(path, format='png')  # Guarda como PNG
        print(f"Gráfico guardado como: {path}")
    
    return f1, recall, precision, accuracy


def test(model_local, loader):
    model_local.eval()  # Se cambia a modo evaluación
    running_corrects = 0

    all_labels = []
    all_preds = []

    # Se desactiva la gradiante. Puesto que es Test
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

                # Forward pass
            # Ejecución del modelo y predicciones
            outputs = model_local(inputs)
            _, preds = torch.max(outputs, 1)

            # Acumular todos los labels y predicciones
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds

# =============================================================
# Métricas Estándar
# =============================================================
# Ejecutar función de test
print("-"*20)
all_labels, all_preds = test(model, test_loader)
calcular_metricas(all_labels, all_preds, matrix=True)
print("-"*20)
print()

# Con atención
# Accuracy: 0.7241
# Precision: 0.7961
# Recall: 0.7241
# F1-score: 0.6666

# Sin atención
# Accuracy: 0.8103
# Precision: 0.8260
# Recall: 0.8103
# F1-score: 0.8062


# =============================================================
# PREPARACIÓN
# =============================================================
# Clases disponibles
clases = df['label'].unique()
num_subsets = 6
imgs_por_clase_en_subset = 5
subset_size = imgs_por_clase_en_subset * len(clases)

# 1. Separar por clase
df_por_clase = {c: df[df['label'] == c].sample(frac=1, random_state=42).reset_index(drop=True)
                for c in clases}

# 2. Crear bloques balanceados por clase
bloques_por_clase = {
    c: np.array_split(df_por_clase[c], num_subsets)
    for c in clases
}

# 3. Construir los 6 subconjuntos balanceados
subsets = []
for i in range(num_subsets):
    subset_i = pd.concat([bloques_por_clase[c][i] for c in clases]).sample(frac=1).reset_index(drop=True)
    subsets.append(subset_i)

# Mostrar tamaños
# for i, s in enumerate(subsets):
#     print(f"Subset {i+1}: {len(s)} imágenes")
#     print(s['label'].value_counts())
#     print("-"*40)



# ************************************
# Mann Whitney
# ************************************
modelo_sin_atencion = cargar_modelo(model_path, tri=False)
modelo_con_atencion = cargar_modelo(model_path_attention, tri=True)

f1_con_atencion = []
f1_sin_atencion = []

for i, subset in enumerate(subsets):
    subset_mini_dataset = LanscapeDataset(dataframe=subset, transforms=transforms)
    dataloader_subset = DataLoader(subset_mini_dataset, shuffle=False)
    print(dataloader_subset)

    all_labels_A, all_preds_A = test(modelo_con_atencion, dataloader_subset)
    all_labels_B, all_preds_B = test(modelo_sin_atencion, dataloader_subset)

    f1A, _, _, _ = calcular_metricas(all_labels_A, all_preds_A, matrix=False)
    f1B, _, _, _ = calcular_metricas(all_labels_B, all_preds_B, matrix=False)

    f1_con_atencion.append(f1A)
    f1_sin_atencion.append(f1B)

    print("F1 del modelo con atención:", f1_con_atencion)
    print("F1 del modelo sin atención:", f1_sin_atencion)

# Aplicación de Mann whitney
stat, p = mannwhitneyu(f1_con_atencion, f1_sin_atencion, alternative='greater')

print("Estadístico U:", stat)
print("p-value:", p)


# ************************************
# Kruskal-Wallis
# ************************************
f1_por_clase = {c: [] for c in clases}
f1_por_clase_atencion = {c: [] for c in clases}

for i, subset in enumerate(subsets):
    subset_mini_dataset = LanscapeDataset(dataframe=subset, transforms=transforms)
    dataloader_subset = DataLoader(subset_mini_dataset, shuffle=False)

    all_labels_kruskal_A, all_preds_kruskal_A = test(modelo_sin_atencion, dataloader_subset)
    all_labels_kruskal_B, all_preds_kruskal_B = test(modelo_con_atencion, dataloader_subset)

    # F1 por clase para este subgrupo
    f1A = f1_score(all_labels_kruskal_A, all_preds_kruskal_A, average=None, labels=clases)
    f1B = f1_score(all_labels_kruskal_B, all_preds_kruskal_B, average=None, labels=clases)

    for idx, c in enumerate(clases):
        f1_por_clase[c].append(f1A[idx])
        f1_por_clase_atencion[c].append(f1B[idx])

print(f1_por_clase)
print(f1_por_clase_atencion)


grupos = [f1_por_clase[c] for c in clases]
grupos_atencion = [f1_por_clase_atencion[c] for c in clases]

stat, p = kruskal(*grupos)
stat_atencion, p_atencion = kruskal(*grupos_atencion)

print("Estadístico H de Kruskal-Wallis:", stat)
print("p-value:", p)
print("Estadístico H de Kruskal-Wallis (Atencion):", stat_atencion)
print("p-value (Stencion):", p_atencion)



# DUNN TEST
# Para modelo sin atención
data_long_sin = pd.DataFrame({
    "F1":  sum([f1_por_clase[c] for c in clases], []),
    "Clase": sum([[c]*len(f1_por_clase[c]) for c in clases], [])
})

# Para modelo con atención
data_long_atencion = pd.DataFrame({
    "F1":  sum([f1_por_clase_atencion[c] for c in clases], []),
    "Clase": sum([[c]*len(f1_por_clase_atencion[c]) for c in clases], [])
})

# -----------------------------
# Dunn test con corrección Bonferroni
# -----------------------------

# Modelo sin atención
dunn_sin = sp.posthoc_dunn(data_long_sin, val_col='F1', group_col='Clase', p_adjust='bonferroni')
print("Dunn test (sin atención):")
print(dunn_sin)

# Modelo con atención
dunn_atencion = sp.posthoc_dunn(data_long_atencion, val_col='F1', group_col='Clase', p_adjust='bonferroni')
print("\nDunn test (con atención):")
print(dunn_atencion)

# Modelo sin atención
data_sin = pd.DataFrame({
    "F1": sum([f1_por_clase[c] for c in clases], []),
    "Clase": sum([[c]*len(f1_por_clase[c]) for c in clases], []),
    "Modelo": "Sin Atención"
})

# Modelo con atención
data_atencion = pd.DataFrame({
    "F1": sum([f1_por_clase_atencion[c] for c in clases], []),
    "Clase": sum([[c]*len(f1_por_clase_atencion[c]) for c in clases], []),
    "Modelo": "Con Atención"
})

# Concatenar ambos
data_plot = pd.concat([data_sin, data_atencion], axis=0)
# data_plot = pd.concat([data_atencion], axis=0)
data_plot["ClaseNombre"] = data_plot["Clase"].map(CLASS_NAMES)

# -----------------------------
# Generar el boxplot
# -----------------------------
plt.figure(figsize=(10,6))

# Boxplot
sns.boxplot(
    x="ClaseNombre", 
    y="F1", 
    hue="Modelo", 
    data=data_plot,
    palette=["#FF9999","#66B2FF"]
    # palette=["#FF9999"]
)

# Swarmplot con colores distintos según modelo
palette_swarm = {"Sin Atención": "#FF6666", "Con Atención": "#3399FF"}  # tonos más visibles
# palette_swarm = {"Con Atención": "#3399FF"}  # tonos más visibles
sns.swarmplot(
    x="ClaseNombre", 
    y="F1", 
    hue="Modelo", 
    data=data_plot, 
    dodge=True, 
    palette=palette_swarm,
    size=7,      # tamaño de los puntos
    alpha=0.9
)

# Evitar doble leyenda
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title="Modelo")
plt.legend([], [], frameon=False) # sin labels

# Ajustes finales
plt.title("Comparación de F1-score por clase entre modelos", fontsize=16)
plt.ylabel("F1-score", fontsize=12)
plt.xlabel("Clase de madera", fontsize=12)
plt.ylabel("Precisión", fontsize=12)
plt.xlabel("Especies", fontsize=12)
plt.ylim(0,1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# path = os.path.join("Graficas/" + "boxplot_kruskel_solo_atencion.jpg")
path = os.path.join("Graficas/" + "boxplot_kruskel.jpg")
plt.savefig(path, format='png')  # Guarda como PNG
print(f"Gráfico guardado como: {path}")