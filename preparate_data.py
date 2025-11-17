
from glob import glob
import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets



def copy_organize_dataset(base_path, base_target):
    for damage in os.listdir(base_path):
        path_damage = os.path.join(base_path, damage)
        for condition in os.listdir(path_damage):
            path_damage_condition = os.path.join(path_damage, condition)
            for leaf_face in os.listdir(path_damage_condition):
                print(leaf_face)
                path_damage_condition_leaf_face = os.path.join(path_damage_condition, leaf_face)
                
                dir_damage_target = os.path.join(base_target, f"{damage.lower()}-{leaf_face}")
                
                os.makedirs(dir_damage_target, exist_ok=True)
                for file_image in tqdm(glob(f"{path_damage_condition_leaf_face}/**/*.jpg"), desc= f"Images {damage} {leaf_face}"):
                    base_name = os.path.basename(file_image)
                    target = os.path.join(dir_damage_target, base_name)
                    shutil.copy(file_image, target)

def count_images_dataset(base_path):
    exts = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".bmp")
    conteo = {}
    for carpeta in os.listdir(base_path):
        path = os.path.join(base_path, carpeta)
        if not os.path.isdir(path):
            continue
        
        # Contar imágenes dentro
        total = sum(1 for f in os.listdir(path) if f.lower().endswith(exts))
        conteo[carpeta] = total

    # Mostrar resultados
    for carpeta, total in conteo.items():
        print(f"{carpeta}: {total} imágenes")



def split_datset(data_dir, output_dir):
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    # Crear carpetas si no existen
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Cargar dataset completo
    dataset = datasets.ImageFolder(root=data_dir)
    samples = dataset.samples          # lista de (ruta, label)
    classes = dataset.classes          # nombres de las clases

    # Crear subcarpetas de clases en train/test
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir,   cls), exist_ok=True)

    # Extraer etiquetas
    labels = [label for _, label in samples]

    # División estratificada
    train_idx, test_idx = train_test_split(
        range(len(samples)),
        test_size=0.25,
        stratify=labels,
        random_state=42
    )

    return train_idx, test_idx, samples, classes
# --- Copiar archivos ---
def copiar_indices(indices, destino, samples, classes):
    for idx in tqdm(indices, "Copiando:"):
        source_path, label = samples[idx]
        class_name = classes[label]

        dest_path = os.path.join(destino, class_name)

        shutil.copy2(source_path, dest_path)

    print(f"Copiados {len(indices)} archivos en {destino}")

base_path = "./datos/imagenes_filtradas"
base_target = "./datos/dataset_daño_hojas_palta"

copy_organize_dataset(base_path, base_target)
count_images_dataset(base_target)

# Ruta donde están tus datos originales
data_dir = './datos/dataset_daño_hojas_palta'

# Carpeta donde guardarás los nuevos splits
output_dir = './datos/dataset_split'
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
print("Dividir datos en train an test")
train_idx, test_idx, samples, classes = split_datset(data_dir, output_dir,)
# Copiar train y test
copiar_indices(train_idx, train_dir, samples, classes)
copiar_indices(test_idx, test_dir, samples, classes)

print("✓ División terminada correctamente.")

