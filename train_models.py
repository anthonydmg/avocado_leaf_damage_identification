import torch
from sklearn.metrics import f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def load_datasets(train_dir, val_dir, batch_size):
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]


    # Transformaciones para ENTRENAMIENTO (con Data Augmentation)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_nums, std=std_nums)
    ])

    # Transformaciones para VALIDACIÓN (sin Data Augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(INPUT_SIZE + 32),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_nums, std=std_nums)
    ])

    
    dataset_for_train = datasets.ImageFolder(
        root=train_dir,
        transform=train_transforms  # Aplicará augmentation
    )

    dataset_for_val = datasets.ImageFolder(
        root=val_dir,
        transform=val_transforms    # Aplicará solo el recorte/normalización
    )

    train_set = dataset_for_train

    val_set = dataset_for_val

    # Crear los DataLoaders
    dataloaders = {
        'train': DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=True,  # Mezclar entrenamiento
            num_workers=4
        ),
        'val': DataLoader(
            val_set, 
            batch_size=batch_size, 
            shuffle=False, # No mezclar validación
            num_workers=4
        )
    }

    return dataloaders, (train_set, val_set)



def load_model(num_classes, device):
    # --- 2. CARGA DEL MODELO (EFFICIENTNETV2-S) ---
    print("Cargando modelo EfficientNetV2-S pre-entrenado...")

    # Cargar pesos pre-entrenados
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)

    # Congelar todas las capas base
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar el clasificador final
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)
    print("Modelo listo para transfer learning.")
    print("-" * 30)

    return model



# --- 4. BUCLE DE ENTRENAMIENTO ---

def train_model(model, num_epochs):
    
    # Función de pérdida
    criterion = nn.CrossEntropyLoss()

    # Filtramos los parámetros para asegurar que solo entrenamos los que
    # tienen requires_grad = True (es decir, nuestra nueva capa)
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=0.001)

    # Programador de tasa de aprendizaje (Learning Rate Scheduler)
    # Reduce el LR cada 7 épocas en un factor de 0.1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    start_time = time.time()

    # Guardar los mejores pesos del modelo
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = dict(train_accuracy = [], 
                   train_loss = [], 
                   train_f1 = [],
                   val_accuracy = [],
                   val_loss = [],
                   val_f1 = [])

    for epoch in range(num_epochs):
        print(f'Época {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Cada época tiene una fase de entrenamiento y una de validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Poner el modelo en modo entrenamiento
            else:
                model.eval()   # Poner el modelo en modo evaluación

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            progress_bar = tqdm(dataloaders[phase], desc = f"Epochs {epoch}")
            # Iterar sobre los datos
            for inputs, labels in progress_bar:
                # Mover datos a la GPU/CPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Poner a cero los gradientes del optimizador
                optimizer.zero_grad()

                # Forward pass
                # Rastrear historial solo si es 'train'
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # Obtener la clase predicha
                    loss = criterion(outputs, labels)

                    # Backward pass + optimizar solo si es 'train'
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Estadísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # Para F1
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({"loss": loss.item()})
            
            # Actualizar el scheduler si estamos en la fase de 'train'
            if phase == 'train':
                scheduler.step()

            # Calcular pérdida y precisión de la época
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # F1 macro
            epoch_f1 = f1_score(all_labels, all_preds, average="macro")

            # Guardar historial
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_accuracy"].append(epoch_acc)
            history[f"{phase}_f1"].append(epoch_f1)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} "
                  f"Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

            # Guardar el modelo si es el mejor hasta ahora (en validación)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'🎉 ¡Nuevo mejor modelo guardado con Acc: {best_acc:.4f}!')

        print() # Línea en blanco entre épocas

    # Fin del entrenamiento
    time_elapsed = time.time() - start_time
    print(f'Entrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Mejor Precisión (Val): {best_acc:.4f}')

    # Cargar los mejores pesos en el modelo
    model.load_state_dict(best_model_wts)
    return model, history

def plot_training_history(history, save_results_dir = "./results"):
    epochs = range(1, len(history["train_loss"]) + 1)
    os.makedirs(save_results_dir, exist_ok=True)

    # ===== Plot Loss =====
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_results_dir, "loss.png"))
    plt.show()
    plt.close()   # ← IMPORTANTE

    print(history["val_accuracy"])
    # ===== Plot Accuracy =====
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_accuracy"], label="Train")
    plt.plot(epochs, history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_results_dir, "accuracy.png"))
    plt.show()
    plt.close()   # ← IMPORTANTE

    print(history["val_f1"])
    # ===== Plot F1 Score =====
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_f1"], label="Train")
    plt.plot(epochs, history["val_f1"], label="Val")
    plt.title("F1 Score (Macro)")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_results_dir, "f1_score.png"))
    plt.show()
    plt.close()   # ← IMPORTANTE


if __name__ == '__main__':

    INPUT_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 20          # Número de épocas para entrenar
    SEED = 42                # Para división reproducible

    # Ruta de datos y parámetros
    # Ruta de datos ya divididos
    train_dir = './datos/dataset_split/train'
    val_dir   = './datos/dataset_split/test'   # usa test o val, como lo hayas nombrado

    dataloaders, (train_set, val_set) = load_datasets(train_dir, val_dir, batch_size=BATCH_SIZE)
    
    # --- 7. Verificación (Opcional) ---
    class_names = train_set.classes
    print(f"\nClases encontradas: {class_names}")
    print("¡DataLoaders listos! 🚀")
    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    num_classes = len(class_names)


    # Configurar dispositivo (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cargar Modelo
    model = load_model(num_classes, device)

    print(f"Usando dispositivo: {device} 🚀")
    print(f"Clases encontradas ({num_classes}): {class_names}")
    print(f"Imágenes de entrenamiento: {dataset_sizes['train']}")
    print(f"Imágenes de validación: {dataset_sizes['val']}")
    print("-" * 30)

    # EJECUTAR EL ENTRENAMIENTO 

    print("Iniciando el entrenamiento...")
    model_entrenado, history = train_model(model, num_epochs=NUM_EPOCHS)

    print("¡Entrenamiento finalizado!")
    plot_training_history(history)

    # Guardar el modelo final 
    ruta_modelo_guardado = "./efficientnetv2_s_final.pth"
    torch.save(model_entrenado.state_dict(), ruta_modelo_guardado)
    print(f"Modelo guardado en: {ruta_modelo_guardado}")