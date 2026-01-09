import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from PIL import Image
import os
import copy
from tqdm import tqdm
import pandas as pd
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torchvision import tv_tensors
import seaborn as sns
# ==========================================
# 1. DATASETS Y TRANSFORMACIONES
# ==========================================

# Definimos transformaciones. 
# Nota: Usualmente en Train se agrega 'RandomHorizontalFlip' o rotaciones,
# pero en Val solo se redimensiona y normaliza.
#train_transforms = transforms.Compose([
#    transforms.Resize((512, 512)),
#    transforms.RandomHorizontalFlip(p=0.5), # Aumentación de datos simple
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#])


train_transforms = v2.Compose([
    v2.Resize((512, 512)),
    
    # --- GEOMÉTRICAS (Se aplican a IMAGEN y MÁSCARA) ---
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    
    # --- FOTOMÉTRICAS (v2 las aplicará SOLO a la IMAGEN automáticamente) ---
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),

    # --- FINALIZACIÓN Y DTYPES (Punto Crítico) ---
    v2.ToImage(),
    
    # Aquí definimos DTypes distintos para cada uno:
    v2.ToDtype({
        tv_tensors.Image: torch.float32, 
        tv_tensors.Mask: torch.int64,   # La máscara se queda como enteros (Long)
    }, scale=True), 
    
    # Normalize detectará que la Mask es un tv_tensor y NO la tocará.
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#val_transforms = transforms.Compose([
#    transforms.Resize((512, 512)),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#])

val_transforms = v2.Compose([
    # Redimensiona ambos (imagen y máscara) a 512x512
    # v2 usará Bilinear para la imagen y Nearest para la máscara automáticamente
    v2.Resize((512, 512)), 
    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        # Asegúrate de que la lógica de reemplazo sea exacta para tu caso
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_MASK_LONG.npy"))

        # 1. Cargar Imagen y Máscara
        image = Image.open(img_path).convert("RGB")
        mask_np = np.load(mask_path) 
        
        # Convertimos la máscara a PIL para que la v2 la maneje correctamente
        mask = Image.fromarray(mask_np.astype(np.uint8)) 
        mask = tv_tensors.Mask(mask) # <-- ESTA ES LA CLAVE
        # 2. Aplicar transformaciones sincronizadas
        if self.transform:
            # ¡AQUÍ ESTÁ EL CAMBIO! Pasamos ambos a la vez
            image, mask = self.transform(image, mask)
        # 3. Asegurar que la máscara sea LongTensor para la función de pérdida
        # Si la transformación v2 ya la convirtió a tensor, solo hacemos el cast
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask))
            
        return image, mask.long()

# ==========================================
# 2. FUNCIONES DE ENTRENAMIENTO
# ==========================================

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    # 1. Crear la carpeta si no existe
    output_dir = "./results_seg"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Carpeta creada: {output_dir}")

    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # Diccionario para guardar métricas
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_pixels = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} {phase}")
            
            for images, masks in progress_bar:
                images = images.to(device)
                masks = masks.to(device)

                # Si la máscara tiene forma [B, 1, H, W], quitamos el 1
                if len(masks.shape) == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    
                    if phase == 'train':
                        loss_main = criterion(outputs['out'], masks)
                        loss_aux = criterion(outputs['aux'], masks)
                        loss = loss_main + (0.4 * loss_aux)
                    else:
                        loss = criterion(outputs['out'], masks)
                    
                    preds = torch.argmax(outputs['out'], dim=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                progress_bar.set_postfix({"loss": loss.item()})
                
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == masks.data)
                total_pixels += torch.numel(masks.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / total_pixels

            # 2. Guardar métricas en el historial
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                ruta_modelo = os.path.join(output_dir, "mejor_modelo_deeplab_mobilenetv3.pth")
                torch.save(model.state_dict(), ruta_modelo)
                print(f"  -> Modelo guardado en {ruta_modelo}")

    # 3. Generar y guardar los gráficos al finalizar todas las épocas
    plt.figure(figsize=(12, 5))

    # Gráfico de Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='Val Loss')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    # Gráfico de Precisión (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), history['train_acc'], label='Train Acc')
    plt.plot(range(1, num_epochs + 1), history['val_acc'], label='Val Acc')
    plt.title('Precisión (Pixel Accuracy)')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    # Guardar la figura
    plt.tight_layout()
    grafico_path = os.path.join(output_dir, "entrenamiento_graficos.png")
    plt.savefig(grafico_path)
    plt.show()
    print(f"Gráficos guardados en: {grafico_path}")

    print(f'Mejor Val Loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def get_deeplab_model(num_classes):
    print("🏗️ Cargando DeepLabV3 con MobileNetV3 Large...")
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
    
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    if model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    return model

# ==========================================
# 3. NUEVAS FUNCIONES DE EVALUACIÓN FINAL
# ==========================================

def fast_hist(a, b, n):
    """Calcula la matriz de confusión para un batch (optimizado)"""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def evaluate_final_test(model, dataloader, device, num_classes, save_path, class_names=None):
    """
    Ejecuta evaluación completa, guarda métricas en CSV/TXT y genera 
    un gráfico de la matriz de confusión normalizada.
    """
    # 1. Crear carpeta de destino
    os.makedirs(save_path, exist_ok=True)

    model.eval()
    model.to(device)
    # Matriz de conteos crudos
    hist = np.zeros((num_classes, num_classes))
    
    print(f"\n{'='*60}\n🚀 INICIANDO EVALUACIÓN FINAL\n{'='*60}")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluando Test Set"):
            images = images.to(device)
            masks = masks.cpu().numpy()
            
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Asegúrate de que fast_hist esté disponible
            hist += fast_hist(masks.flatten(), preds.flatten(), num_classes)

    # --- CÁLCULOS ---
    # Evitar divisiones por cero añadiendo un pequeño epsilon
    epsilon = 1e-10
    
    acc_global = np.diag(hist).sum() / (hist.sum() + epsilon)
    
    # Recall por clase
    acc_cls = np.diag(hist) / (hist.sum(axis=1) + epsilon)
    
    # Precision e IoU con manejo de errores
    with np.errstate(divide='ignore', invalid='ignore'):
        prec_cls = np.diag(hist) / (hist.sum(axis=0) + epsilon)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iou_cls = np.diag(hist) / (union + epsilon)
        
    mIoU = np.nanmean(iou_cls)
    
    if class_names is None:
        class_names = [f"C{i}" for i in range(num_classes)]
        
    # --- REPORTE Y DATAFRAME ---
    results_df = pd.DataFrame({
        "Clase": class_names,
        "IoU": iou_cls,
        "Precision": prec_cls,
        "Recall": acc_cls
    }).fillna(0.0)

    # Guardar CSV y TXT
    results_df.to_csv(os.path.join(save_path, "metrics_per_class.csv"), index=False)
    
    with open(os.path.join(save_path, "evaluation_summary.txt"), "w", encoding="utf-8") as f:
        f.write("RESUMEN EVALUACIÓN\n" + "="*30 + "\n")
        f.write(f"Global Accuracy: {acc_global:.4%}\n")
        f.write(f"Mean IoU (mIoU): {mIoU:.4f}\n\n")
        f.write(results_df.round(4).to_string(index=False))

    # --- 🎨 NUEVO: GRAFICAR MATRIZ DE CONFUSIÓN ---
    print("🎨 Generando gráfico de matriz de confusión...")
    
    # 1. Normalizar la matriz para que las filas sumen 1 (mostrar porcentajes de Recall)
    # Sumamos epsilon para evitar error si una clase no existe en el test set
    hist_norm = hist / (hist.sum(axis=1, keepdims=True) + epsilon)

    # 2. Configurar el gráfico
    plt.figure(figsize=(10, 8))
    
    # Crear el heatmap con Seaborn
    sns.heatmap(hist_norm, 
                annot=True,        # Mostrar los números en las celdas
                fmt='.1%',         # Formato de porcentaje con 1 decimal (ej. 95.3%)
                cmap='Blues',      # Paleta de colores azul (puedes usar 'Reeds', 'Viridis', etc.)
                xticklabels=class_names, # Etiquetas eje X (Predicción)
                yticklabels=class_names, # Etiquetas eje Y (Real)
                cbar_kws={'label': 'Porcentaje de Píxeles (Recall)'},
                square=True)       # Celdas cuadradas

    plt.title('Matriz de Confusión Normalizada')
    plt.ylabel('Clase Verdadera (Ground Truth)')
    plt.xlabel('Clase Predicha por el Modelo')
    plt.xticks(rotation=45, ha='right') # Rotar etiquetas si son largas
    plt.tight_layout() # Ajustar para que no se corten los textos

    # 3. Guardar la imagen
    plot_filename = os.path.join(save_path, "confusion_matrix_heatmap.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close() # Cerrar la figura para liberar memoria

    print(f"\n✅ Evaluación completada. Resultados guardados en: {save_path}")
    
    return results_df

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    # Configuración
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 3 
    BATCH_SIZE = 8
    LR = 0.0001
    EPOCHS = 30
    
    # NOMBRES DE CLASES (PERSONALIZAR SEGÚN TU CASO)
    # Ejemplo: Si 0 es fondo, 1 hoja sana, 2 hoja enferma
    CLASS_NAMES = ["Fondo", "Hoja-Haz", "Hoja-Enves"] 

    # RUTAS
    train_dir_img = './datos/segmentacion/ds-leaf_segmentation-2-splits/train/images'
    train_dir_mask = './datos/segmentacion/ds-leaf_segmentation-2-splits/train/masks'
    val_dir_img = './datos/segmentacion/ds-leaf_segmentation-2-splits/test/images'
    val_dir_mask = './datos/segmentacion/ds-leaf_segmentation-2-splits/test/masks'

    # Datasets y Loaders
    train_dataset = SegmentationDataset(train_dir_img, train_dir_mask, transform=train_transforms)
    val_dataset = SegmentationDataset(val_dir_img, val_dir_mask, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Datos Train: {len(train_dataset)} | Datos Val: {len(val_dataset)}")

    # Modelo
    model = get_deeplab_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 1. ENTRENAMIENTO
    print("\n🔄 INICIANDO ENTRENAMIENTO...")
    model = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, DEVICE, num_epochs=EPOCHS)
    
    # 2. EVALUACIÓN FINAL (Sobre el conjunto de validación/test)
    # Usamos el modelo que acaba de ser devuelto (que tiene cargados los mejores pesos)
    evaluate_final_test(model, val_loader, DEVICE, NUM_CLASSES,"./results_seg", CLASS_NAMES)