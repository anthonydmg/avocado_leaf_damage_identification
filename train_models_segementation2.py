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
# ==========================================
# 1. DATASETS Y TRANSFORMACIONES
# ==========================================

# Definimos transformaciones. 
# Nota: Usualmente en Train se agrega 'RandomHorizontalFlip' o rotaciones,
# pero en Val solo se redimensiona y normaliza.
train_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5), # Aumentación de datos simple
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        # Asumo que tu convención de nombres es correcta
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_MASK_LONG.npy"))

        image = Image.open(img_path).convert("RGB")
        
        # 1. Cargar NumPy
        mask_np = np.load(mask_path) 
        
        # 2. Convertir NumPy -> PIL Image para poder usar .resize()
        # Es importante asegurar que sea int32 o int64 antes de pasar a PIL para no perder datos,
        # aunque PIL 'I' mode maneja int32.
        mask_pil = Image.fromarray(mask_np.astype(np.int32)) 

        # -----------------------------------------------------------
        # SOLUCIÓN AL PROBLEMA DE SINCRONIZACIÓN (Transformaciones)
        # -----------------------------------------------------------
        # Lo ideal es aplicar las transformaciones JUNTAS. 
        # Como es difícil hacerlo con 'self.transform' estándar si separas lógica,
        # al menos aseguramos el resize manual correcto aquí:
        
        # Aplicar resize a la imagen (si está en self.transform, esto es redundante, 
        # pero asegúrate que self.transform NO tenga RandomFlip si lo haces así separado)
        if self.transform:
            image = self.transform(image)
            
        # 3. Redimensionar Máscara usando NEAREST (Vecino más cercano)
        # Esto mantiene los valores 0, 1, 2... sin crear decimales.
        mask_pil = mask_pil.resize((512, 512), resample=Image.NEAREST)
        
        # 4. Convertir PIL -> Tensor Long
        mask_np = np.array(mask_pil)
        mask_tensor = torch.from_numpy(mask_np).long() # .long() es lo que pide CrossEntropy
        
        return image, mask_tensor

# ==========================================
# 2. FUNCIONES DE ENTRENAMIENTO
# ==========================================

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

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

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    
                    if phase == 'train':
                        loss_main = criterion(outputs['out'], masks)
                        loss_aux = criterion(outputs['aux'], masks)
                        loss = loss_main + (0.4 * loss_aux)
                        preds = torch.argmax(outputs['out'], dim=1)
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

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "mejor_modelo_deeplab_mobilenatv3.pth")
                print("  -> Modelo guardado (Mejora en Loss)")

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

def evaluate_final_test(model, dataloader, device, num_classes, class_names=None):
    """
    Ejecuta una evaluación completa calculando IoU, Precision y Recall por clase
    basado en la matriz de confusión global.
    """
    model.eval()
    model.to(device)
    
    # Matriz de confusión acumulada (Global)
    hist = np.zeros((num_classes, num_classes))
    
    print("\n" + "="*50)
    print("🚀 INICIANDO EVALUACIÓN FINAL (MÉTRICAS COMPLETAS)")
    print("="*50)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluando Test Set"):
            images = images.to(device)
            masks = masks.cpu().numpy() # Ground truth a CPU/Numpy
            
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1).cpu().numpy() # Prediccion a CPU/Numpy
            
            # Acumular matriz de confusión
            hist += fast_hist(masks.flatten(), preds.flatten(), num_classes)

    # --- CÁLCULOS ---
    # Accuracy Global
    acc_global = np.diag(hist).sum() / hist.sum()
    
    # Accuracy por Clase (Recall)
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    
    # Precision por Clase
    with np.errstate(divide='ignore', invalid='ignore'):
        prec_cls = np.diag(hist) / hist.sum(axis=0)
    
    # IoU por Clase
    union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_cls = np.diag(hist) / union
        
    mIoU = np.nanmean(iou_cls)
    
    # --- REPORTE ---
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
        
    # Crear DataFrame
    results_df = pd.DataFrame({
        "Clase": class_names,
        "IoU": iou_cls,
        "Precision": prec_cls,
        "Recall": acc_cls
    })
    
    print("\n📊 RESULTADOS POR CLASE:")
    print(results_df.round(4).to_string(index=False))
    print("-" * 50)
    print(f"✅ Global Accuracy: {acc_global:.2%}")
    print(f"✅ Mean IoU (mIoU): {mIoU:.4f}")
    print("=" * 50)


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
    evaluate_final_test(model, val_loader, DEVICE, NUM_CLASSES, CLASS_NAMES)