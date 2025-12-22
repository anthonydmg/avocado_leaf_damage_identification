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
# 2. FUNCIÓN DE ENTRENAMIENTO Y VALIDACIÓN
# ==========================================

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Cada época tiene fase de entrenamiento y fase de validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Habilita Dropout y BatchNorm
                dataloader = train_loader
            else:
                model.eval()   # Congela Dropout y BatchNorm
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_pixels = 0
            progress_bar = tqdm(dataloader, desc = f"Epochs {epoch} {phase}")
            # Iterar sobre los datos
            for images, masks in progress_bar:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                # Trackeo de historial solo en Train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    
                    # DeepLab devuelve diccionario. 
                    # En Train usamos 'aux' loss también. En Val solo 'out'.
                    if phase == 'train':
                        loss_main = criterion(outputs['out'], masks)
                        loss_aux = criterion(outputs['aux'], masks)
                        loss = loss_main + (0.4 * loss_aux)
                        preds = torch.argmax(outputs['out'], dim=1)
                    else:
                        loss = criterion(outputs['out'], masks)
                        preds = torch.argmax(outputs['out'], dim=1)

                    # Backward + Optimize solo si es training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                progress_bar.set_postfix({"loss": loss.item()})
                # Estadísticas
                running_loss += loss.item() * images.size(0)
                
                # Cálculo simple de precisión (Accuracy) píxel a píxel
                running_corrects += torch.sum(preds == masks.data)
                total_pixels += torch.numel(masks.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / total_pixels

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Guardar el modelo si es el mejor hasta ahora (solo en fase val)
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "mejor_modelo_deeplab.pth")
                print("  -> Modelo guardado (Mejora en Loss)")

    print(f'Mejor Val Loss: {best_loss:.4f}')
    
    # Cargar los mejores pesos antes de devolver el modelo
    model.load_state_dict(best_model_wts)
    return model

def get_deeplab_model(num_classes):
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

# ==========================================
# 3. EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    # Configuración
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 3 
    BATCH_SIZE = 16
    LR = 0.0001
    EPOCHS = 20

    # RUTAS (MODIFICAR)
    # Estructura recomendada de carpetas:
    # /dataset/train/images
    # /dataset/train/masks
    # /dataset/val/images
    # /dataset/val/masks
    
    train_dir_img = './datos/segmentacion/ds-leaf_segmentation-2-splits/train/images'
    train_dir_mask = './datos/segmentacion/ds-leaf_segmentation-2-splits/train/masks'
    val_dir_img = './datos/segmentacion/ds-leaf_segmentation-2-splits/test/images'
    val_dir_mask = './datos/segmentacion/ds-leaf_segmentation-2-splits/test/masks'

    # 1. Crear Datasets
    # Usamos train_transforms para train (tiene data augmentation)
    train_dataset = SegmentationDataset(train_dir_img, train_dir_mask, transform=train_transforms)
    
    # Usamos val_transforms para val (solo resize y normalize, SIN augmentation)
    val_dataset = SegmentationDataset(val_dir_img, val_dir_mask, transform=val_transforms)

    # 2. Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Datos Train: {len(train_dataset)} | Datos Val: {len(val_dataset)}")

    # 3. Modelo, Loss, Optimizer
    model = get_deeplab_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Entrenar
    model = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, DEVICE, num_epochs=EPOCHS)