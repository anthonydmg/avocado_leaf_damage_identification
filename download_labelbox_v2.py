import json
import os
import requests
import labelbox as lb
import urllib.request
from PIL import Image
import numpy as np
import cv2
from glob import glob
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

classes_ids = {
    "background": 0,
    "leaf-haz": 1,
    "leaf-enves": 2
}


def download_masks(ndjson_file, output_dir, headers = None):
    # Crear directorio de salida
    os.makedirs(os.path.join(output_dir,"objects_masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir,"masks_long"), exist_ok=True)
    # Cargar el JSON
    with open(ndjson_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Extrear las anotaciones
    for item in data:
        print("item:",item.keys())
        data_row = item.get("data_row", {})
        media_attributes = item.get("media_attributes", {})
        height = media_attributes.get("height")
        width = media_attributes.get("width")
        image_id = data_row.get("id", "unknown")  # ID de la imagen para nombrar archivos
        print("image_id:", image_id)
        image_name = data_row.get("external_id", "unknown").replace(" ", "_")  # Nombre de la imagen limpio
        print("image_name:", image_name)

        # Extraer los proyectos
        projects = item.get("projects", {})
        for project_id, project_data in projects.items():
            print("project_id:", project_id)
            labels = project_data.get("labels", [])
            for label in labels:
                annotations = label.get("annotations", {})
                print(annotations.keys())
               
                # Descargar máscaras individuales
                objects = annotations.get("objects", [])
                basename_img = image_name[:-4]
                os.makedirs(os.path.join(output_dir, "objects_masks", basename_img), exist_ok= True)
                mask_long = np.zeros((height, width), dtype= np.long)
                
                if not objects:
                    instance_mask_path = os.path.join(output_dir, "objects_masks", image_name[:-4], f'{basename_img}_00_MASK.JPG')
                    print(f"Generando Mascara vacia: {instance_mask_path}...")
                    empty_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    image = Image.fromarray(empty_mask)
                    image.save(instance_mask_path)
                    continue

                for i, obj in enumerate(objects):

                    instance_mask_path = os.path.join(output_dir, "objects_masks", image_name[:-4], f'{basename_img}_{i:02}_MASK.JPG')
                    if os.path.isfile(instance_mask_path):
                        print("Ya Descargado .....")
                        print()
                        continue
                    
                    print(obj.keys())
                    instance_mask = obj.get("mask", {})
                    mask_url = instance_mask.get("url")
                    class_name = obj.get("name", None)

                    print("mask_url:", mask_url)
                    if mask_url:
                        print(f"Descargando máscara de instancia: {instance_mask_path}...")
                        try:
                            req = urllib.request.Request(mask_url, headers=headers)
                            mask_image = Image.open(urllib.request.urlopen(req))
                            if mask_image.mode == "RGBA":
                                mask_image = mask_image.convert("RGB")
                            mask_image.save(instance_mask_path)

                            mask_arr = np.array(mask_image)
                            print("mask_arr:", mask_arr.shape)
                            print("classes_ids[class_name]:", classes_ids[class_name])
                            print("mask_long:", mask_long.shape)
                            mask_long[mask_arr > 0] = classes_ids[class_name]

    
                            print(f"Máscara compuesta descargada en: {instance_mask_path}")
                        except requests.exceptions.RequestException as e:
                            print(f"Error descargando {instance_mask_path}: {e}")

                mask_long_path = os.path.join(output_dir, "masks_long", f'{basename_img}_MASK_LONG.npy')
                np.save(mask_long_path, mask_long)

def copy_images(base_dir_images, masks_dir, output_dir_images):
    files = glob(f"{base_dir_images}/**/*.jpg", recursive=True)
    paths = {os.path.basename(file)[:-4]: file for file in files}
    files_masks = glob(f"{masks_dir}/*.npy", recursive=True)
    basename_masks = [os.path.basename(f_mask).replace("_MASK_LONG.npy","") for f_mask in files_masks]
    print("paths:", basename_masks)
    
    os.makedirs(output_dir_images, exist_ok=True)
    
    for b_mask in basename_masks:
        im_path = paths[b_mask]
        shutil.copy(im_path, os.path.join(output_dir_images, b_mask + ".jpg")) 

def copy_to_files(source_files, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for file_path in tqdm(source_files, "Copy:"):
        base_name = os.path.basename(file_path)
        shutil.copyfile(file_path, f"{target_dir}/{base_name}")

def split_data(dir_base, target_dir = "./datos/segmentacion/ds-leaf_segmentation-2-splits"):
    images_paths = glob(f"{dir_base}/images/*.jpg")
    train_im_paths, test_im_paths = train_test_split(images_paths, test_size=0.3, random_state=42, shuffle=True)


    copy_to_files(train_im_paths, f"{target_dir}/train/images")
    copy_to_files(test_im_paths, f"{target_dir}/test/images")
    
    train_mask_paths = [os.path.join(dir_base,"masks","masks_long",os.path.basename(im_path).replace(".jpg", "_MASK_LONG.npy")) for im_path in train_im_paths]
    test_mask_paths = [os.path.join(dir_base,"masks","masks_long",os.path.basename(im_path).replace(".jpg", "_MASK_LONG.npy")) for im_path in test_im_paths]
    #val_mask_paths = [im_path.replace("images_undistorned", "masks_undistorned").replace(".JPG", "_MASK.JPG") for im_path in val_im_paths]

    copy_to_files(train_mask_paths, f"{target_dir}/train/masks")
    copy_to_files(test_mask_paths, f"{target_dir}/test/masks")

if __name__ == "__main__":
    #API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbTNjbTh1bjUwOHRoMDd3ODVydTY0YXZ1Iiwib3JnYW5pemF0aW9uSWQiOiJjbTNjbTh1bXUwOHRnMDd3ODc4OGkyMnhhIiwiYXBpS2V5SWQiOiJjbWdtdGJmanYzNm8yMDd5MWgzenRocTgxIiwic2VjcmV0IjoiYjkxOTczNzFmZTM1MmY1MzBkYmVmMzFjNmNiMjI0NDQiLCJpYXQiOjE3NjAyMTk2MjUsImV4cCI6MTc2NzQ3NzIyNX0.TbdQf-yUGmGWJ81j_FK-2XgAQ2QdJ_K7ttlGagAp2eo"
    API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbWplbDRvZWQweDN1MDcwZmJydXFoZzhpIiwib3JnYW5pemF0aW9uSWQiOiJjbWplbDRvZTYweDN0MDcwZjM5aWExZGgxIiwiYXBpS2V5SWQiOiJjbWpnNnUzNDgwc2ZqMDcyOTVlemRldWl3Iiwic2VjcmV0IjoiNWY0MWE5MDRkZjQ3MjRiOGY1MjQ1ZjBjMDRiYTI3NjQiLCJpYXQiOjE3NjYzNDk0MTQsImV4cCI6MTc4MDg2NDYxNH0.yPRgctJnq9DJcSj0cDr5-n-QOiiacXZ-2nlF2Y7stsc"
    client = lb.Client(api_key=API_KEY)
    json_file = "./datos/segmentacion/Segmentation_Datos_C2.ndjson"
    output_dir = "./datos/segmentacion/etiquetados2/masks/"
    
    #download_masks(json_file, output_dir, headers= client.headers)
    base_dir_images = "./datos/imagenes_filtradas_actualizado"
    masks_dir = "datos/segmentacion/etiquetados2/masks/masks_long"
    output_dir_images = "datos/segmentacion/etiquetados2/images"
    #copy_images(base_dir_images, masks_dir, output_dir_images)
    
    split_data(dir_base = "datos/segmentacion/all_etiquetados", target_dir = "./datos/segmentacion/ds-leaf_segmentation-2-splits")