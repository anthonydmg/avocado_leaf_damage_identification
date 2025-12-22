import json
import os
import requests
import labelbox as lb
import urllib.request
from PIL import Image
import numpy as np

def download_masks(ndjson_file, output_dir, headers = None):
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

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
                composite_mask_path = os.path.join(output_dir, f"{image_name[:-4]}_MASK.JPG")
                # Descargar máscaras individuales
                objects = annotations.get("objects", [])
                if not objects:
                    print(f"Generando Mascara vacia: {composite_mask_path}...")
                    empty_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    image = Image.fromarray(empty_mask)
                    image.save(composite_mask_path)
                    continue

                print(objects[0].keys())
                composite_mask = objects[0].get("composite_mask", {})
                composite_mask_url = composite_mask.get("url")
                print("composite_mask_url:", composite_mask_url)
                if composite_mask_url:
                    print(f"Descargando máscara compuesta: {composite_mask_path}...")
                    try:
                        req = urllib.request.Request(composite_mask_url, headers=headers)
                        image = Image.open(urllib.request.urlopen(req))
                        if image.mode == "RGBA":
                            image = image.convert("RGB")
                        image.save(composite_mask_path)

                        #gray_image = image.convert("L")

                        # Convertir a array numpy
                        #gray_array = np.array(gray_image)

                        # Crear máscara binaria: lo blanco se convierte en 255, lo demás en 0
                        # Puedes ajustar el umbral si quieres, aquí uso 200 como referencia
                        #threshold = 200
                        #binary_mask = (gray_array > 0).astype(np.uint8) * 255

                        # Convertir a imagen PIL y guardar
                        #mask_image = Image.fromarray(binary_mask)
                        #composite_mask_binary_path = os.path.join(output_dir, f"binary_mask/{image_name[:-4]}_MASK.JPG")
                        #mask_image.save(composite_mask_binary_path)

                        print(f"Máscara compuesta descargada en: {composite_mask_path}")
                    except requests.exceptions.RequestException as e:
                        print(f"Error descargando {composite_mask_url}: {e}")

if __name__ == "__main__":
    API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbTNjbTh1bjUwOHRoMDd3ODVydTY0YXZ1Iiwib3JnYW5pemF0aW9uSWQiOiJjbTNjbTh1bXUwOHRnMDd3ODc4OGkyMnhhIiwiYXBpS2V5SWQiOiJjbWdtdGJmanYzNm8yMDd5MWgzenRocTgxIiwic2VjcmV0IjoiYjkxOTczNzFmZTM1MmY1MzBkYmVmMzFjNmNiMjI0NDQiLCJpYXQiOjE3NjAyMTk2MjUsImV4cCI6MTc2NzQ3NzIyNX0.TbdQf-yUGmGWJ81j_FK-2XgAQ2QdJ_K7ttlGagAp2eo"
    client = lb.Client(api_key=API_KEY)
    json_file = "./data/Etiquetado-Biochumbi-Oct-2025.ndjson"
    output_dir = "./data/etiquetados/campos-biochumbi-oct-2025/masks"
    download_masks(json_file, output_dir, headers= client.headers)