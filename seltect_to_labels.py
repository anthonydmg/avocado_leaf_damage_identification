from glob import glob
import os
import random
import shutil

labeled_images_path1 = glob("datos/segmentacion/all_etiquetados/images/*.jpg")

labeled_basenames = [os.path.basename(lb_im) for lb_im in labeled_images_path1]

print("labeled_basenames:", len(labeled_basenames))


all_images_files = glob("datos/dataset_daño_hojas_palta_v2/**/*.jpg")

print("all_images_files:", len(all_images_files))

all_images_files_filtered = [im_file for im_file in all_images_files if os.path.basename(im_file) not in labeled_basenames]

print("Imagenes Sin Etiqutar:", len(all_images_files_filtered))


dirs_images = [
    ("datos/dataset_daño_hojas_palta_v2/hojas-daño-arañita-roja-haz", 60),
    ("datos/dataset_daño_hojas_palta_v2/hojas-daño-mosa-blanca-aleuropleurocelus-enves", 30),
    ("datos/dataset_daño_hojas_palta_v2/hojas-daño-mosa-blanca-aleuropleurocelus-haz", 30),
    ("datos/dataset_daño_hojas_palta_v2/hojas-daño-queresas-enves", 40),
    ("datos/dataset_daño_hojas_palta_v2/hojas-sana-enves", 50),
    ("datos/dataset_daño_hojas_palta_v2/hojas-sana-haz", 30)
    ]


dir_a_etiquetar = os.path.join("./datos/segmentacion", "imagenes_a_etiquetar")
os.makedirs(dir_a_etiquetar, exist_ok=True)

dir_a_etiquetar_haz = os.path.join("./datos/segmentacion", "imagenes_a_etiquetar", "haz")
os.makedirs(dir_a_etiquetar_haz, exist_ok=True)
dir_a_etiquetar_enves = os.path.join("./datos/segmentacion", "imagenes_a_etiquetar", "enves")
os.makedirs(dir_a_etiquetar_enves, exist_ok=True)

for dir_images, mount_images in dirs_images:
    images_files = glob(f"{dir_images}/*.jpg")
    print("images_files:", len(images_files))
    images_files_filtered = [im_file for im_file in images_files if os.path.basename(im_file) not in labeled_basenames]
    print("images_files_filtered:", len(images_files_filtered))
    files_selected = random.sample(images_files_filtered, mount_images)
    for im_file in files_selected:
        if "haz" in im_file:
            target_path = os.path.join(dir_a_etiquetar,"haz", os.path.basename(im_file))
        else:
            target_path = os.path.join(dir_a_etiquetar,"enves", os.path.basename(im_file))

        shutil.copy2(im_file, target_path)
