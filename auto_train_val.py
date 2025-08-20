import os, shutil, random

source_dir = "augmented_dataset_ladybird"
dest_dir = "ladybird_split"
split_ratio = 0.8  # 80% train, 20% val

species_list = os.listdir(source_dir)

for species in species_list:
    species_path = os.path.join(source_dir, species)
    images = os.listdir(species_path)
    random.shuffle(images)
    
    split_point = int(len(images) * split_ratio)
    train_imgs = images[:split_point]
    val_imgs = images[split_point:]
    
    for mode in ['train', 'val']:
        os.makedirs(os.path.join(dest_dir, mode, species), exist_ok=True)
    
    for img in train_imgs:
        shutil.copy(os.path.join(species_path, img), os.path.join(dest_dir, 'train', species, img))
    for img in val_imgs:
        shutil.copy(os.path.join(species_path, img), os.path.join(dest_dir, 'val', species, img))

print("✅ Dataset split into train and val folders!")