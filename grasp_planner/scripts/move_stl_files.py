import os
import glob
import shutil
from pathlib import Path
from tqdm import tqdm


##### USER CONFIGURABLES #####
main_folder = "/Development/hand_rl_ws/src/hand_prosthesis_rl_control_pkgs/assets/shapenet"
destination_folder = "/Development/hand_rl_ws/src/hand_prosthesis_rl_control_pkgs/assets/graspit_shapenet"
##############################

# Create desired directory (raiseS error if already exists)
#os.mkdir(destination_folder)

# Get category folders
cat_folders = sorted(glob.glob(main_folder + "/*"))

for cat_id, cat_folder in enumerate(cat_folders):
    # Find folders in each category
    obj_folders = sorted(glob.glob(cat_folder + "/*"))

    # 
    category_name = Path(cat_folder).name
    os.mkdir(os.path.join(destination_folder, category_name))
    
    for obj_id, obj_folder in tqdm(enumerate(obj_folders)):
        
        files = glob.glob(obj_folder + '/model.*')
        # Filter paths to keep only those ending with ".obj" or ".stl"
        
        # Search for .obj files
        files_obj = [path for path in files if path.endswith((".obj"))]
        
        # Raise error if more than one .obj file is found
        if len(files_obj) > 1:
            raise ValueError(f"Error: {obj_folder} contains more than one obj file.")
        
        elif len(files_obj) == 0:
            raise ValueError(f"Error: {obj_folder} does not contain any obj file.")
        
        else:        
            file_obj = files_obj[0]
        
        # New name to be assigned based on object id
        destination_name = f"obj_{obj_id + 1}.obj"
        
        # Copy to new destination with new name
        shutil.copy(file_obj, os.path.join(destination_folder, category_name, destination_name))
        
        # Rename original folder (since they had weird names)
        os.rename(obj_folder, os.path.join(cat_folder, f"obj_{obj_id + 1}"))
            