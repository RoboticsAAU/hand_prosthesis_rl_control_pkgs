import os # to walk through directories, to rename files
import sys

# Libraries
import trimesh # for converting voxel grids to meshes (to import objects into simulators)

# Modules
import tools_sdf_generator
import glob

from pathlib import Path
import shutil
from tqdm import tqdm

##### USER CONFIGURABLES #####
main_folder = "/home/chathushka/Documents/hand_prosthesis_ws/src/hand_prosthesis_rl_control_pkgs/assets/shapenet"
destination_folder = "/home/chathushka/Documents/hand_prosthesis_ws/src/hand_prosthesis_rl_control_pkgs/assets/shapenet_sdf"
##############################


def process_file(filename, scaling_factor):
    """Processes a single file, performing scaling, volume calculation,and mesh generation."""
    
    # Generate a folder to store the mesh within the destination folder structure
    category_name = Path(filename).parent.parent.name  # Extract category name
    object_name = Path(filename).parent.name           # Extract object name
    converted_sdf_dir = os.path.join(destination_folder)  # Destination folder
    destination_path = os.path.join(converted_sdf_dir, category_name, object_name,)

    # Create category and object folders if they don't exist
    os.makedirs(destination_path, exist_ok=True)

    # Load the mesh
    print("Loading the mesh")
    mesh = trimesh.load(filename)
    # scaling_factor = 100
    mesh.apply_scale(scaling=scaling_factor)
    mass = mesh.volume # WATER density

    # Print volume and mass information
    print(f"\n\nMesh volume for {filename}: {mesh.volume} (used as mass)")
    print(f"Mass (equal to volume) for {filename}: {mass}")
    print(f"Mesh convex hull volume for {filename}: {mesh.convex_hull.volume}\n\n")
    print(f"Mesh bounding box volume for {filename}: {mesh.bounding_box.volume}")

    # Mesh processing steps
    print("Merging vertices closer than a pre-set constant...")
    print(f"Processing {filename}...")
    mesh.merge_vertices()
    print("Removing duplicate faces...")
    mesh.remove_duplicate_faces()
    print("Making the mesh watertight...")
    trimesh.repair.fill_holes(mesh)
    # print("Fixing inversion and winding...")
    # trimesh.repair.fix_winding(mesh)
    # trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)

    # Print volume information after processing
    print(f"\n\nMesh volume for {filename}: {mesh.volume}")
    print(f"Mesh convex hull volume for {filename}: {mesh.convex_hull.volume}")
    print(f"Mesh bounding box volume for {filename}: {mesh.bounding_box.volume}\n\n")

    # Compute the center of mass
    center_of_mass = mesh.center_mass
    moments_of_inertia = mesh.moment_inertia  

    # Print center of mass and moments of inertia
    print(f"Computing the center of mass for {filename}: ")
    print(center_of_mass)
    print(f"Computing moments of inertia for {filename}: ")
    print(moments_of_inertia)

    # Generate STL and SDF files
    trimesh.exchange.export.export_mesh(mesh=mesh,file_obj=os.path.join(destination_path, "mesh.stl"), file_type="stl")

    tools_sdf_generator.generate_model_sdf(directory=destination_path,object_name="mesh",
    center_of_mass=center_of_mass,inertia_tensor=moments_of_inertia,mass=mass,model_stl_path=os.path.join(destination_path, "mesh.stl"),scale_factor=1.0 )

    #tools_sdf_generator.generate_model_sdf(directory=directory,object_name="mesh",center_of_mass=center_of_mass,inertia_tensor=moments_of_inertia,mass=mass,model_stl_path=os.path.join(directory, "mesh.stl"),scale_factor=1.0 )


if __name__ == "__main__":

    """This script is designed to be used with a pre-populated list of file paths.
"Ensure the `recursive` module provides a list named `file_paths`."""
    
    scaling_factor = 1.0
    
    # Create the main destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    for category_folder in tqdm(Path(main_folder).glob("*")):
        category_name = category_folder.name
        destination_category_folder = os.path.join(destination_folder, category_name)
        os.makedirs(destination_category_folder, exist_ok=True)

        print(f"Processing category folder: {category_folder}")  # Debug print


        # Define object_folder loop within the category_folder loop
        for obj_folder in category_folder.glob("obj_*"):
            print(f"Processing object folder: {obj_folder}")  # Debug print

            filename = obj_folder / "convex.obj"
            if not filename.exists():
                filename = obj_folder / "model.stl"

            print(f"File to be processed: {filename}")  # Debug print

            if filename.exists():
                process_file(filename, scaling_factor)

                # Copy the processed file to the destination folder
                destination_path = os.path.join(destination_category_folder, obj_folder.name)
                shutil.copy(filename, destination_path)

                print("File processed and copied to destination")  # Debug print
            else:
                print("File does not exist!")  # Debug print