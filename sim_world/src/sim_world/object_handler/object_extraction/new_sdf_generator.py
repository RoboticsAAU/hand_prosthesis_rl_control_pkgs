import os
from pathlib import Path
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET

##### USER CONFIGURABLES #####
# Set the main folder containing object folders
main_folder = "/home/chathushka/Documents/hand_prosthesis_ws/src/hand_prosthesis_rl_control_pkgs/assets/shapenet_sdf"
# Set the template sdf file path
template_sdf_file = "template_mesh.sdf"
##############################

# Function to extract details from sdf file
def extract_sdf_details(sdf_file):
    # logic to extract details from the sdf file
    # Initilize the dictionary to store the extracted details

    details ={"mass": None,
                "ixx": None,
                "ixy": None,
                "ixz": None,
                "iyy": None,
                "iyz": None,
                "izz": None,
                "mesh_stl_path": None,}
    
    # Parse the sdf file
    tree = ET.parse(sdf_file)
    root = tree.getroot()

    # Extract the mass
    mass_element = root.find(".//inertial/mass")
    if mass_element is not None:
        details["mass"] = float(mass_element.text)
        # if mass element value is 0, set it to 0.5
        if details["mass"] == 0:
            details["mass"] = 0.5
    
    # Extract the inertia tensor components
    inertia_element = root.find(".//inertial/inertia")
    if inertia_element is not None:
        details["ixx"] = float(inertia_element.find("ixx").text)
        details["ixy"] = float(inertia_element.find("ixy").text)
        details["ixz"] = float(inertia_element.find("ixz").text)
        details["iyy"] = float(inertia_element.find("iyy").text)
        details["iyz"] = float(inertia_element.find("iyz").text)
        details["izz"] = float(inertia_element.find("izz").text)
    
    # Extract the mesh stl path
    link_element = root.find(".//uri")
    if link_element is not None:
        relative_path = Path(link_element.text.strip()).name
        details["mesh_stl_path"] = relative_path

    return details

# Function to replace details in template sdf file and save it
def replace_and_save_sdf(template_sdf_file, extracted_details, output_sdf_file):
    #read the template sdf file
    with open(template_sdf_file, "r") as file:
        template_sdf_content = file.read()

    # Replace the placeholders with extracted details
    template_sdf_content = template_sdf_content.replace("{model_mass}", str(extracted_details["mass"]))
    template_sdf_content = template_sdf_content.replace("{inertia_values[0]}", str(extracted_details["ixx"]))
    template_sdf_content = template_sdf_content.replace("{inertia_values[1]}", str(extracted_details["ixy"]))
    template_sdf_content = template_sdf_content.replace("{inertia_values[2]}", str(extracted_details["ixz"]))
    template_sdf_content = template_sdf_content.replace("{inertia_values[3]}", str(extracted_details["iyy"]))
    template_sdf_content = template_sdf_content.replace("{inertia_values[4]}", str(extracted_details["iyz"]))
    template_sdf_content = template_sdf_content.replace("{inertia_values[5]}", str(extracted_details["izz"]))
    template_sdf_content = template_sdf_content.replace("{mesh_stl_path}", extracted_details["mesh_stl_path"])

    # Save the modified sdf file
    with open(output_sdf_file, "w") as file:
        file.write(template_sdf_content)
    
    return output_sdf_file


if __name__ == "__main__":

    # Iterate through each object folder
    for category_folder in tqdm(Path(main_folder).glob("*")):
        for obj_folder in category_folder.glob("obj_*"):
            print(f"Processing object folder: {obj_folder}")  # Debug print
            # Get the mesh sdf file path
            mesh_sdf_file = obj_folder / "mesh.sdf"

            # Extract details from the mesh sdf file
            extracted_details = extract_sdf_details(mesh_sdf_file)
            print(f"All the details extracted : {obj_folder}")  # Debug print
            
            # Replace details in the template sdf file and save it
            output_sdf_file = obj_folder / "mesh_new.sdf"
            replace_and_save_sdf(template_sdf_file, extracted_details, output_sdf_file)
            print("Template file rearrenged")  # Debug print
            print(f"Modified SDF file saved: {output_sdf_file}")