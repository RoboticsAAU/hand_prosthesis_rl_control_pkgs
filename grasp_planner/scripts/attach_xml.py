import os
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

if __name__ == "__main__":
    # Folder where the STL files are stored
    folder = "/Development/hand_rl_ws/src/hand_prosthesis_rl_control_pkgs/assets/graspit_shapenet"
    xml_template = "/Development/graspit/models/objects/test_obj.xml"
    
    # Load the XML template
    tree = ET.parse(xml_template)
    root = tree.getroot()  
    
    # Find the geometryFile element
    geometry_file_element = root.find('.//geometryFile')    
    
    # Get category folders
    cat_folders = sorted(glob.glob(folder + "/*"))
    
    for cat_folder in cat_folders:
        
        # Find all .wrl files in the category folder
        cat_files = glob.glob(cat_folder + "/*.wrl")
        
        # Get the category name
        category_name = os.path.basename(os.path.normpath(cat_folder))

        # Find folders in each category
        for file in tqdm(cat_files):
            file_name = Path(file).stem
            
            # Change the value of the geometryFile element
            geometry_file_element.text = file_name + ".wrl"
            
            # Write the new XML file
            tree.write(os.path.join(folder, category_name, file_name + ".xml"), xml_declaration=True)