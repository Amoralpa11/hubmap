import json
import numpy as np
from skimage.draw import polygon2mask

def get_image_json_data(image_id):
    
    # Loading anatomical and glomerula  mask from json files
    image_data_json = {}
    
    # Set paths bor anat and glom files
    common_path = f'data/train/{image_id}'
    anat_path = f'{common_path}-anatomical-structure.json'
    glom_path = f'{common_path}.json'

    mask_types = ['anat', 'glom']
    paths = [anat_path, glom_path]

    # Read json files into dict
    for mask_type, path in zip(mask_types, paths):
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        image_data_json[mask_type] = data
        
    return image_data_json

def get_cortex_polygons(data):
    
    anat_data = data['anat']
    
    # keep only the cortex information, drop the medula data
    cortex_data = [tissue for tissue in anat_data if tissue['properties']
                   ['classification']['name'] == 'Cortex']
    
    polygon_list = []
    for cd in cortex_data:
        # Extracting polygon vertex
        polygon = np.array(cd['geometry']['coordinates']).reshape(-1, 2)

        # Get geometry type, either polygon or multipolygon
        geometry_type = cd['geometry']['type']
        if geometry_type == 'Polygon':
            polygon_list.append(np.array(polygon))
        # In some cases, cortex is made of several polygons and needs a different processing
        elif geometry_type == 'MultiPolygon':
            polygon_list += [np.array(arr) for arr in np.array([polygon]).reshape(-1)]
        
    return polygon_list

def get_glom_polygons(data):
    
    glom_data = data['glom']
    
    # Extracting polygon vertex
    polygon_list = []
    for glom in glom_data: 
        polygon =  np.array(glom['geometry']['coordinates']).reshape(-1, 2)
        polygon_list.append(polygon)
        
    return polygon_list

def get_mask(image_id, mask_type, train, window=None, out_shape=(256,256)):
       
    """Takes an image id and returns the cortext or glomerula mask depending on mask_type"""
    
    data = get_image_json_data(image_id)
    
    w, h = train[train.sample_id == image_id][['width_pixels', 'height_pixels' ]].values[0]
    
    h0, w0 = 0, 0
    
    h1, w1 = h, w
    
    if not window is None:
        (h0, h1), (w0, w1) = window
    
    
    if mask_type == 'cortex':
        polygon_list = get_cortex_polygons(data)
    elif mask_type == 'glom':
        polygon_list = get_glom_polygons(data)
    
    # Create an empty boolean matrix
    mask = np.zeros(out_shape, dtype=bool)
        
    # Draw the polygon into the blank mask
    for polygon in polygon_list:
        polygon = np.array(polygon)


        polygon[:,0] = polygon[:,0] - w0
        polygon[:,1] = polygon[:,1] - h0
        
        polygon[:,0] = polygon[:,0]/(w1-w0)*out_shape[1]
        polygon[:,1] = polygon[:,1]/(h1-h0)*out_shape[0]

        mask = mask + polygon2mask(out_shape, polygon[:,::-1])
        
    return mask