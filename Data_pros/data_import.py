import pyavsrad as pyavs
import numpy as np
from pyavsrad_extension.extensions import TAVSFldToNumpyNdArray
import pydicom as pd
import os
import sys


def get_dicom_paths(data_dir, patID):
    """Retrieves the CT dicom file based on the input configuration. Converts the
    ct dicoms to an AVS field and returns the TScan object

    :arg data_dir: directory of the data
    :arg patID: patient ID
    :return dicom_paths: path to DICOMS
    """
    patient_dir = os.path.join(data_dir, patID)
    dicom_paths = {}
    for dcm_file in os.listdir(patient_dir):
        full_path = os.path.join(patient_dir, dcm_file)
        dcm = pd.read_file(full_path)
        dicom_paths[dcm.Modality] = full_path
    return dicom_paths


def load_ct(ct_dicom_path):
    """Retrieves the CT dicom file based on the input configuration. Converts the
    ct dicoms to an AVS field and returns the TScan object

    :arg input: pydantic object that points to the requested input data
    :return scan: TScan object
    """
    scan = pyavs.TScan()
    dummy = pyavs.TAVSField()
    try:
        [scan.Data, scan.Transform, dummy, scan.Properties] = pyavs.READ_DICOM_FROM_DISK(
            str(ct_dicom_path), '*.dcm', False, False)
    except Exception as e:
        print("Problem with loading ct scan from " + ct_dicom_path)
        print(f"error: {str(e)} ")
    return scan


def load_RTDOSE(rtdose_dicom_path):
    dose = pyavs.TScan()
    dummy = pyavs.TAVSField()
    try:
        #print("Create TScan object")
        [dose.Data, dose.Transform, dummy, dose.Properties] = pyavs.READ_DICOM_FROM_DISK(
            str(rtdose_dicom_path), '*.dcm', True, False)
        #print("Loaded dose succesfully")
    except Exception as e:
        print("Problem with loading dose from " + rtdose_dicom_path)
        print(f"error: {str(e)} ")

    return dose


def load_structures(rtstruct_dicom_path):
    """Retrieves an RTSTRUCT dicom file based on the input configuration. Converts the
    structuree to an AVS field and return the TDelineation object

    :arg input: pydantic object that points to the requested input data
    :return structure: TDelineation object
    """
    structures = pyavs.TDelineation()
    try:
        #print("Create TDelineation object")
        [structures.dots, structures.index, structures.lut] = pyavs.DICOM_FILE_GET_RTSTRUCTS(
            rtstruct_dicom_path, False, True)
        #print("TDelineation object created succesfully")
    except Exception as e:
        print("Problem with loading rtstruct from " + rtstruct_dicom_path)
        print(f"error: {str(e)} ")

    # get the specified structures
    num_structures = pyavs.DIL_GET_OBJECTCOUNT(structures.lut)
    spec_structures = {}
    for i in range(0, num_structures):
        structure = pyavs.TDelineation()
        name = pyavs.DIL_GET_NAME(structures.lut, i)
        # NOTE Return contour coordinate information
        [
            structure.dots,
            structure.index,
            structure.lut,
        ] = pyavs.DIL_SELECT(
            structures.dots, structures.index, structures.lut, i
        )
        structure.UseTriangulation(True)
        structure.TriangleDots.Make()
        spec_structures[name] = structure
    return spec_structures


def struct_arr(struc_txt, tot_structures, dose):
    """ Makes numpy arrays from all structures defined in the .txt file.

    :arg input: path of .txt file,
    :return structure: Dictionary
    """
    struct_arrays = {}
    with open(struc_txt) as s:
        structs = [line.rstrip()[1:-1] for line in s]
    for i in range(0, len(structs)):
        name = structs[i]
        try:
            structAVSfld = dose.BurnTo(tot_structures[structs[i]])
            struct_arrays[name] = TAVSFldToNumpyNdArray(structAVSfld.Data)
        except:
            dose_arr = TAVSFldToNumpyNdArray(dose.Data)
            struct_arrays[name] = np.zeros(dose_arr.shape)
            print('Struct not found, making zeros')
    return struct_arrays


def standard_size(size, array):
    """ Adds zeros to the array until it matches the wanted size

    :param array: array to be modified
    :type array: numpy ndarray
    :param size: wanted array size,
    :return mod_array: Modified array
    """
    mod_array = array
    start = [0, 0, 0]
    end = [0, 0, 0]
    for i in range(len(size)):
        if mod_array.shape[i] < size[i]:
            while mod_array.shape[i] + 1 < size[i]:
                mod_array = np.insert(mod_array, 0, [0], axis=i)
                start[i] += 1
                add = np.expand_dims(np.zeros(np.take(mod_array, 0, axis=i).shape), axis=i)
                mod_array = np.concatenate((mod_array, add), axis=i)
                end[i] += 1
            if size[i] - mod_array.shape[i] == 1:
                add = np.expand_dims(np.zeros(np.take(mod_array, 0, axis=i).shape), axis=i)
                mod_array = np.concatenate((mod_array, add), axis=i)
                end[i] += 1
            elif size[i] - mod_array.shape[i] == 0:
                continue
            else:
                print("Not working properly")
        elif mod_array.shape[i] > size[i]:
            while mod_array.shape[i] - 2 > size[i]:
                mod_array = np.delete(mod_array, 0, axis=i)
                mod_array = np.delete(mod_array, 0, axis=i)
                start[i] += -2
                mod_array = np.delete(mod_array, -1, axis=i)
                end[i] += -1
            if mod_array.shape[i] - size[i] == 2:
                mod_array = np.delete(mod_array, 0, axis=i)
                start[i] += -1
                mod_array = np.delete(mod_array, -1, axis=i)
                end[i] += -1
            elif mod_array.shape[i] - size[i] == 1:
                mod_array = np.delete(mod_array, 0, axis=i)
                start[i] += -1
        else:
            continue
    return mod_array, start, end


# Paths
def input_data(N):
    """ Extracts a dicom file of certain patient,
    converts to numpy arrays and makes structure masks of correct size.

    :param N: number of the patient in the patient list
    :return structures_mask: 4 dimensional array of structure mask booleans
    :return dose_arr: 3 dimensional array of dose
    :return start_mod: Amount of voxels added at begin of each dimension
    :return end_mod: Amount of voxels added at end of each dimension
    """
    ### MODIFY START ###
    # MODIFY THIS PART FOR DIFFERENT FILE LOCATIONS #
    with open(r'C:\Users\t.vd.hoeven\Dose_DL\Lists\patient_IDs.txt') as f:                             #Location of list with patient IDs
        patIDs = [line.rstrip()[1:-1] for line in f]
    patID = patIDs[N]
    struc_list = r'C:\Users\t.vd.hoeven\Dose_DL\Lists\Structures.txt'    #Location of list with structures
    PTV_struct = r'C:\Users\t.vd.hoeven\Dose_DL\Lists\PTV.txt'           #Location of list with PTV names
    data_dir = r'D:\dicomserver1417d\data'                      #Location of DICOM data
    # MODIFY STANDARDIZED ARRAY SIZE #
    input_size = [144, 64, 96]                                                                                          #Standardized size
    ### MODIFY STOP ###

    dicom_paths = get_dicom_paths(data_dir, patID)

    # load plan, structures, scan and dose
    dose = load_RTDOSE(dicom_paths['RTDOSE'])
    structures = load_structures(dicom_paths['RTSTRUCT'])

    # Make structures
    struct_arrays = struct_arr(struc_list, structures, dose)
    with open(struc_list) as s:
        structs = [line.rstrip()[1:-1] for line in s]

    ptv_array = struct_arr(PTV_struct, structures, dose)

    structures_mask = np.zeros([(len(structs)+1), input_size[0], input_size[1], input_size[2]])
    for i in range(len(structs)):
        name = structs[i]
        structures_mask[i, :, :, :], start_mod, end_mod = standard_size(input_size, struct_arrays[name])
    structures_mask[-1, :, :, :], start_mod, end_mod = standard_size(input_size, ptv_array['PTVpros+vs'])
    structures_mask = structures_mask > 0

    dose_arr = TAVSFldToNumpyNdArray(dose.Data)
    dose_arr, start_mod, end_mod = standard_size(input_size, dose_arr)
    return structures_mask, dose_arr, start_mod, end_mod