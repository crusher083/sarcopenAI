import os
from typing import List
from numpy import ndarray, newaxis, zeros
from nrrd import read
from pydicom import read_file


def dicom_list(data_dir: str) -> List[str]:
    '''Locate all .dcm files in directory and return filenames as list'''
    dicoms = []

    for dirs, subdirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                dicoms.append(os.path.join(data_dir, filename))
    return sorted(dicoms)


def nrrd_list(data_dir: str, region: str) -> List[str]:
    '''Locate referenced .nrrd files in directory'''
    nrrd_list = []

    for dirs, subdirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if region.lower() in filename.lower():
                nrrd_list.append(os.path.join(data_dir, filename))

    return sorted(nrrd_list)


def list_as_arr(filenames_list: List[str]) -> ndarray:
    '''Load all DICOM files from list into np.ndarray channels last format'''
    if filenames_list[0].endswith('.dcm'):
        ref_dcm = read_file(filenames_list[0])
        # Load dimensions based on the number of rows & columns
        dims = (len(filenames_list), int(ref_dcm.Rows), int(ref_dcm.Columns))
        array = zeros(dims, dtype=ref_dcm.pixel_array.dtype)

        for filename in filenames_list:
            # read the file
            ds = read_file(filename)
            # store the raw image data
            array[filenames_list.index(filename), :, :] = ds.pixel_array

    elif filenames_list[0].endswith('.nrrd'):
        ref_nrrd, header = read(filenames_list[0])
        # Load dimensions based on the number of rows & columns
        dims = (len(filenames_list), ref_nrrd.shape[0], ref_nrrd.shape[1])
        array = zeros(dims)

        for filename in filenames_list:
            data, header = read(filename)
            array[filenames_list.index(filename), :, :] = data.T

    return array[:, :, :, newaxis]


def load_dicom(data_dir: str) -> ndarray:
    '''Load all DICOM images'''
    list_from_dir = dicom_list(data_dir)
    dicom_array = list_as_arr(list_from_dir)
    print(f'{len(list_from_dir)} DICOM images loaded.')
    return dicom_array


def load_nrrd(data_dir: str, region: str) -> ndarray:
    '''Load all .nrrd masks for specified region'''
    list_from_dir = nrrd_list(data_dir, region)
    nrrd_array = list_as_arr(list_from_dir)
    print(f'{len(list_from_dir)} .nrrd masks from {region} loaded.')
    return nrrd_array


if __name__ == '__main__':
    load_dicom(data_dir)