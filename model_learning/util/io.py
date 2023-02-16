import gzip
import json
import os
import pickle
import re
import shutil
import tempfile
import zipfile
import numpy as np
from typing import List, Dict

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class _NpEncoder(json.JSONEncoder):
    """
    Supports encoding of numpy data types.
    See: https://stackoverflow.com/a/57915246/16031961
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_file_changed_extension(file_path: str, ext: str, prefix: str = '', suffix: str = '') -> str:
    """
    Changes the extension of the given file.
    :param str file_path: the path to the file.
    :param str ext: the new file extension.
    :param str prefix: the prefix to add to the resulting file name.
    :param str suffix: the suffix to add to the resulting file name.
    :rtype: str
    :return: the file path with the new extension.
    """
    ext = ext.replace('.', '')
    return os.path.join(os.path.dirname(file_path),
                        f'{prefix}{get_file_name_without_extension(file_path)}{suffix}.{ext}')


def get_file_name_without_extension(file_path: str) -> str:
    """
    Gets the file name in the given path without extension.
    :param str file_path: the path to the file.
    :rtype: str
    :return: the file name in the given path without extension.
    """
    return os.path.basename(file_path).split('.')[0]


def get_files_with_extension(dir_path: str, extension: str, sort: bool = True) -> List[str]:
    """
    Gets all files in the given directory with a given extension.
    :param str dir_path: the directory from which to retrieve the files.
    :param str extension: the extension of the files to be retrieved.
    :param bool sort: whether to sort list of files based on file name.
    :rtype: list[str]
    :return: the list of files in the given directory with the required extension.
    """
    file_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.' + extension)]
    if sort:
        file_list.sort()
    return file_list


def get_directory_name(dir_path: str) -> str:
    """
    Gets the directory name in the given path.
    :param str dir_path: the path (can be a file).
    :rtype: str
    :return: the directory name in the given path.
    """
    return os.path.basename(os.path.dirname(dir_path))


def create_clear_dir(dir_path: str, clear: bool = False):
    """
    Creates a directory in the given path. If it exists, optionally clears the directory.
    :param str dir_path: the path to the directory to create/clear.
    :param bool clear: whether to clear the directory if it exists.
    :return:
    """
    if clear and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def delete_file(file_path: str):
    """
    Deletes the file in the given path if it exists.
    :param str file_path: the path to the file to be deleted.
    :return:
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def save_dict_json(dictionary: Dict, file_path: str):
    """
    Saves the given dictionary to a json file.
    :param dict dictionary: the dictionary to save.
    :param str file_path: the path to the json file where to save the dictionary.
    """
    with open(file_path, 'w') as fp:
        json.dump(dictionary, fp, indent=4, cls=_NpEncoder)


def save_object(obj, file_path: str, compress_gzip: bool = True):
    """
    Saves a pickle binary file containing the given data.
    :param obj: the object to be saved.
    :param str file_path: the path of the file in which to save the data.
    :param bool compress_gzip: whether to gzip the output file.
    """
    with gzip.open(file_path, 'wb') if compress_gzip else open(file_path, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(file_path: str):
    """
    Loads an object from the given pickle file, possibly gzip compressed.
    :param str file_path: the path to the file containing the data to be loaded.
    :return: the data loaded from the file.
    """
    try:
        with gzip.open(file_path, 'rb') as file:
            return pickle.load(file)
    except OSError:
        with open(file_path, 'rb') as file:
            return pickle.load(file)


def save_object_json(obj, file_path: str, compress_gzip: bool = True):
    """
    Saves a file containing the given data in a JSON format.
    :param obj: the object to be saved.
    :param str file_path: the path of the file in which to save the data.
    :param bool compress_gzip: whether to gzip the output file.
    """
    import jsonpickle
    jsonpickle.set_preferred_backend('json')
    jsonpickle.set_encoder_options('json', indent=4, sort_keys=False)
    with gzip.open(file_path, 'wt') if compress_gzip else open(file_path, 'w') as file:
        json_str = jsonpickle.encode(obj)
        file.write(json_str)


def load_object_json(file_path: str):
    """
    Loads an object from the given JSON file, possibly gzip compressed.
    :param str file_path: the path to the file containing the data to be loaded.
    :return: the data loaded from the file.
    """
    import jsonpickle
    try:
        with gzip.open(file_path, 'rt') as file:
            return jsonpickle.decode(file.read())
    except OSError:
        with open(file_path, 'r') as file:
            return jsonpickle.decode(file.read())


def compress_files(input_files: List[str], file_path: str, keep_structure: bool = True):
    """
    Creates a zip compressed file containing the provided input files.
    :param list[str] input_files: a list with the paths to the input files to be compressed.
    :param str file_path: the path to the output zip archive containing the compressed files.
    :param bool keep_structure: whether to keep the same directory structure of the input files in the zip archive. If
    `False`, then files will all be saved at the root level inside the archive, with directory structure used to rename
    the files, e.g., `path/to/file.txt` will be saved as `path_to_file.txt`.
    """
    with zipfile.ZipFile(file_path, mode='w') as zf:
        for input_file in input_files:
            file_name = input_file if keep_structure else re.sub(r'[\\|/]', '_', input_file)
            zf.write(input_file, file_name, compress_type=zipfile.ZIP_DEFLATED)


def extract_files(file_path: str, output_dir: str = None) -> str:
    """
    Extracts files from a compressed zip archive.
    :param str file_path: the path to the input zip archive containing the compressed files.
    :param str or None output_dir: the path to the directory in which to extract the files. If `None`, files will be
    extracted to a temporary directory.
    :rtype: str
    :return: the path to the directory into which the files were extracted.
    """
    if output_dir is None:
        output_dir = tempfile.TemporaryDirectory().name
    with zipfile.ZipFile(file_path, mode='r') as zf:
        zf.extractall(str(output_dir))
    return output_dir


def replace_lines(file_path: str, pattern: str, subst: str, keep_old: bool = True):
    """
    Replaces lines matching the given pattern in a text file.
    From: https://stackoverflow.com/a/39110/16031961
    :param str file_path: the path to the file in which to replace the lines.
    :param str pattern: the text pattern to be replaced.
    :param str subst: the text to replace whenever the pattern is found.
    :param bool keep_old: whether to keep a backup of the old file in the same directory.
    :return:
    """
    # checks file
    if not os.path.isfile(file_path):
        raise ValueError(f'Cannot find specified file: {file_path}')

    # create temp file
    fh, abs_path = tempfile.mkstemp()
    with os.fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))

    # copy the file permissions from the old file to the new file
    shutil.copymode(file_path, abs_path)

    # rename or remove original file
    if keep_old:
        i = 1
        backup_file = f'{file_path}.bak'
        while os.path.isfile(backup_file):
            backup_file = f'{file_path}.bak{i}'
            i += 1
        shutil.move(file_path, backup_file)
    else:
        os.remove(file_path)

    # move new file
    shutil.move(abs_path, file_path)
