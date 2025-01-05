from dataclasses import dataclass
import os


@dataclass
class Paths:
    root_dir = os.path.normpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir))
    script_dir = os.path.normpath(os.path.join(root_dir, 'scripts'))
    dataset_dir = os.path.normpath(os.path.join(root_dir, 'dataset'))
    data_dir = os.path.normpath(os.path.join(root_dir, 'data'))
    dataset_file = os.path.join(dataset_dir, 'APA-DDoS-Dataset.csv')


@dataclass
class Const:
    mhz = 10e6


@dataclass
class Str:
    mhz = 10e6