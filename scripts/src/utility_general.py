import os
from datetime import datetime


def get_current_date():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')


def check_folder_exists_and_create(path: str):
    if not os.path.exists(path):
        os.mkdir(path)
