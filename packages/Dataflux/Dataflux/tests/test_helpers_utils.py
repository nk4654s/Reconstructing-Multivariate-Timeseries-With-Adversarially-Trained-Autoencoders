"""Dataflux/data/utils.py"""
import os
import unittest
import numpy as np
from Dataflux.helper import get_file, to_float

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.folder_path = 'temp'
        self.file_name = 'test_file.ext'
        self.postfix = 'postfix'

    def test_to_float(self):
        """Test to_float function"""
        value = "123.45"
        result = to_float(value)
        self.assertEqual(result, 123.45)

        value = "abc"
        result = to_float(value)
        self.assertTrue(np.isnan(result))

    def test_get_file_create_folder(self):
        """Test get_file function"""
        file_name = get_file(os.path.join(self.folder_path, self.file_name), self.postfix)

        f = open(file_name, "w", encoding="utf-8")
        f.write("test")
        f.close()

        self.assertTrue(os.path.exists(file_name))
        self.assertTrue(os.path.isfile(file_name))
        self.assertTrue(os.path.getsize(file_name)!= 0)
        os.remove(file_name)
        os.removedirs(self.folder_path)

    def test_get_file_increasing_postfix(self):
        """Test get_file function"""
        file_names = []
        for i in range(10):
            file_name = get_file(os.path.join(self.folder_path, self.file_name), self.postfix)

            f = open(file_name, "w", encoding="utf-8")
            f.write(f"test_{i}")
            f.close()

            file_names.append(file_name)

        self.assertEqual(len(file_names), len(os.listdir(self.folder_path)))
        for file_name in file_names:
            os.remove(file_name)

    def test_get_file_overwrite(self):
        """Test get_file function"""
        for i in range(10):
            file_name = get_file(os.path.join(self.folder_path, self.file_name), overwrite=True)

            f = open(file_name, "w", encoding="utf-8")
            f.write(f"{i}")
            f.close()

        self.assertEqual(1, len(os.listdir(self.folder_path)))
        f = open(file_name, "r", encoding="utf-8")
        value = int(f.read()[0])
        f.close()

        self.assertEqual(value, 9)
        os.remove(file_name)
        os.removedirs(self.folder_path)
