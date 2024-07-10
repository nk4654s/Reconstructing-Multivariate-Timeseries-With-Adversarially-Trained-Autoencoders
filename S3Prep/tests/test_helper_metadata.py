"""S3Prep/helper/metadata.py"""
import shutil
import unittest
from S3Prep.helper import metadata

class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.metadata = metadata.Metadata()

    def test_put(self):
        """Test Metadata.put method"""
        self.metadata.put("test_value", metadata.Meta.FILES)
        self.assertEqual(self.metadata[metadata.Meta.FILES], "test_value")

    def test_drop(self):
        """Test Metadata.drop method"""
        self.metadata.put("test_value", metadata.Meta.FILES)
        self.metadata.drop(metadata.Meta.FILES)
        with self.assertRaises(KeyError):
            _ = self.metadata[metadata.Meta.FILES]

    def test_getitem(self):
        """Test Metadata.__getitem__ method"""
        self.metadata.put("test_value", metadata.Meta.FILES)
        self.assertEqual(self.metadata[metadata.Meta.FILES], "test_value")

    def test_save_load(self):
        """Test Metadata.save and load methods"""
        self.metadata.put("test_value", metadata.Meta.FILES)
        self.metadata.save("temp")
        loaded_metadata = metadata.Metadata()
        loaded_metadata.load("temp")
        self.assertEqual(loaded_metadata[metadata.Meta.FILES], "test_value")
        shutil.rmtree("temp")

if __name__ == '__main__':
    unittest.main()
