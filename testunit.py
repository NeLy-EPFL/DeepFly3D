import unittest
import subprocess


class TestDf3d(unittest.TestCase):
    def test_cli(self):
        text = subprocess.check_output(["df3d-cli ./data/test/  -vv"], shell=True)
        self.assertTrue("Saved the pose at" in str(text))

    def test_video2d(self):
        text = subprocess.check_output(["df3d-cli ./data/test/  -vv"], shell=True)
        self.assertTrue("Saved the pose at" in str(text))

    def test_video3d(self):
        text = subprocess.check_output(["df3d-cli ./data/test/  -vv"], shell=True)
        self.assertTrue("Saved the pose at" in str(text))


if __name__ == "__main__":
    unittest.main()
