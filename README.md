# Introduction
This is the Python version of evaluation.m for [Sphereface](https://github.com/wy1iu/sphereface).

# Requirements
* You need to install lmdb by typing command `pip install lmdb`.
* You need to use **extract_features.bin** in caffe to extract 512-d feature vector from **layer fc5**. The command can be found in **file** folder.
* You need to change the **root_dir** in the main function to your own path.

# Usage
```python
# python lfw_acc_test.py
```
# Result
The result is shown in result.png
