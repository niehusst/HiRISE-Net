# HiRISE-Net

A personal project to replicate the results of the "Deep Mars" paper by Wagstaff et al. In this project I build a simple forward feed Convolutional Neural Network (CNN) to classify image from the Mars orbital image (HiRISE) labeled data set.

#### Contents:
- map-proj/: Directory containing individual cropped landmark images
- labels-map-proj.txt: Class labels (ids) for each landmark image
- label_data.py: Python dictionary that maps class ids to semantic names
- deps.txt: Dependencies of this project (that can be pip installled)
- classifier_model.py: The tensorflow model and data cleaning

## Author

* **Liam Niehus-Staab** - [niehusst](https://github.com/niehusst)

## Acknowledgements 
The HiRISE data used in this project comes from the DOI:

10.5281/zenodo.1048301

Idea for this project and the data originates from the following paper:

Kiri L. Wagstaff, You Lu, Alice Stanboli, Kevin Grimes, Thamme Gowda,
and Jordan Padams. "Deep Mars: CNN Classification of Mars Imagery for
the PDS Imaging Atlas." Proceedings of the Thirtieth Annual Conference
on Innovative Applications of Artificial Intelligence, 2017.

