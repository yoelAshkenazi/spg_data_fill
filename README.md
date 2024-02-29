
***Classification with Missing Features***

This repository contains code for classification tasks with missing features. The classification model is implemented using PyTorch and PyTorch Geometric, with additional functionality for handling missing data.

**Requirements**

  numpy
  
  pandas
  
  torch
  
  torch-geometric
  
  torch-sparse
  
  torch-scatter
  
  matplotlib

  
**Usage**
Clone the repository: git clone https://github.com/yourusername/spg_data_fill.git

Navigate to the repository:

cd spg_data_fill

Run the main.py file from an IDE after selecting the test to simulate from temp_tester.py:

IDE: Select `main.py` and execute.

Pre-Ran Tests

This repository contains a folder with pre-ran tests, each with associated results.


**Structure**

main.py: Entry point of the code. Runs tests specified in temp_tester.py.

temp_tester.py: Contains the tests to be executed.

data_filler.py: Module for handling missing data in the dataset.

build_classifier.py: Contains code for building the classification model.

**How to Run Tests**

Open temp_tester.py and specify the test you want to simulate by uncommenting the corresponding line.

Run main.py from an IDE.

The selected test will be executed with the specified parameters.

**Classification Model**

The current tests use XGBoost (XGB) for classification. To change to a neural network model or an optimized version in the future, modify the code in build_classifier.py.


**Data Handling**

The data_filler.py file contains functionality to deal with missing data in the dataset. It provides methods to impute missing values based on different strategies.

