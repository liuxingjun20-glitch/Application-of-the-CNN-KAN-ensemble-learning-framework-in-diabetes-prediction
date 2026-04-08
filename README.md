# Application-of-the-CNN-KAN-ensemble-learning-framework-in-diabetes-prediction
## Project Overview
This project implements a **BorutaShap feature selection + CNN+KAN stacking ensemble framework** for diabetes risk prediction.  
It is designed to handle high-dimensional clinical tabular data and improve prediction accuracy.

## Dataset
The project uses the following dataset:

**Diabetes Health Dataset Analysis**  
- Author: Rabie El kharoua  
- Year: 2024  
- Link (Kaggle DOI): [https://doi.org/10.34740/KAGGLE/DSV/8665939](https://doi.org/10.34740/KAGGLE/DSV/8665939)  
The dataset used in this study contains comprehensive health-related data for 1,879 patients. Each patient is uniquely identified by an ID ranging from 6000 to 7878. 
The dataset includes multiple categories of variables, such as:
- Demographic characteristics
- Lifestyle factors
- Medical history
- Clinical measurements
- Medication usage
- Reported symptoms
- Quality of life indicators
- Environmental exposure factors
- Health behavior variables
- ## Code information
The main components of the code include:
- Descriptive statistical analysis of the dataset
- Data standardization
- Feature selection using the BorutaShap algorithm
- Training of deep learning models including CNN, GRU, MLP, and KAN
- Stacking ensemble learning
- Model evaluation using classification metrics such as Accuracy, Precision, Recall, F1-score, and AUC
- Experiments using all features as model inputs for comparison
- Ablation experiments for BorutaShap feature selection and the CNN+KAN ensemble framework
- ## Usage Instructions
To reproduce the experiments reported in this study, please follow these steps:
1. Clone or download this repository to your local machine.
2. Download the dataset and place the dataset file in the project directory.
3. Ensure that all required Python libraries are installed (see the Requirements section).
4. Open the provided Jupyter Notebook file in the project directory.
5. Run the notebook cells sequentially to execute the full experimental pipeline.
## Requirements
The experiments require the following Python libraries:

| Library       | Version       |
|---------------|---------------|
| numpy         | 2.3.5         |
| pandas        | 3.0.0         |
| scikit-learn  | 1.8.0         |
| scipy         | 1.17.0        |
| matplotlib    | 3.10.8        |
| seaborn       | 0.13.2        |
| tensorflow    | 2.20.0        |
| torch         | 2.10.0+cpu    |
| xgboost       | 3.1.3         |
| BorutaShap    | borutashap-modern 1.1.0 |
| KAN           | pykan 0.2.8   |
## Software Environment
-**Operating System**: Windows 11, version 10.0.26200, 64-bit (AMD64)  
- **Processor Architecture**: x86_64 / AMD64  
- **Python Version**: 3.13  
- **Computing Device**: CPU (no GPU used)  
- **Development Environment**: Jupyter Notebook 
