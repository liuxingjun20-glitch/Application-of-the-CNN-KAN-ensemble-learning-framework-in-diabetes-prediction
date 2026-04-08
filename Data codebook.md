# Diabetes Dataset: Data Codebook

## Overview
This codebook provides a detailed description of all features in the diabetes dataset, including categorical variables that are coded numerically.  
It is intended to help convert numeric codes into meaningful labels for analysis and modeling.

## Dataset Features

### Demographic Details
| Feature | Description |
|---------|-------------|
| Age | Age of the patient, ranging from 20 to 90 years |
| Gender | Sex of the patient, coded as 0 = Male, 1 = Female |
| Ethnicity | Ethnic background, coded as 0 = Caucasian, 1 = African American, 2 = Asian, 3 = Other |
| SocioeconomicStatus | Socioeconomic status, coded as 0 = Low, 1 = Middle, 2 = High |
| EducationLevel | Education level, coded as 0 = None, 1 = High School, 2 = Bachelor’s, 3 = Higher |

### Lifestyle Factors
| Feature | Description |
|---------|-------------|
| BMI | Body Mass Index of the patient, ranging from 15 to 40 |
| Smoking | Smoking status, coded as 0 = No, 1 = Yes |
| AlcoholConsumption | Weekly alcohol intake in units, ranging from 0 to 20 |
| PhysicalActivity | Weekly hours of physical activity, ranging from 0 to 10 |
| DietQuality | Diet quality score, ranging from 0 to 10 |
| SleepQuality | Sleep quality score, ranging from 4 to 10 |

### Medical History
| Feature | Description |
|---------|-------------|
| FamilyHistoryDiabetes | Family history of diabetes, coded as 0 = No, 1 = Yes |
| GestationalDiabetes | History of gestational diabetes, coded as 0 = No, 1 = Yes |
| PolycysticOvarySyndrome | Presence of polycystic ovary syndrome, coded as 0 = No, 1 = Yes |
| PreviousPreDiabetes | History of pre-diabetes, coded as 0 = No, 1 = Yes |
| Hypertension | Presence of hypertension, coded as 0 = No, 1 = Yes |

### Clinical Measurements
| Feature | Description |
|---------|-------------|
| SystolicBP | Systolic blood pressure, ranging from 90 to 180 mmHg |
| DiastolicBP | Diastolic blood pressure, ranging from 60 to 120 mmHg |
| FastingBloodSugar | Fasting blood sugar, ranging from 70 to 200 mg/dL |
| HbA1c | Hemoglobin A1c, ranging from 4.0% to 10.0% |
| SerumCreatinine | Serum creatinine, ranging from 0.5 to 5.0 mg/dL |
| BUNLevels | Blood urea nitrogen (BUN), ranging from 5 to 50 mg/dL |
| CholesterolTotal | Total cholesterol, ranging from 150 to 300 mg/dL |
| CholesterolLDL | LDL cholesterol, ranging from 50 to 200 mg/dL |
| CholesterolHDL | HDL cholesterol, ranging from 20 to 100 mg/dL |
| CholesterolTriglycerides | Triglycerides, ranging from 50 to 400 mg/dL |

### Medications
| Feature | Description |
|---------|-------------|
| AntihypertensiveMedications | Use of antihypertensive medications, coded as 0 = No, 1 = Yes |
| Statins | Use of statins, coded as 0 = No, 1 = Yes |
| AntidiabeticMedications | Use of antidiabetic medications, coded as 0 = No, 1 = Yes |

### Symptoms and Quality of Life
| Feature | Description |
|---------|-------------|
| FrequentUrination | Presence of frequent urination, coded as 0 = No, 1 = Yes |
| ExcessiveThirst | Presence of excessive thirst, coded as 0 = No, 1 = Yes |
| UnexplainedWeightLoss | Presence of unexplained weight loss, coded as 0 = No, 1 = Yes |
| FatigueLevels | Fatigue levels, ranging from 0 to 10 |
| BlurredVision | Presence of blurred vision, coded as 0 = No, 1 = Yes |
| SlowHealingSores | Presence of slow-healing sores, coded as 0 = No, 1 = Yes |
| TinglingHandsFeet | Presence of tingling in hands or feet, coded as 0 = No, 1 = Yes |
| QualityOfLifeScore | Quality of life score, ranging from 0 to 100 |

### Environmental and Occupational Exposures
| Feature | Description |
|---------|-------------|
| HeavyMetalsExposure | Exposure to heavy metals, coded as 0 = No, 1 = Yes |
| OccupationalExposureChemicals | Occupational exposure to harmful chemicals, coded as 0 = No, 1 = Yes |
| WaterQuality | Quality of water, coded as 0 = Good, 1 = Poor |

### Health Behaviors
| Feature | Description |
|---------|-------------|
| MedicalCheckupsFrequency | Frequency of medical check-ups per year, ranging from 0 to 4 |
| MedicationAdherence | Medication adherence score, ranging from 0 to 10 |
| HealthLiteracy | Health literacy score, ranging from 0 to 10 |

### Diagnosis Information (Target Variable)
| Feature | Description |
|---------|-------------|
| Diagnosis | Diagnosis status for Diabetes, coded as 0 = No, 1 = Yes |

## Notes
- Numeric codes for categorical variables can be converted to factors in Python or R for analysis.  
- Continuous variables (Age, BMI, Blood Pressure, Cholesterol, etc.) remain numeric.  
