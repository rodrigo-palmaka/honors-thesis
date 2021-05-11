# honors-thesis

This repository will serve as a hub for my Data Science Honors Thesis, produced
through the 2020-21 academic year at the University of California, Berkeley.

# Corporate Credit Rating Prediction in the Energy Sector  

#### LICENSE: GNU General Public License v3.0

#### datasets: .xlsx files or otherwise. Data of features used for credit rating prediction.

- energ_specific_all_new.xlsx: Quarterly data query of Energy sector specific variables from 2006-2017 (same time frame for all datasets).
- ratios_2_all_energ_new.xlsx: Monthly financial ratios data for all queried energy companies. 
- ratings_all_energ_new.xlsx: Monthly S&P Long Term Issuer credit ratings.

#### images: image outputs of graphs used for fearture engineering, visualization of models, etc.

#### notebooks: this folder will house all .ipynb files containing executable code used for machine learning training.

- EDA.ipynb: Explores label distribution, PCA, and CCA.
- GDA.ipynb: Tests assumptions for LDA/QDA, performs tests with and without regularization/SMOTE.
- SVM.ipynb: Implementation and testing of baseline + ordinal SVM models. Includes hyperparameter tuning cells
- Random_Forest.ipynb: Testing of Random Forest model.
- Neural_nets_cuda.ipynb: PyTorch implementations of baseline and CORAL NNs. Notebook to be run on Google Colab GPU environment (Recommended).
- Neural_nets_cpu.ipynb: Non-Google Colab version of above. 
- LSTM_cuda.ipynb: Implementation and testing of baseline LSTM and LSTM-OR for Google Colab or local environment.
- LSTM_cpu.ipynb: Copy of above notebook.

#### Final_Prospectus.pdf: prospectus providing a summary of the goals of this project, methodology, and ethical questions.

#### Honors_Thesis_5_10.pdf: up to date version of full thesis written component.

#### Symposium_Presentation: final copy of slide deck presented at 2021 Data Science Honors Symposium.

#### requirements.txt: list of packages/dependencies.   
