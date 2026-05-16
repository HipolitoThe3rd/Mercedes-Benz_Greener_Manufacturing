An end-to-end Machine Learning pipeline utilizing XGBoost and Principal Component Analysis (PCA) to predict the time a vehicle spends on the test bench, optimizing testing efficiency and reducing environmental impact.

\#\# 📌 Project Overview  
Since inventing the automobile in 1886, Mercedes-Benz has pioneered breakthrough automotive innovations. Today, with an immense selection of features and customizable options, every vehicle configuration built on Daimler's production line is unique. To guarantee safety and reliability, engineers utilize a robust testing system. 

However, optimizing testing speed across thousands of feature permutations is complex and time-consuming. This project addresses the challenge by implementing a powerful algorithmic approach to predict test bench times. By optimizing testing schedules, this pipeline helps accelerate production and lower carbon dioxide emissions while maintaining Mercedes-Benz’s stringent quality and safety standards.

\#\# 🛠️ Requirements & Prerequisites  
Before executing the pipeline, ensure your environment has the required data science packages installed.

* Python  
* Pandas  
* Numpy  
* Scikit-learn  
* Xgboost 

### **File Architecture**

For the script to resolve internal relative paths correctly, maintain the following directory structure:

📂 mercedes-benz-prediction/  
├── 📄 train.csv                       \# Training dataset with target configurations  
├── 📄 test.csv                        \# Testing dataset for evaluation  
└── 📄 main.py                         \# Complete solution code execution script

The two csv files in the diagram must be in the exact same folder as [main.py](http://main.py) for this program to work.

## **⚙️ Data Pipeline & Solution Workflow**

The project solution executes five core processing stages sequentially:

1. **Variance Filtering**  
   * Computes the variance across all numerical features using train\_features.var().  
   * Columns displaying zero variance ($Var \= 0$) are entirely discarded. Because every vehicle shares an identical configuration value for these metrics, they offer zero predictive power.  
2. **Data Integrity & Structural Validation**  
   * Checks for null elements using .isnull().sum().sum() to guarantee data completeness.  
   * Analyzes structural unique counts for categorical features to understand and evaluate label cardinality.  
3. **Robust Label Encoding**  
   * Encodes categorical string inputs into numeric representations required by the XGBoost algorithm.  
   * Combines unique strings from both training and testing datasets during encoder fitting to ensure the script does not crash if the test set introduces unseen categories.  
4. **Dimensionality Reduction (PCA)**  
   * The dataset features nearly 400 features, many consisting of highly collinear dummy variables.  
   * Principal Component Analysis (PCA) condenses the feature space into **12 principal components**, lowering processing overhead and mitigating model overfitting to dataset noise.  
5. **Gradient Boosted Tree Regression**  
   * Instantiates an XGBoost Regressor optimized via a continuous squared error loss objective (reg:squarederror).  
   * Generates continuous time-value predictions based on the 12 extracted PCA components and pairs results with unique vehicle IDs.

## **🚀 Execution Guide**

### **1\. Position Terminal inside the Project Workspace**

Open your terminal or command line prompt and use the change directory command (cd) to navigate into the specific folder containing your files:

cd path/to/your/folder

### **2\. Execute the Pipeline**

Run the main Python script using the following command:

python main.py

### **3\. Understanding Program Output**

* **Scenario A: Silent Execution (Blank Terminal)**  
  If the command finishes processing and drops back to a blank prompt after a brief pause without printing anything, the execution was completely successful. The pipeline completed its model training and feature encoding tasks seamlessly in the background. Check your directory for a newly generated submission.csv containing your output data\!  
* **Scenario B: Data Logging and Indicators**  
  Depending on explicit functions inside your code, you may see terminal print statements mapping performance metrics (e.g., Model R2 Score) or dataset previews (train\_df.head()), along with normal operational warnings from scikit-learn or xgboost.  
* **Scenario C: FileNotFoundError Traceback**  
  If the terminal outputs an error stating FileNotFoundError: \[Errno 2\] No such file or directory: 'train.csv', your terminal context is not looking at the correct folder directory. Re-verify your path utilizing step 1\.

# **Write to file**

with open('README.md', 'w', encoding='utf-8') as f:

f.write(readme\_content)

print("README.md generated successfully.")