
# Efficiency Analysis and Classification with Machine Learning

The title reflects the ambition of creating a versatile and innovative machine learning tool focused on decision trees, with applications ranging from automobile efficiency analysis to broader classification and regression problems. It suggests a forward-thinking approach to automating machine learning processes and emphasizes the project's contribution to decision tree algorithms and their applications.

## Introduction
This project explores the application of machine learning techniques to analyze and predict automobile efficiency based on various vehicle characteristics. Utilizing the `auto-mpg.data`, a well-known dataset in the machine learning community, we delve into regression and classification problems, employing different models and evaluating their performance. This project also includes a comprehensive implementation and analysis task based on decision trees, addressing various scenarios of feature and output types.

## Assignment Overview
The project encompasses a series of tasks aimed at implementing and evaluating a decision tree algorithm from scratch, covering cases with discrete and real features and outputs. The decision tree implementation should support both Gini Index and Information Gain as criteria for node splitting and include capabilities for visualization.

### Tasks
1. **Decision Tree Implementation (`tree/base.py`):**
   - Implement the decision tree to handle:
     i) discrete features, discrete output;
     ii) discrete features, real output;
     iii) real features, discrete output;
     iv) real features, real output.
   - Support Gini Index and Information Gain for splitting.
   - Enable decision tree visualization.

2. **Performance Metrics (`metrics.py`):**
   - Complete the implementation of performance metrics functions.

3. **Usage and Validation (`usage.py` & `classification-exp.py`):**
   - Utilize these scripts to validate the decision tree implementation and performance metrics.

4. **Dataset Generation:**
   - Generate datasets using `sklearn.datasets.make_classification` for experimentation.

5. **Experiments (`experiments.py`, `auto-efficiency.py`):**
   - Conduct runtime complexity analysis.
   - Apply the decision tree to the automobile efficiency problem.
   - Compare the custom decision tree model with scikit-learn's decision tree.

6. **Subjective Answers:**
   - Document timing analysis and display plots in `assignment_q<question-number>_subjective_answers.md`.

## Project Structure and Tasks
- `tree` (Directory): Contains the decision tree module, including the base class and utility functions. Key tasks involve implementing the decision tree algorithm capable of handling different types of data and criteria for splitting.
- `metrics.py`: Implement performance metrics to evaluate the decision tree's predictions.
- `usage.py`: Demonstrates the application of the decision tree on generated datasets, including training, testing, and evaluation.
- `classification-exp.py`: Contains classification experiments, including the usage of the decision tree for the automotive efficiency problem and comparison with scikit-learn's implementation.
- `experiments.py`: Focuses on experiments to analyze the runtime complexity of the decision tree algorithm, varying the number of samples (N) and features (M).

### Experiments and Results
- **Dataset Usage:** The project includes detailed instructions on generating a dataset for training and testing the decision tree, specifying the split for training and testing purposes, and evaluating the model's performance using accuracy, precision, and recall metrics.
- **Cross-Validation:** Implement 5-fold cross-validation to determine the optimum depth of the decision tree, including nested cross-validation for thorough analysis.
- **Automotive Efficiency Problem:** Apply the decision tree to analyze automobile efficiency, comparing the custom implementation with scikit-learn's decision tree module.
- **Runtime Complexity Analysis:** Conduct experiments on the decision tree's runtime complexity, comparing empirical results with theoretical expectations across different data scenarios.


## Setup and Installation
To set up the project environment:
1. Ensure Python 3.6+ is installed.
2. Install required dependencies: `pip install -r requirements.txt` (Note: The actual command may vary based on the project's dependency management tool, e.g., Poetry, indicated by `pyproject.toml`).

## Usage
To replicate the analysis and results:
- Run `python usage.py` to see an example workflow, including data loading, model training, and evaluation.
- Execute `python experiments.py` to perform comprehensive experiments and generate performance reports.

## Results
The project aims to provide insights into the efficiency of automobiles, with models evaluated based on metrics like accuracy, precision, recall, and F1 score. Generated plots (e.g., `learn_M.png`, `predict_M.png`) visualize the models' learning processes and predictive capabilities.

## Contributing
Paras Gupta and Tarun Sharma, IIT Gandhinagar.

## License
This project is open source and available under the [MIT License](LICENSE).
