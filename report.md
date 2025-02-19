# Texture Classification Using GLCM and LBP Methods

## Introduction
The project aimed to implement and compare two machine learning-based texture analysis methods, Gray Level Co-occurrence Matrix (GLCM) and Local Binary Patterns (LBP), to classify images into three categories: stone, brick, and wood. This comparison intended to assess which method provides more accurate and reliable classification results in varying textural contexts.

## Methods
### Dataset Preparation
- A dataset of 150 images, 50 for each category (stone, brick, wood), was used.
- Images were preprocessed and split into a 70-30 train-test ratio.

### Feature Extraction
- **GLCM**: Extracted features included contrast, correlation, energy, and homogeneity. Parameters like different distances and angles were adjusted to optimize texture discrimination.
- **LBP**: Generated histograms based on the LBP codes, testing different radii and the number of neighbor points to effectively capture texture information.

### Classification
- **GLCM Features**: Used SVM for classification due to its effectiveness in handling high-dimensional data.
- **LBP Features**: Employed Random Forest classifier, leveraging its ability to manage overfitting in datasets with a high variability in feature importance.

### Evaluation Metrics
- Metrics used for evaluation included accuracy, precision, recall, and the confusion matrix, providing a comprehensive assessment of each classifier's performance.

## Results
- **GLCM Model**:
  - Achieved an accuracy of 82.22%, precision of 85.73%, and recall of 82.22%.
  - Per-Class Accuracy: Stone: 86.42%, Brick: 92.31%, Wood: 92.31%.
  - Confusion Matrix: Stone, Brick, and Wood classes showed high classification accuracy, with minor confusions between brick and wood.

- **LBP Model**:
  - Recorded lower performance with an accuracy of 62.22%, precision of 66.40%, and recall of 62.22%.
  - Per-Class Accuracy: Stone: 52.63%, Brick: 61.54%, Wood: 76.92%.
  - Confusion Matrix: Exhibited some misclassification between all three categories, particularly between stone and brick.

## Observations
- **GLCM** outperformed LBP in all evaluated metrics, suggesting that GLCM's method of capturing spatial relationships between pixel intensities provides a more robust feature set for classification tasks in this context.
- **LBP** demonstrated some utility but was less effective, possibly due to its localized and rotation-invariant features that may not capture broader textural patterns as effectively as GLCM.
- The choice of classifier also impacted performance, with SVM showing better results with GLCM features compared to Random Forest with LBP features, likely due to SVM's capability to manage the subtle nuances in high-dimensional spaces better.

## Conclusion
The analysis indicated that GLCM combined with SVM offers a more accurate approach for texture classification in the tested scenarios. This finding supports the use of GLCM in applications requiring high reliability and precision in texture classification. Future work could explore hybrid models combining both GLCM and LBP features or integrating other classification algorithms to further enhance accuracy and processing speed.
