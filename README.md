# Linear Regression and Image Processing Project

## Overview

This project demonstrates two main tasks:

1. **Linear Regression** to model the relationship between height and weight using a synthetic dataset.
2. **Image Processing** techniques with OpenCV to enhance and clean images through grayscale conversion, histogram equalization, binarization, and morphological operations.

---

## Project Structure

- **Linear Regression**: Loads data, splits into training and test sets, fits a linear regression model, evaluates with Mean Squared Error (MSE), and plots results.
- **Image Processing**: Reads an image, converts it to grayscale, applies histogram equalization, binarizes using Otsu's method, and applies morphological opening to remove noise.

---

## Packages Used

- `pandas`: For loading and handling CSV data.
- `numpy`: For numerical operations and array reshaping.
- `scikit-learn`: For machine learning utilities including data splitting, linear regression model, and evaluation metrics.
- `matplotlib`: For plotting data points, regression lines, and displaying images.
- `opencv-python` (cv2): For image processing operations such as reading images, grayscale conversion, histogram equalization, thresholding, and morphology.

---

## Part 1: Linear Regression

- First, we import necessary libraries: `pandas`, `numpy`, `scikit-learn` modules for modeling and evaluation, and `matplotlib` for plotting.
  
- We read a CSV file named `"LinReg_syn_data.csv"` which contains two columns: `height` and `weight`.

- We extract the features (`height`) and target variable (`weight`) from the dataset.

- The data is split into training and testing sets using `train_test_split`, reserving 30% for testing.

- We reshape the data into 2D arrays as required by scikit-learn's `LinearRegression`.

- A linear regression model is trained on the training data.

- Predictions are made on the test set.

- Model performance is evaluated using Mean Squared Error (MSE), which measures average squared difference between actual and predicted values.

- Finally, a scatter plot of the original data points (`height` vs `weight`) is plotted, along with the regression line from the test set predictions.

---

## Part 2: Image Processing with OpenCV

- We import OpenCV (`cv2`), `numpy`, and `matplotlib` for image handling and visualization.

- An image file `"py.png"` is loaded in color.

- It is converted to grayscale to simplify further processing.

- Histogram equalization is applied on the grayscale image to enhance contrast by spreading out the intensity values.

- We apply Otsu's thresholding method to convert the grayscale image into a binary image automatically selecting an optimal threshold.

- Morphological opening is performed on the binary image using a 3x3 kernel to remove small noise and smooth the object edges.

- The processed images (grayscale, histogram equalized, and morphologically opened) are saved as separate image files.

- Finally, all these images including the original are displayed in a 2x2 grid using Matplotlib for easy visual comparison.

---

### Linear Regression

- Linear regression models the linear relationship between a dependent variable (weight) and an independent variable (height).
- It minimizes the sum of squared errors between predicted and actual values.
- Mean Squared Error (MSE) is used to evaluate the model's performance: the lower the MSE, the better the model fits the data.

### Image Processing

- **Grayscale Conversion**: Converts a color image to shades of gray, simplifying further processing.
- **Histogram Equalization**: Improves image contrast by redistributing pixel intensities.
- **Binarization (Thresholding)**: Converts grayscale image to binary (black & white) based on pixel intensity, using Otsu's method for automatic threshold selection.
- **Morphological Operations**: Processes images based on shapes. The opening operation removes noise by eroding then dilating the image with a structuring element (kernel).

---

## How to Run

1. Ensure you have the dataset file `LinReg_syn_data.csv` and the image file `py.png` in your project directory.
2. Install required Python packages (if not already installed):

    ```bash
    pip install pandas numpy scikit-learn matplotlib opencv-python
    ```

3. Run your Python script containing the provided code.
4. Observe:
   - The printed Mean Squared Error for the regression model.
   - A scatter plot showing original data and regression line.
   - Four displayed images: original, grayscale, histogram equalized, and morphologically processed.

---

## Future Improvements

- Expand regression to multiple variables.
- Add interactive visualization.
- Explore other morphological operations and image filters.
- Include error handling for missing files or invalid inputs.

---

Feel free to contribute, raise issues, or request features!
