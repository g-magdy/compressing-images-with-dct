import os
import cv2
import numpy as np
from scipy.fftpack import dct, idct

IMAGE_FILENAME = "test.png"

def main():
    
    # compress the same image 4 times with different ratios
    for m in [1, 2, 3, 4]:
        compressImage(IMAGE_FILENAME, m)
            
    
def compressImage(filename : str, m : int):
    original_image = cv2.imread(filename)
    blue_matrix, green_matrix, red_matrix = cv2.split(original_image)
    base, ext = os.path.splitext(filename)
    r_matrix = compress(red_matrix, m)
    g_matrix = compress(green_matrix, m)    
    b_matrix = compress(blue_matrix, m)
    rgb_image = np.stack((b_matrix, g_matrix, r_matrix), axis=2).astype(np.uint8)
    cv2.imwrite(f"{base}_comp_m_{m}.png", rgb_image)
    cv2.imshow(f"{base}_comp_m_{m}.png", rgb_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    

def DCT2(matrix : np.ndarray):
    matrix = matrix.astype(float)
    return dct(dct(matrix.T, norm="ortho").T, norm="ortho")

def IDCT2(coefficients : np.ndarray):
    return idct(idct(coefficients.T, norm="ortho").T, norm="ortho")

def compress(matrix : np.ndarray, m : int):
    
    sub_matrix_rows, sub_matrix_cols = 8, 8
    
    # Calculate the number of sub-matrices
    num_sub_matrices_rows = int(matrix.shape[0] / sub_matrix_rows)
    num_sub_matrices_cols = int(matrix.shape[1] / sub_matrix_cols)
    
    compressed_matrix = np.zeros_like(matrix)

    # Iterate through the original matrix to extract sub-matrices
    for row in range(num_sub_matrices_rows):
        for col in range(num_sub_matrices_cols):
            # Define starting indices for the current sub-matrix
            start_row = row * sub_matrix_rows
            start_col = col * sub_matrix_cols
            
            # Extract sub-matrix using slicing
            sub_matrix = matrix[start_row:start_row + sub_matrix_rows, start_col:start_col + sub_matrix_cols]
            coefficients = DCT2(sub_matrix)
            
            # FILTERING STEP: 
            # use a bit mask to keep only the first (m) coefficients
            mask = np.zeros(64).reshape(8, 8)
            mask[0:m, 0:m] = 1
            filtered_coefficients = coefficients * mask
            
            # get a new compressed block and add it to the compressed matrix
            image_data = IDCT2(filtered_coefficients)
            image_data = np.clip(image_data, 0, 255).astype(np.uint8)
            compressed_matrix[start_row:start_row + sub_matrix_rows, start_col:start_col + sub_matrix_cols] = image_data
    
    return compressed_matrix


if __name__ == "__main__":
    main()