import os
import cv2
import math
import numpy as np
from scipy.fftpack import dct, idct

IMAGE_FILENAME = "image1.png"

def main():
    base, ext = os.path.splitext(IMAGE_FILENAME)
    original_image = cv2.imread(IMAGE_FILENAME)
    blue_matrix, green_matrix, red_matrix = cv2.split(original_image)
    showImage(blue_matrix, "blue_component")
    showImage(green_matrix, "green_component")
    showImage(red_matrix, "red_component")
    # compress the same image 4 times with different ratios
    for m in [1, 2, 3, 4]:
        # compress
        blue_matrix_comp = compress(blue_matrix, m)
        green_matrix_comp = compress(green_matrix, m)
        red_matrix_comp = compress(red_matrix, m)
        compressed_image = np.stack((blue_matrix_comp, green_matrix_comp, red_matrix_comp), axis=2).astype(np.uint8)
        #showImage(compressed_image, f"{base}_comp_m{m}")

        # decompress
        blue_matrix_decomp = decompress(blue_matrix, m)
        green_matrix_decomp = decompress(green_matrix, m)
        red_matrix_decomp = decompress(red_matrix, m)
        decompressed_image = np.stack((blue_matrix_decomp, green_matrix_decomp, red_matrix_decomp), axis=2).astype(np.uint8)
        showImage(decompressed_image, f"{base}_decomp_m{m}")
        PSNR = calcPSNR(original_image, decompressed_image)
        print(f"PSNR for m = {m} is {PSNR}")

def showImage(matrix : np.ndarray, filename : str):
    cv2.imwrite(f"{filename}.png", matrix)
    cv2.imshow(f"{filename}.png", matrix)
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
            
            mask = np.zeros(64).reshape(8, 8)
            mask[0:m, 0:m] = 1
            filtered_coefficients = coefficients * mask

            compressed_matrix[start_row:start_row + sub_matrix_rows, start_col:start_col + sub_matrix_cols] = filtered_coefficients
    
    return compressed_matrix

def decompress(matrix : np.ndarray, m : int):
    
    sub_matrix_rows, sub_matrix_cols = 8, 8
    
    # Calculate the number of sub-matrices
    num_sub_matrices_rows = int(matrix.shape[0] / sub_matrix_rows)
    num_sub_matrices_cols = int(matrix.shape[1] / sub_matrix_cols)
    
    decompressed_matrix = np.zeros_like(matrix)

    # Iterate through the original matrix to extract sub-matrices
    for row in range(num_sub_matrices_rows):
        for col in range(num_sub_matrices_cols):
            # Define starting indices for the current sub-matrix
            start_row = row * sub_matrix_rows
            start_col = col * sub_matrix_cols
            
            # Extract sub-matrix using slicing
            sub_matrix = matrix[start_row:start_row + sub_matrix_rows, start_col:start_col + sub_matrix_cols]
            coefficients = DCT2(sub_matrix)
            
            mask = np.zeros(64).reshape(8, 8)
            mask[0:m, 0:m] = 1
            filtered_coefficients = coefficients * mask
                        
            image_data = IDCT2(filtered_coefficients)
            image_data = np.clip(image_data, 0, 255).astype(np.uint8)
            decompressed_matrix[start_row:start_row + sub_matrix_rows, start_col:start_col + sub_matrix_cols] = image_data

    
    return decompressed_matrix

def calcPSNR(original_image : np.ndarray, decompressed_image : np.ndarray):
    MSE = np.mean((original_image - decompressed_image) ** 2)
    PSNR = 10 * math.log10((255 ** 2) / MSE)
    return PSNR


if __name__ == "__main__":
    main()