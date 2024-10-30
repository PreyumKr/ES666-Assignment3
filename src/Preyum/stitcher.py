import os
import cv2
import glob
import random
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


class PanaromaStitcher():

    def __init__(self):
        self.sift = cv2.SIFT_create()
        # self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        global cnt
        cnt = 0
        FLANN_INDEX_KDTREE = 4
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def calcHomography_normalized(self, X, Y, M, N, M_hat, N_hat):
        T = [[2/N, 0, -N/2], [0, 2/M, -M/2], [0, 0, 1]]
        T_hat = [[2/N_hat, 0, -N_hat/2], [0, 2/M_hat, -M_hat/2], [0, 0, 1]]
        x_homogeneous = np.hstack((X, np.ones((X.shape[0], 1))))
        y_homogeneous = np.hstack((Y, np.ones((Y.shape[0], 1))))
        normalized_x = (T @ x_homogeneous.T).T
        normalized_y = (T_hat @ y_homogeneous.T).T
        x_norm = normalized_x[:, :2]
        y_norm = normalized_y[:, :2]

        A = []
        for i in range(len(x_norm)):
            x1, y1 = x_norm[i]
            x2, y2 = y_norm[i]
            # A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
            # A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])
            A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
            A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
        A = np.array(A)
        try:
            _, _, Vt = np.linalg.svd(A)
        except:
            return None
        normalized_homography = Vt[-1].reshape(3, 3)
        homography = np.linalg.inv(T_hat) @ normalized_homography @ T
        return homography / homography[2, 2]

    #Originally SIFT worked for gray but it gives result for color as well
    def ransac(self, X, Y, M, N, M_hat, N_hat): 
        # thresh = np.sqrt(5.99)*0.3
        thresh = 3.0
        inliers_best = []
        inliers_max = 0
        best_homography = None
        iter_max = 20000

        
        for _ in tqdm(range(iter_max), desc="Processing"):
            rand_index = np.random.choice(len(X), min(4, len(X)), replace=False)
            X_sampled = X[rand_index]
            Y_sampled = Y[rand_index]

            homography = self.calcHomography_normalized(X_sampled, Y_sampled, M, N, M_hat, N_hat)
            if homography is None:
                continue
            
            X_homogeneous = np.hstack((X, np.ones((X.shape[0], 1))))
            Yhat_homogeneous = (homography @ X_homogeneous.T).T
            Yhat_homogeneous[Yhat_homogeneous[:, 2] == 0, 2] = 1e-10
            Yhat = Yhat_homogeneous[:, :2] / Yhat_homogeneous[:, 2, np.newaxis]
            diff = np.linalg.norm(Y - Yhat, axis=1)
            inliers = np.where(diff < thresh)[0]

            if len(inliers) > inliers_max:
                inliers_max = len(inliers)
                best_homography = homography
                inliers_best = inliers

        # print(f"Selecting Best Homography from {iter_max} iterations:")
        if best_homography is not None and inliers_max >= 2:
            best_homography = self.calcHomography_normalized(X[inliers_best], Y[inliers_best], M, N, M_hat, N_hat)
        return best_homography, inliers_max
    
    def image_warp(self, img, homography_matrix, output_shape):
        output_height, output_width = output_shape
        grid_x, grid_y = np.meshgrid(np.arange(output_width), np.arange(output_height))
        horizontal_ones = np.ones_like(grid_x)
        coordinates_img2 = np.stack([grid_x, grid_y, horizontal_ones], axis=-1).reshape(-1, 3)

        inverse_homography = np.linalg.inv(homography_matrix)
        transformed_coordinates = coordinates_img2 @ inverse_homography.T
        transformed_coordinates[transformed_coordinates[:, 2] == 0, 2] = 1e-10
        transformed_coordinates /= transformed_coordinates[:, 2, np.newaxis]

        source_x = transformed_coordinates[:, 0]  
        source_y = transformed_coordinates[:, 1]

        indices_validitycheck = (
            (source_x >= 0) & (source_x < img.shape[1] - 1) &
            (source_y >= 0) & (source_y < img.shape[0] - 1)
        )

        source_x = source_x[indices_validitycheck]
        source_y = source_y[indices_validitycheck]
        source_x0 = np.floor(source_x).astype(np.int32)
        source_y0 = np.floor(source_y).astype(np.int32)
        source_x1 = source_x0 + 1
        source_y1 = source_y0 + 1

        interpolation_x = source_x - source_x0 
        interpolation_y = source_y - source_y0

        img_flat = img.reshape(-1, img.shape[2])
        source_indices = source_y0 * img.shape[1] + source_x0
        interpolation_a = img_flat[source_indices]
        interpolation_b = img_flat[source_y0 * img.shape[1] + source_x1]
        interpolation_c = img_flat[source_y1 * img.shape[1] + source_x0]
        interpolation_d = img_flat[source_y1 * img.shape[1] + source_x1]

        weight_a = (1 - interpolation_x) * (1 - interpolation_y)
        weight_b = interpolation_x * (1 - interpolation_y)
        weight_c = (1 - interpolation_x) * interpolation_y
        weight_d = interpolation_x * interpolation_y
        
        warped_pixels = (interpolation_a * weight_a[:, np.newaxis] + interpolation_b * weight_b[:, np.newaxis] +
                         interpolation_c * weight_c[:, np.newaxis] + interpolation_d * weight_d[:, np.newaxis])

        # output image
        warped_image = np.zeros((output_height * output_width, img.shape[2]), dtype=img.dtype)
        warped_image[indices_validitycheck] = warped_pixels
        warped_image = warped_image.reshape(output_height, output_width, img.shape[2])

        return warped_image
    
    def warp_images_inverse(self, img2, img1, homography_matrix):
        img1_height, img1_width = img1.shape[:2]
        img2_height, img2_width = img2.shape[:2]
        corners_img2 = np.array([[0, 0], [img2_width, 0], [img2_width, img2_height], [0, img2_height]])

        homogeneous_corners = np.hstack([corners_img2, np.ones((corners_img2.shape[0], 1))])
        transformed_corners = (homography_matrix @ homogeneous_corners.T).T
        transformed_corners[transformed_corners[:, 2] == 0, 2] = 1e-10
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2, np.newaxis]
        
        all_corners = np.vstack((transformed_corners, [[0, 0], [img1_width, 0], [img1_width, img1_height], [0, img1_height]]))
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        translated_homography = translation @ homography_matrix

        output_shape = (y_max - y_min, x_max - x_min)

        warped_img2 = self.image_warp(img2, translated_homography, output_shape)
        
        stitched_image = np.zeros((output_shape[0], output_shape[1], 3), dtype=img1.dtype)
        stitched_image[-y_min:-y_min + img1_height, -x_min:-x_min + img1_width] = img1

        # masks
        mask1 = (stitched_image > 0).astype(np.float32)
        mask2 = (warped_img2 > 0).astype(np.float32)

        # Blend images
        combined_mask = mask1 + mask2
        stitched_image = (stitched_image * mask1 + warped_img2 * mask2) / combined_mask
        stitched_image = np.nan_to_num(stitched_image).astype(np.uint8)
        return stitched_image

    def make_panaroma_for_images_in(self, path):
        global cnt
        cnt += 1
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        stitched_image = cv2.imread(all_images[0])
        homography_matrix_list = []

        for pair in range(1, len(all_images)):
            img1 = stitched_image
            img2 = cv2.imread(all_images[pair])

            M, N, _ = img1.shape
            M_hat, N_hat, _ = img2.shape

            kp1, des1 = self.sift.detectAndCompute(img1,None)
            kp2, des2 = self.sift.detectAndCompute(img2,None)

            matches = self.matcher.knnMatch(des1, des2, k=2)

            # Pick Good Matches
            good_matches= []
            for match in matches:
                pointa = match[0]
                pointb = match[1]
                if pointa.distance < 0.85 * pointb.distance:
                    good_matches.append(pointa)

            good_matches = [(point.queryIdx, point.trainIdx) for point in good_matches]
        
            X = np.float32([kp1[x].pt for (x, y) in good_matches]).reshape(-1, 2)
            Y = np.float32([kp2[y].pt for (x, y) in good_matches]).reshape(-1, 2)

            homography_matrix, _ = self.ransac(X, Y, M, N, M_hat, N_hat)
            if homography_matrix is None:
                print(f"Warning: Failed to compute homography for image {pair-1} and image {pair}. Skipping Panaroma Generation.")
                break
            homography_matrix_list.append(homography_matrix)

            stitched_image = self.warp_images_inverse(img1, img2, homography_matrix)
            cv2.imwrite(f"image_{cnt}{pair}.png", stitched_image)

        return stitched_image, homography_matrix_list