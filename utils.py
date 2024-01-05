import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def normalize_image(image):
    normalized_image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255)
    return normalized_image.astype(np.uint8)


def crop_image(image, top, bottom, left, rigth):
    return image[top:bottom, left:rigth]


def plot_all(img_fixed, img_moving, pts_fixed, pts_moving, plot_name):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_title(plot_name, fontsize=10)
    if len(img_fixed.shape) == 3:
        img_fixed = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2GRAY)
    if len(img_moving.shape) == 3:
        img_moving = cv2.cvtColor(img_moving, cv2.COLOR_BGR2GRAY)
    ax.imshow(img_fixed, cmap="Greens")
    ax.imshow(img_moving, alpha=0.5, cmap="Reds")
    # imshow dimension has x as the vertical axis and y as the horizontal axis.
    # Therefore use the reverse order of axis for plotting the scatter plots.
    ax.scatter([pts_fixed[:, 0]], [pts_fixed[:, 1]], s=20)
    ax.scatter([pts_moving[:, 0]], [pts_moving[:, 1]], s=20)

    # Plot arrows
    for pt_fixed, pt_moving in zip(pts_fixed, pts_moving):
        ax.arrow(
            pt_moving[0],
            pt_moving[1],
            pt_fixed[0] - pt_moving[0],
            pt_fixed[1] - pt_moving[1],
            length_includes_head=True,
            color="r",
            head_width=10,
            alpha=0.5,
        )

    plt.axis("off")
    plt.show()


def get_rotation_and_translation(X, Y):
    # Step 1: Compute mean
    mean_x = np.mean(X, axis=0)
    mean_y = np.mean(Y, axis=0)

    # Step 2: Center points at the origin
    centered_x = X - mean_x
    centered_y = Y - mean_y

    # Step 3: Compute covariance
    covarience = (centered_y.T @ centered_x) / (X.shape[0])

    # Step 4: Use SVD to represent the covariance matrix as rotations and stretching matrices
    U, _, Vt = np.linalg.svd(covarience, full_matrices=False)

    # Step 5: Remove the stretching matrix to obtain the final rotation matrix
    R = U @ Vt

    # Check for mirroring
    if np.linalg.det(R) < 0:
        # Mirror the matrix
        Vt[-1, :] *= -1
        R = U @ Vt

    # Step 6: Compute optimal translation vector
    t = mean_y - R @ mean_x

    return R, t


def transform_image(Y, t, R):
    # Get image dimensions
    height, width = Y.shape[:2]

    # Define the transformation matrix
    M = np.zeros((2, 3))
    M[:2, :2] = R
    M[:2, 2] = t

    # Perform the image transformation
    transformed_image = cv2.warpAffine(Y, M, (width, height))

    return transformed_image


def get_ssd(X, Y):
    # Ensure X and Y have the same shape
    assert X.shape == Y.shape, "Matrices must have the same shape."

    # Compute element-wise squared differences
    squared_diff = np.square(X - Y)

    # Sum up the squared differences
    ssd = np.sum(squared_diff)

    return ssd


def iterative_closest_points(
    X, Y, image_X, image_Y, tolerance=0.01, threshold=2, plot=False, verbose=False
):
    current_ssd = 0
    previous_ssd = np.inf
    iteration = 1

    # Loop continues until convergence
    while abs(current_ssd - previous_ssd) > 1:
        # Step 1: Establish correspondences using nearest neighbor
        nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(X)
        distances, indexes = nearest_neighbors.kneighbors(Y, return_distance=True)

        # To avoid duplicates, in case of multiple points y ∈ Y corresponding to a
        # single point x ∈ X keep only the closest point to x.
        distances_df = pd.DataFrame(
            data=np.hstack((indexes, distances)), columns=["indexes", "distances"]
        )
        min_distances_df = distances_df.groupby(["indexes"]).agg("min").reset_index()
        distances_df = (
            distances_df.reset_index()
            .merge(min_distances_df, on=["indexes", "distances"])
            .set_index("index")
        )

        # Remove outlier points using Z-score method
        closest_distances = distances_df["distances"].values
        z_scores = np.abs(
            (closest_distances - np.mean(closest_distances)) / np.std(closest_distances)
        )
        distances_df = distances_df[z_scores < threshold]

        X_indexes = distances_df["indexes"].values.astype("int")
        Y_indexes = distances_df.index

        X_corresponding_points = X[X_indexes]
        Y_closest_points = Y[Y_indexes]

        # Step 2: Estimate rotation and translation
        R, t = get_rotation_and_translation(X_corresponding_points, Y_closest_points)

        # Plot point correspondences
        if plot:
            plot_all(
                image_X,
                image_Y,
                X_corresponding_points,
                Y_closest_points,
                f"Point correspondences between moving and fixed images (iteration #{iteration})",
            )

        # Compute Sum of Squared Distances
        ssd = (
            get_ssd(X_corresponding_points, Y_closest_points)
            / X_corresponding_points.shape[0]
        )
        previous_ssd = current_ssd
        current_ssd = ssd
        if verbose:
            print("Sum of Squared Differences: ", ssd)

        # Step 3: Apply the obtained rotation and translation to the points and the image
        Y = (Y - t) @ R
        image_Y = transform_image(image_Y, -t, R.T)

        iteration += 1

    return Y, image_Y


def plot_side_by_side(images, titles, figsize=(20, 5)):
    fig, ax = plt.subplots(1, len(images), figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        if img.dtype == 'uint16':
            img = normalize_image(img)
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title(title)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)