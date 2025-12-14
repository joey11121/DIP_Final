import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM


def gaussian_filter_density(img, points):
    """
    Generate density map using adaptive Gaussian kernels (k-nearest-neighbor).
    """
    img_shape = [img.shape[0], img.shape[1]]
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)

    print(f"Image size: {img_shape}, total points = {gt_count}")

    if gt_count == 0:
        return density

    # Build KDTree
    leafsize = 2048
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    distances, _ = tree.query(points, k=4)

    print("Generating density...")

    for i, pt in enumerate(points):
        y, x = int(pt[1]), int(pt[0])
        if y >= img_shape[0] or x >= img_shape[1]:
            continue

        pt2d = np.zeros(img_shape, dtype=np.float32)
        pt2d[y, x] = 1.

        if gt_count > 3:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = 15

        density += gaussian_filter(pt2d, sigma, mode='constant')

    print("Done.")
    return density


def safe_load_mat(mat_path):
    """Try reading .mat file, return None if missing."""
    if not os.path.exists(mat_path):
        print(f"⚠️  Warning: Missing MAT file: {mat_path}")
        return None
    return io.loadmat(mat_path)


if __name__ == "__main__":

    root = "/eva_data/joyz/MCNN-pytorch/data/ShanghaiTech"

    part_A_train = os.path.join(root, "part_A/train_data/images")
    part_A_test  = os.path.join(root, "part_A/test_data/images")

    path_sets = [part_A_train, part_A_test]

    img_paths = []
    for path in path_sets:
        img_paths.extend(glob.glob(os.path.join(path, "*.jpg")))

    print(f"Found {len(img_paths)} images.")

    for img_path in img_paths:
        print("\nProcessing:", img_path)

        mat_path = (
            img_path.replace(".jpg", ".mat")
                    .replace("images", "ground-truth")
                    .replace("IMG_", "GT_IMG_")
        )

        mat = safe_load_mat(mat_path)
        if mat is None:
            continue  # 跳過缺失的 MAT

        img = plt.imread(img_path)
        points = mat["image_info"][0, 0][0, 0][0]

        k = gaussian_filter_density(img, points)

        # Output path for .npy
        out_path = (
            img_path.replace(".jpg", ".npy")
                    .replace("images", "ground-truth")
        )

        np.save(out_path, k)
        print(f"Saved density map → {out_path}")
