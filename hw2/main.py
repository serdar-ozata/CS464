# Finds the PCAs of the provided dog images
# Results are attached. However, due to its size, the data is not put.
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
dataSize = 5239
imagesToBeReconstructed = [1, 50, 250, 500, 1000, 4096]


class PCA:

    def __init__(self):
        self.u = np.empty((4096, 4096, 3), dtype=float)
        self.s = np.empty((4096, 3), dtype=float)
        self.vh = np.empty((4096, 5239, 3), dtype=float)
        data = np.ndarray((dataSize, 4096, 3), dtype=np.float32)
        idx = 0
        for i in range(2, 1184):
            data[idx, :, :] = get_image(i, "./data/flickr_dog_")
            idx += 1
        for i in range(2, 4059):
            data[idx, :, :] = get_image(i, "./data/pixabay_dog_")
            idx += 1
        self.R = data[:, :, 0]
        self.G = data[:, :, 1]
        self.B = data[:, :, 2]

        for arr in self.getMats():
            mean = getMeanRow(arr)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    arr[i][j] = arr[i][j] - mean[i]
        print("Data read")
        pc_r = self.measurePCAAndPVE(self.R, "R", 0)
        pc_g = self.measurePCAAndPVE(self.G, "G", 1)
        pc_b = self.measurePCAAndPVE(self.B, "B", 2)
        print("PCA measured")
        self.PCs = np.empty((64, 64, 3, 10), dtype=np.float32)
        self.PCs[:, :, 0, :] = pc_r
        self.PCs[:, :, 1, :] = pc_g
        self.PCs[:, :, 2, :] = pc_b

    def getMats(self):
        return [self.R, self.G, self.B]

    def show(self):
        for i in range(10):
            plt.imshow(self.PCs[:, :, :, i])
            plt.show()

    def createImage(self, im_idx, k):
        res = np.empty((64, 64, 3), dtype=float)
        for i in range(3):
            prod = np.zeros((4096, 4096), dtype=float)
            prod[:, :k] = self.u[:, :k, i] @ np.diag(self.s[:k, i])
            r = prod @ self.vh[:, im_idx, i]
            r_max = r.max()
            r_min = r.min()
            r -= r_min
            r /= r_max - r_min
            res[:, :, i] = r.reshape((64, 64))
        plt.imshow(res)
        plt.show()

    def measurePCAAndPVE(self, arr, name, idx):
        transpose = arr.transpose()
        u, s, vh = np.linalg.svd(transpose, full_matrices=False)
        e_vals = np.linalg.eigvals(np.cov(transpose))
        total_pve = np.float32(0)
        # sigma_sum = s.sum()
        eigen_sum = e_vals.sum()
        at_least_case = False
        ten_case = False
        pc = np.empty((64, 64, 10), dtype=float)
        for i in range(4096):
            total_pve += e_vals[i] / eigen_sum
            if total_pve > 0.7 and (not at_least_case):
                print(str(i + 1) + " PCs are needed to obtain at least 70% PVE for channel " + name + ". pve: "
                      + str(total_pve))
                at_least_case = True
            if i > 9:
                ten_case = True
            else:
                print("PC " + str(i) + " of " + name + ": " + str(e_vals[i] / eigen_sum))
            if ten_case and at_least_case:
                break
        for i in range(10):
            inner_p = u[:, i]
            pc[:, :, i] = inner_p.reshape((64, 64))
            pc_max = pc[:, :, i].max()
            pc_min = pc[:, :, i].min()
            pc[:, :, i] -= pc_min
            pc[:, :, i] /= pc_max - pc_min
        self.u[:, :, idx] = u
        self.s[:, idx] = s
        self.vh[:, :, idx] = vh
        return pc


def get_image(i, dir_path):
    im_number = str(i)
    im_number = "0" * (6 - len(im_number)) + im_number
    image_path = dir_path + im_number + ".jpg"

    image = PIL.Image.open(image_path).resize((64, 64), PIL.Image.Resampling.BILINEAR)
    width, height = image.size
    pixel_values = list(image.getdata())
    pixel_values = np.array(pixel_values).reshape((width * height, 3))
    return pixel_values


def getMeanRow(arr):
    ret = np.empty(arr.shape[0], dtype=np.float32)
    for i in range(arr.shape[0]):
        ret[i] = arr[i].mean()
    return ret


def question3():
    for i in imagesToBeReconstructed:
        p.createImage(0, i)
        print(i)


p = PCA()
p.show()
