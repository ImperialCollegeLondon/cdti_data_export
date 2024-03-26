import nibabel as nib

# import matplotlib.pyplot as plt
import numpy as np

# 2 slices in-vivo
path = "path_to_nii_file.nii"
Nifti_img = nib.load(path)
nii_data = np.rot90(np.array(Nifti_img.get_fdata()), k=1, axes=(0, 1))
nii_aff = Nifti_img.affine
nii_hdr = Nifti_img.header
print(nii_aff, "\n", nii_hdr)
print(nii_data.shape)


# for i in range(nii_data.shape[3]):
#     plt.subplot(1, nii_data.shape[3], i + 1)
#     plt.imshow(nii_data[:, :, 0, i])
# plt.show()
