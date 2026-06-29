import numpy as np

def print_np(numpy_array):
    print(f"Array:\n{numpy_array}\nwith shape:\n{numpy_array.shape}\n")

# Create 1D Numpy Array and get shape
np1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
print_np(np1)

# Create 2D Array and get shape
np2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print_np(np2)

# Reshape 2D
np3 = np1.reshape([4,-1])
print_np(np3)

# Reshape 3D
np4 = np1.reshape([2,3,-1])
print_np(np4)

# Flatten to 1D
np5 = np4.reshape(-1)
print_np(np5)

# Loop per elements
for x in np.nditer(np4):
    print(x)