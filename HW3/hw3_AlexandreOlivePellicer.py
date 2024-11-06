import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.linalg import lstsq
from matplotlib.lines import Line2D
import cv2
from scipy.linalg import null_space

#####################
#POINT TO POINT
#####################
def obtain_h(x, x_prima):
    # compute h following the mathematics of the report
    a = [
    [x[0], x[1], 1, 0, 0, 0, -x[0]*x_prima[0], -x[1]*x_prima[0]],
    [0, 0, 0, x[0], x[1], 1, -x[0]*x_prima[1], -x[1]*x_prima[1]],
    [x[2], x[3], 1, 0, 0, 0, -x[2]*x_prima[2], -x[3]*x_prima[2]],
    [0, 0, 0, x[2], x[3], 1, -x[2]*x_prima[3], -x[3]*x_prima[3]],
    [x[4], x[5], 1, 0, 0, 0, -x[4]*x_prima[4], -x[5]*x_prima[4]],
    [0, 0, 0, x[4], x[5], 1, -x[4]*x_prima[5], -x[5]*x_prima[5]],
    [x[6], x[7], 1, 0, 0, 0, -x[6]*x_prima[6], -x[7]*x_prima[6]],
    [0, 0, 0, x[6], x[7], 1, -x[6]*x_prima[7], -x[7]*x_prima[7]]
    ]
    b = x_prima
    
    a_inverse = np.linalg.inv(a)

    # h = A^{-1}b
    h = np.dot(a_inverse, b)
    
    #Add known coefficient and reshape
    h = np.append(h, 1)
    h = h.reshape(3, 3)
    return h

def project(img, homography_matrix, new_img_dims=(5000, 5000, -5000, -5000)):
    # Homography matrix and image dimensions
    H_matrix = np.asarray(homography_matrix)
    original_height, original_width, _ = img.shape
    new_height, new_width, y_offset, x_offset = new_img_dims

    # Create a grid for the new image dimensions
    y_grid, x_grid = np.indices((new_height, new_width))
    y_grid = y_grid + y_offset
    x_grid = x_grid + x_offset

    # Homogeneous coordinates for the new image
    homogeneous_coords = np.vstack([x_grid.ravel(), y_grid.ravel(), np.ones(x_grid.size)])

    # Apply the inverse homography
    H_inverse = np.linalg.inv(H_matrix)
    transformed_coords = H_inverse @ homogeneous_coords
    transformed_coords /= transformed_coords[2]

    # Integer coordinates for transformed pixels
    x_transformed = transformed_coords[0].astype(int)
    y_transformed = transformed_coords[1].astype(int)

    # Mask for valid pixel locations within original image bounds
    valid_pixel_mask = (
        (x_transformed >= 0) & (x_transformed < original_width) &
        (y_transformed >= 0) & (y_transformed < original_height)
    )

    # Initialize the transformed image
    transformed_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Assign valid transformed pixels to the new image
    valid_y_grid = y_grid.ravel()[valid_pixel_mask] - y_offset
    valid_x_grid = x_grid.ravel()[valid_pixel_mask] - x_offset
    transformed_image[valid_y_grid, valid_x_grid] = img[y_transformed[valid_pixel_mask], x_transformed[valid_pixel_mask]]

    # Display the image
    plt.imshow(transformed_image)
    plt.axis('off')
    plt.show()

    return transformed_image

#Example for image 1
img = mpimg.imread('HW3_images/board_1.jpeg')

# Coordinates of point P,Q,R,S on Image 1
p = (70,420)
q = (422,1759)
r = (1356,1952)
s = (1222,139)
# Coordinates of point P,Q,R,S on Image 1 (undistorted)
p_prima = (70,420)
q_prima = (70,1804)
s_prima = (1255,420)
r_prima = (1255,1804)

pqrs = [p[0], p[1], q[0], q[1], r[0], r[1], s[0], s[1]]
pqrs_prima1 = [p_prima[0], p_prima[1], q_prima[0], q_prima[1], r_prima[0], r_prima[1], s_prima[0], s_prima[1]]
h = obtain_h(pqrs, pqrs_prima1)
print("Homography: \n",h)
img_corrected = project(img,h, new_img_dims=(5000, 5000, -2000, -3000))

#####################
#TWO STEP APPROACH
#####################


def projective_homography(points, image):
    # Extract x and y coordinates from points
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    
    # Compute lines between pairs of points
    lines = []
    for i in range(0, len(points), 2):  # Pairs of points: (1,2), (3,4), (5,6), (7,8)
        line = np.cross(points[i], points[i+1]).astype(np.float64)  # Ensure float64 dtype
        line /= line[2]  # Normalize the line
        lines.append(line)
    
    # Compute vanishing points by crossing pairs of lines
    v_as = []
    for i in range(0, len(lines), 2):  # Pairs of lines: (1,2), (3,4)
        v_a = np.cross(lines[i], lines[i+1]).astype(np.float64)  # Ensure float64 dtype
        v_a /= v_a[2]  # Normalize the vanishing point
        v_as.append(v_a)
    
    # Compute vanishing line by crossing the two vanishing points
    v_l = np.cross(v_as[0], v_as[1]).astype(np.float64)  # Ensure float64 dtype
    v_l /= v_l[2]  # Normalize the vanishing line

    # Construct the projective homography matrix using the vanishing line
    h_p = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [v_l[0], v_l[1], v_l[2]]
    ], dtype=np.float64)  # Ensure float64 dtype

    # Compute the inverse of the homography for distortion correction
    h_p_d = np.linalg.inv(h_p)

    return h_p_d

def affine_homography(points_o, image,point_pairs,homography_projective_distortion):
    
    # Initialize lists for transformed coordinates with direct array initialization
    x, y = [], []

    # Apply inverse homography and append transformed x and y coordinates for each point
    for point in points_o:
        transformed_point = np.dot(np.linalg.inv(homography_projective_distortion), np.transpose(np.array(point)))
        x.append(transformed_point[0] / transformed_point[2])  # Normalize by third coordinate
        y.append(transformed_point[1] / transformed_point[2])

    x = [x[0], x[1], x[0], x[2], x[3], x[4], x[3], x[5]]
    y = [y[0], y[1], y[0], y[2], y[3], y[4], y[3], y[5]]
    
    # Compute the cross products using a list comprehension
    cross_products = [np.cross(p1, p2) for p1, p2 in point_pairs]
    
    # Unpack the cross products into variables
    l_dash, m_dash, l_dash_dash, m_dash_dash = cross_products

    # Extract coefficients for the line equations
    l1_dash, l2_dash = l_dash[0], l_dash[1]
    m1_dash, m2_dash = m_dash[0], m_dash[1]

    l1_dash_dash, l2_dash_dash = l_dash_dash[0], l_dash_dash[1]
    m1_dash_dash, m2_dash_dash = m_dash_dash[0], m_dash_dash[1]
    
    # Construct the A matrix using perpendicular line equations
    A = np.array([
        [l1_dash * m1_dash, l1_dash * m2_dash + l2_dash * m1_dash, l2_dash * m2_dash],
        [l1_dash_dash * m1_dash_dash, l1_dash_dash * m2_dash_dash + l2_dash_dash * m1_dash_dash, l2_dash_dash * m2_dash_dash]
    ])
    
    # Find the solution for Ax=0, i.e., the nullspace of A
    s = null_space(A).flatten()  # Ensure s is flattened to a 1D array
    
    # Normalize s to avoid division by zero
    if s[-1] != 0:
        s = s / s[-1]  # Normalize the last element
    
    # Construct the S matrix from the solution vector
    if len(s) >= 3:  # Ensure s has at least 3 elements to construct S
        S = np.array([
            [s[0] / s[2], s[1] / s[2]],
            [s[1] / s[2], 1]
        ])
    else:
        raise ValueError("The null space solution vector is too short to construct the S matrix.")

    # Perform SVD on the S matrix
    U, Sigma, V_transpose = np.linalg.svd(S)
    
    # Rebuild the affine transformation matrix A from the singular values
    a_a = np.dot(U, np.dot(np.diag(np.sqrt(Sigma)), V_transpose))
    
    # Construct the affine homography matrix
    h_a = np.array([
        [a_a[0, 0], a_a[0, 1], 0],
        [a_a[1, 0], a_a[1, 1], 0],
        [0, 0, 1]
    ])
    
    return h_a

img1 = mpimg.imread('HW3_images/board_1.jpeg')

#projective homography
points = [(70,420,1),(1222,139,1),(422,1759,1),(1356,1952,1),(422,1759,1),(70,420,1),(1356,1952,1),(1222,139,1)]
homography_projective_distortion = projective_homography(points,img1)
plt.figure()
project(img1,homography_projective_distortion, new_img_dims=(5000, 5000, -2000, -2000))

plt.figure()
points_o = [(544,368,1),(591,583,1),(841,307,1),(895,647,1),(877,536,1),(602,687,1)]
h_a = affine_homography(points_o,img1,points,
                                                         homography_projective_distortion)

    
homography_affine_distortion = np.matmul(homography_projective_distortion,h_a)
project(img1,np.linalg.inv(homography_affine_distortion), new_img_dims=(5000, 5000, -1500, -3000))

#####################
#ONE STEP APPROACH
#####################

def one_step(points, image):
    # points should be a list of 15 3D points.
    
    # Compute cross products for 5 pairs of perpendicular lines
    lines = []
    for i in range(0, 15, 3):
        l = np.cross(points[i], points[i+1])
        m = np.cross(points[i], points[i+2])
        # Normalize by the third coordinate (homogeneous coordinates)
        l = l / l[2]
        m = m / m[2]
        lines.append((l, m))

    # Construct the A and B matrices
    A = []
    B = []
    for l, m in lines:
        A.append([l[0] * m[0], 0.5 * (l[1] * m[0] + l[0] * m[1]), l[1] * m[1], 0.5 * (l[2] * m[0] + l[0] * m[2]), 0.5 * (l[2] * m[1] + l[1] * m[2])])
        B.append([-l[2] * m[2]])

    A = np.matrix(A)
    B = np.matrix(B)

    # Solve for the conic coefficients
    conic_coefficients = np.linalg.inv(A) @ B
    conic_coefficients = conic_coefficients / np.amax(conic_coefficients)

    # Compute the conic matrix
    conic_matrix = np.matrix([[float(conic_coefficients[0]), float(conic_coefficients[1]) / 2, float(conic_coefficients[3]) / 2],
                              [float(conic_coefficients[1]) / 2, float(conic_coefficients[2]), float(conic_coefficients[4]) / 2],
                              [float(conic_coefficients[3]) / 2, float(conic_coefficients[4]) / 2, 1]])

    # Compute A*A^T and Av for later use
    A_A_transpose_matrix = np.matrix([[float(conic_coefficients[0]), float(conic_coefficients[1]) / 2],
                                      [float(conic_coefficients[1]) / 2, float(conic_coefficients[2])]])
    A_A_transpose_matrix = A_A_transpose_matrix / np.amax(A_A_transpose_matrix)
    A_v_matrix = np.matrix([[float(conic_coefficients[3]) / 2], [float(conic_coefficients[4]) / 2]])

    # Derive A from A*A^T using SVD
    U_matrix, Sigma_matrix, V_transpose_matrix = np.linalg.svd(A_A_transpose_matrix)
    A = U_matrix @ np.diag(np.sqrt(Sigma_matrix)) @ V_transpose_matrix

    # Derive v using A and Av
    v_vector = np.linalg.inv(A) @ A_v_matrix

    # Construct the homography matrix
    homography_matrix = np.matrix([[A[0, 0], A[0, 1], 0],
                                   [A[1, 0], A[1, 1], 0],
                                   [float(v_vector[0]), float(v_vector[1]), 1]])

    return homography_matrix

#Example with first image
img = mpimg.imread('HW3_images/board_1.jpeg')
points = [(978,1335,1), (952,1159,1), (723,1321,1), (519,1474,1), (748,1492,1), (483,1317,1), (1074,250,1), (730,325,1), (1103,619,1), (1000,636,1), (1021,851,1), (1103,619,1), (451,386,1), (734,325,1), (512,701,1)]
plt.figure()
h = one_step(points,img)
project(img, np.linalg.inv(h), new_img_dims=(6000, 6000, 0, -4000))
