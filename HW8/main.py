import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2
from scipy.optimize import least_squares
import os
from scipy.stats import gmean
import sympy as sp

def find_intersection(hline, vline):
    # Unpack points (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    x1, y1 = hline[0][0], hline[0][1]
    x2, y2 = hline[1][0], hline[1][1]
    x3, y3 = vline[0][0], vline[0][1]
    x4, y4 = vline[1][0], vline[1][1]

    # Line 1 coefficients (A1x + B1y = C1)
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    # Line 2 coefficients (A2x + B2y = C2)
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    # Determinant (D)
    D = A1 * B2 - A2 * B1

    # Calculate the intersection point (x, y)
    x = (C1 * B2 - C2 * B1) / D
    y = (A1 * C2 - A2 * C1) / D

    return (int(x), int(y))

def group_lines(lines, part):
    clusters = []
    temp_cluster = [lines[0]]
    
    dist = []
    for k in range(len(lines) - 1):
        dist.append(lines[k + 1][0] - lines[k][0])
    threshold = np.partition(dist, -part)[-part]
    
    for line in lines[1:]:
        rho = line[0]
        prev_rho = temp_cluster[-1][0]
        if rho - prev_rho < threshold:
            temp_cluster.append(line)
        else:
            clusters.append(temp_cluster)
            temp_cluster = [line]
    if temp_cluster:
        clusters.append(temp_cluster)
    
    return clusters
    
def get_line(line_form, img, tipo):
    final_lines = []
    for lines in line_form:
        if tipo == "h":
            for i, line in enumerate(lines):
                if line[0] > 0:
                    lines[i] = [line[0], line[1]]
                else:
                    lines[i] = [-line[0], line[1] - np.pi]

        if tipo == "v":
            for i, line in enumerate(lines):
                if line[2] == 1:
                    lines[i] = [line[0], line[1]]
                else:
                    lines[i] = [line[0], line[1] - np.pi]

        rho_val = np.array([line[0] for line in lines])
        theta_val = np.array([line[1] for line in lines])
        new_rho = gmean(rho_val)
        new_theta = np.mean(theta_val)
        new_rho, new_theta = (-new_rho, new_theta + np.pi) if new_theta < 0 else (new_rho, new_theta)
        
        pt1 = (int(math.cos(new_theta) * new_rho + 5000 * (-math.sin(new_theta))), int(math.sin(new_theta) * new_rho + 5000 * (math.cos(new_theta))))
        pt2 = (int(math.cos(new_theta) * new_rho - 5000 * (-math.sin(new_theta))), int(math.sin(new_theta) * new_rho - 5000 * (math.cos(new_theta))))
        final_lines.append([pt1, pt2])
        cv2.line(img, pt1, pt2, (255 * i, 0, 255 * (1 - i)), 3, cv2.LINE_AA)
        
    return final_lines

def get_intersections(path, name):
    img = cv2.imread(path)
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 400, 300)
    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/{name}_edges.jpg', edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50, None, 0, 0)
    vertical_lines = []
    horizontal_lines = []
    img_lines = np.copy(img)
    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            if (rho < 0 and theta > 3 * np.pi / 4) or (rho > 0 and theta < np.pi / 4):
                vertical_lines.append((np.abs(lines[i][0][0]), lines[i][0][1], np.sign(rho)))
            else:
                horizontal_lines.append((lines[i][0][0], lines[i][0][1]))

            pt1 = (int(math.cos(theta) * rho + 5000 * (-math.sin(theta))), int(math.sin(theta) * rho + 5000 * (math.cos(theta))))
            pt2 = (int(math.cos(theta) * rho - 5000 * (-math.sin(theta))), int(math.sin(theta) * rho - 5000 * (math.cos(theta))))
            cv2.line(img_lines, pt1, pt2, (0, 255, 255), 4, cv2.LINE_AA)

    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/{name}_lines.jpg', img_lines)

    vdtype = [('rho', np.float32), ('theta', np.float32), ('sign', int)]
    hdtype = [('rho', np.float32), ('theta', np.float32)]
    horizontal_lines = np.sort(np.array(horizontal_lines, dtype=hdtype), axis=0)
    vertical_lines = np.sort(np.array(vertical_lines, dtype=vdtype), axis=0)

    # Aggregate lines into groups
    real_hlines = group_lines(horizontal_lines, part = 9)
    real_vlines = group_lines(vertical_lines, part = 7)

    # Get unique line
    img_final_lines = np.copy(img)
    assert len(real_hlines) == 10
    assert len(real_vlines) == 8
    hoz_lines = get_line(real_hlines, img_final_lines, "h")
    ver_lines = get_line(real_vlines, img_final_lines, "v")
    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/{name}_all_lines.jpg', img_final_lines)

    intersect = []
    img_intersec = np.copy(img)
    for hoz_line in hoz_lines:
        for ver_line in ver_lines:
            pt = find_intersection(hoz_line, ver_line)
            intersect.append(pt)
            x, y = pt  # Change these values to the pixel location you want
            # Draw the "X" by drawing two diagonal lines that cross at (x, y)
            color = (0, 255, 0)  # Green color in BGR
            thickness = 2  # Thickness of the line
            cv2.line(img_intersec, (x - 5, y - 5), (x + 5, y + 5), color, thickness)
            cv2.line(img_intersec, (x - 5, y + 5), (x + 5, y - 5), color, thickness)
            number = str(len(intersect))  # Change this to the number you want
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1
            text_position = (x + 7, y + 7)  # Position the number next to the "X"
            cv2.putText(img_intersec, number, text_position, font, font_scale, color, text_thickness)

    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/{name}_final_intersec.jpg', img_intersec)
    return intersect

# Function to find homography matrix between domain points and range points
def get_homography(d_pts, r_pts):
    A = []
    for i in range(len(r_pts)):
        x = d_pts[i][0] 
        y = d_pts[i][1]
        x_tilde = r_pts[i][0]
        y_tilde = r_pts[i][1]
        A.append([0, 0, 0, -x, -y, -1, y_tilde * x, y_tilde * y, y_tilde])
        A.append([x, y, 1, 0, 0, 0, -x_tilde * x, -x_tilde * y, -x_tilde])
    A = np.array(A)
    _, _, vh = np.linalg.svd(np.matmul(A.T, A))
    return np.reshape(vh[-1], (3, 3))

def get_homographies(data_path, world_coord):
    jpg_files = [f for f in os.listdir(data_path) if f.lower().endswith('.jpg')]
    print("Num images in dataset: ", len(jpg_files))
    jpg_files.sort()
    
    homographies = []
    intersecs_total = []
    for i, file in enumerate(jpg_files, start=1):
        path = os.path.join(data_path, file)
        intersec_points = get_intersections(path, name=f'Pic_{i}')
        intersecs_total.append(intersec_points)
        homographies.append(get_homography(world_coord, intersec_points))
    return homographies, intersecs_total

def calculate_w_matrix_coefficients(h):
    """
    Given a 3x3 homography matrix `h`, calculate the coefficients for the system of equations
    based on the relationships involving the `w` matrix.
    """
    # Extract column vectors from the homography matrix
    h1, h2 = h[:, 0], h[:, 1]
    
    # Calculate components for each equation based on the vector products
    eq1_coeffs = [
        h1[0]**2 - h2[0]**2,
        2 * (h1[0] * h1[1] - h2[0] * h2[1]),
        2 * (h1[0] * h1[2] - h2[0] * h2[2]),
        h1[1]**2 - h2[1]**2,
        2 * (h1[1] * h1[2] - h2[1] * h2[2]),
        h1[2]**2 - h2[2]**2
    ]
    
    eq2_coeffs = [
        h1[0] * h2[0],
        h1[0] * h2[1] + h1[1] * h2[0],
        h1[0] * h2[2] + h1[2] * h2[0],
        h1[1] * h2[1],
        h1[1] * h2[2] + h1[2] * h2[1],
        h1[2] * h2[2]
    ]
    
    return np.array([eq1_coeffs, eq2_coeffs])

def estimate_w(homographies):
    """
    Estimate the `w` matrix coefficients using a set of homography matrices.
    """
    lhs = []
    for h in homographies:
        # Append coefficients for each equation derived from the homography
        lhs.append(calculate_w_matrix_coefficients(h)[0])  # eq1 coefficients
        lhs.append(calculate_w_matrix_coefficients(h)[1])  # eq2 coefficients

    # Convert to a numpy array for numerical computation
    lhs = np.asarray(lhs, dtype=np.float64)
    
    # Use SVD to find the solution
    _, _, vh = np.linalg.svd(lhs)
    w_solution = vh[-1, :]  # Last row of V matrix provides the solution
    return w_solution

def estimate_k(w):
    """
    Estimate the intrinsic matrix K based on the elements of the `w` matrix.
    `w` should be a 6-element array containing [w11, w12, w13, w22, w23, w33].
    """
    # Extract individual elements from the input list `w`
    w11, w12, w13, w22, w23, w33 = w

    # Compute intermediate parameters using numeric operations
    y0 = (w12 * w13 - w11 * w23) / (w11 * w22 - w12 ** 2)
    lam = w33 - (w13 ** 2 + y0 * (w12 * w13 - w11 * w23)) / w11
    alphax = np.sqrt(lam / w11)
    alphay = np.sqrt(lam * w11 / (w11 * w22 - w12 ** 2))
    s = -(w12 * alphax ** 2 * alphay) / lam
    x0 = s * y0 / alphay - (w13 * alphax ** 2) / lam

    # Build the intrinsic matrix K using the calculated parameters
    K = np.zeros((3, 3), dtype=np.float64)
    K[0, 0] = alphax
    K[0, 1] = s
    K[0, 2] = x0
    K[1, 1] = alphay
    K[1, 2] = y0
    K[2, 2] = 1.0

    return K

def find_extrinsic_parameter(Hs, K):
    Rs = []
    ts = []
    for H in Hs:
        r1 = np.dot(np.linalg.inv(K), H[:, 0])
        scaling = 1 / np.linalg.norm(r1)
        r1 = scaling * r1
        r2 = scaling * np.dot(np.linalg.inv(K), H[:, 1])
        t = scaling * np.dot(np.linalg.inv(K), H[:, 2])
        r3 = np.cross(r1, r2)
        R = np.column_stack((r1, r2, r3))
        u, s, vh = np.linalg.svd(R)
        conditioned_R = np.matmul(u, vh)
        Rs.append(conditioned_R)
        ts.append(t)
    return Rs, ts

def get_extrinsic_param(homographies, intrinsic_matrix):
    rot = []
    trans = []
    K_inv = np.linalg.inv(intrinsic_matrix)

    for H in homographies:
        # Extract columns from the homography and scale
        r1_prime = K_inv @ H[:, 0]
        r1 = r1_prime / np.linalg.norm(r1_prime)
        r2 = K_inv @ H[:, 1] / np.linalg.norm(r1_prime)
        r3 = np.cross(r1, r2)
        t = K_inv @ H[:, 2] / np.linalg.norm(r1_prime)
        R = np.stack([r1,r2,r3], axis=1)

        # Enforce orthogonality constraint on R using SVD
        u, _, v = np.linalg.svd(R)
        R = u @ v
        
        rot.append(R)
        trans.append(t)

    return rot, trans

def apply_homography(H, points):
    num_points = len(points)
    points_homo = np.hstack([points, np.ones((num_points, 1))])  # Convert to homogeneous coordinates
    transformed_points = H @ points_homo.T  # Apply homography
    transformed_points = transformed_points.T  # Transpose back to Nx3
    return transformed_points[:, :2] / transformed_points[:, 2, np.newaxis]  # Normalize by the last row

def camera_parameters(K, rotations, translations, radio_distortion=False):
    # Camera intrinsics
    params = [K[0, 0], K[0, 1], K[0, 2], K[1, 1], K[1, 2]]
    
    # Append extrinsics for each rotation and translation pair
    for R, t in zip(rotations, translations):
        angle = np.arccos((np.trace(R) - 1) / 2)
        rotation_vector = (angle / (2 * np.sin(angle))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        params.extend(np.hstack((rotation_vector, t)))
    
    # Optionally add radial distortion parameters
    if radio_distortion:
        params.extend([0, 0])
    
    return np.array(params)

def reconstruct_R(params):
    # Intrinsics
    K = np.array([[params[0], params[1], params[2]],
                  [0, params[3], params[4]],
                  [0, 0, 1]])
    
    # Extrinsics
    rotations, translations = [], []
    for i in range(5, len(params), 6):
        rotation_vector = params[i:i+3]
        translation_vector = params[i+3:i+6]
        
        # Convert rotation vector to rotation matrix
        angle = np.linalg.norm(rotation_vector)
        skew_sym_matrix = np.array([
            [0, -rotation_vector[2], rotation_vector[1]],
            [rotation_vector[2], 0, -rotation_vector[0]],
            [-rotation_vector[1], rotation_vector[0], 0]
        ])
        R = np.eye(3) + (np.sin(angle) / angle) * skew_sym_matrix + \
            ((1 - np.cos(angle)) / (angle**2)) * skew_sym_matrix @ skew_sym_matrix
        
        rotations.append(R)
        translations.append(translation_vector)
    
    return K, rotations, translations

def remove_radio_distortion(points, k1, k2, x0, y0):
    x, y = points[:, 0], points[:, 1]
    r_squared = (x - x0)**2 + (y - y0)**2
    x_corrected = x + (x - x0) * (k1 * r_squared + k2 * r_squared**2)
    y_corrected = y + (y - y0) * (k1 * r_squared + k2 * r_squared**2)
    return np.column_stack((x_corrected, y_corrected))

def cost_function(params, all_intersec_points, world_coord, radio_distortion=False, lm = True, img_idx = -1, name = None):
    if radio_distortion:
        K, Rs, ts = reconstruct_R(params[:-2])
        k1 = params[-2]
        k2 = params[-1]
        x0 = params[2]
        y0 = params[4]
    else:
        K, Rs, ts = reconstruct_R(params)

    all_projected_points = []
    for R, t in zip(Rs, ts):
        H = np.matmul(K, np.column_stack((R[:, 0], R[:, 1], t)))
        projected_points = apply_homography(H, world_coord)
        if radio_distortion:
            projected_points = remove_radio_distortion(projected_points, k1, k2, x0, y0)
        all_projected_points.append(projected_points)

    if lm:
        all_projected_points = np.concatenate(all_projected_points, axis=0)
        all_intersec_points = np.concatenate(all_intersec_points, axis=0)
        diff = all_intersec_points - all_projected_points
        return diff.flatten()
    
    else:
        diff = all_intersec_points[img_idx] - all_projected_points[img_idx]
        dx = diff[:, 0]
        dy = diff[:, 1]
        distance = np.sqrt(dx**2 + dy**2)
        mean = np.mean(distance)
        var = np.var(distance)
        reproject(all_projected_points, img_idx, name)
        return mean, var
    
def reproject(all_projected_points, img_idx, name):
    path = f"/home/aolivepe/Computer-Vision/HW8/output/Pic_{img_idx + 1}_final_intersec.jpg"
    img = cv2.imread(path)
    for point in all_projected_points[img_idx]:
        x, y = int(point[0]), int(point[1])  # Change these values to the pixel location you want
        # Draw the "X" by drawing two diagonal lines that cross at (x, y)
        color = (0, 0, 255)  # Green color in BGR
        thickness = 2  # Thickness of the line
        cv2.line(img, (x - 5, y - 5), (x + 5, y + 5), color, thickness)
        cv2.line(img, (x - 5, y + 5), (x + 5, y - 5), color, thickness)
        # cv2.circle(img, (int(point[0]), int(point[1])), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/Pic_{img_idx + 1}{name}reproject.jpg', img)

# Example usage:
# dataset_path = "/home/aolivepe/Computer-Vision/HW8/Dataset2"
dataset_path = "/home/aolivepe/Computer-Vision/HW8/HW8-Files/Dataset1"

# Get world coordinates
x_coords = 10 * np.arange(8)
y_coords = 10 * np.arange(10)
y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
world_coord = np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)

Hs, all_intersec_points = get_homographies(dataset_path, world_coord)
w = estimate_w(Hs)
K = estimate_k(w)
print("K: ", K)
Rs, ts = get_extrinsic_param(Hs, K)
print("Rs[0]: ", Rs[0])
print("ts[0]: ", ts[0])

# Define camera parameters
params = camera_parameters(K, Rs, ts)

# Perform least squares optimization
res_ls = least_squares(cost_function, params, method='lm', args=[all_intersec_points, world_coord])

# Refine parameters
K_refined, Rs_refined, ts_refined = reconstruct_R(res_ls.x)

print(K_refined)
print(Rs_refined[0], ts_refined[0])

print(Rs_refined[7], ts_refined[7])

# Project and print mean and variance
mean, var = cost_function(params, all_intersec_points, world_coord, lm = False, img_idx=0, name='init')
print(mean, var)

mean, var = cost_function(res_ls.x, all_intersec_points, world_coord, lm = False, img_idx=0, name='refined')
print(mean, var)

mean, var = cost_function(params, all_intersec_points, world_coord, lm = False, img_idx=7, name='init')
print(mean, var)

mean, var = cost_function(res_ls.x, all_intersec_points, world_coord, lm = False, img_idx=7, name='refined')
print(mean, var)


# Define camera parameters with radial distortion
params_rd = camera_parameters(K, Rs, ts, radio_distortion=True)

# Perform least squares optimization with radial distortion
res_ls_rd = least_squares(cost_function, params_rd, method='lm', args=[all_intersec_points, world_coord, True])

print(res_ls_rd.x[-2:])

# Project and print mean and variance with radial distortion
mean, var = cost_function(res_ls_rd.x, all_intersec_points, world_coord, lm = False, radio_distortion=True, img_idx=0, name='radio_dis')
print(mean, var)

mean, var = cost_function(res_ls_rd.x, all_intersec_points, world_coord, lm = False, radio_distortion=True, img_idx=7, name='radio_dis')
print(mean, var)

def camera_poses(Rs, ts):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
       
    # # for i in range(len(R)):
    # for i in range(1):
    #     C = -R[i].T @ t[i]
    #     X_x = R[i].T @ np.array([1, 0, 0]).T + C
    #     X_y = R[i].T @ np.array([0, 1, 0]).T + C
    #     X_z = R[i].T @ np.array([0, 0, 1]).T + C
        
    #     ax.quiver(C[0], C[1], C[2], X_x[0], X_x[1], X_x[2], color="r", length=1.0)
    #     ax.quiver(C[0], C[1], C[2], X_y[0], X_y[1], X_y[2], color="g", length=1.0)
    #     ax.quiver(C[0], C[1], C[2], X_z[0], X_z[1], X_z[2], color="b", length=1.0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scale and rectangle size
    scale = 5
    rectangle_size = 2

    # Loop through each R and t pair
    for i, (R, t) in enumerate(zip(Rs, ts)):
        # Compute camera center in world coordinates
        C = -R.T @ t

        # Define camera principal axes
        X_cam = R[:, 0]  # X-axis
        Y_cam = R[:, 1]  # Y-axis
        Z_cam = R[:, 2]  # Z-axis

        # Plot camera center
        # ax.scatter(*C, color='k', marker='o', s=100, label=f"Camera Center {i+1}")

        # Plot camera axes
        ax.quiver(*C, *X_cam, color='r', length=scale, normalize=True, label=f"X_cam {i+1}")
        ax.quiver(*C, *Y_cam, color='g', length=scale, normalize=True, label=f"Y_cam {i+1}")
        ax.quiver(*C, *Z_cam, color='b', length=scale, normalize=True, label=f"Z_cam {i+1}")

        # Plot the camera principal plane (rectangle)
        rect_corners = np.array([
            [1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [-1, 1, 0]
        ]) * rectangle_size

        # Transform rectangle points from camera frame to world frame
        rect_world = C + (R @ rect_corners.T).T
        ax.plot_trisurf(rect_world[:, 0], rect_world[:, 1], rect_world[:, 2], color=np.random.rand(3), alpha=0.3)

    # Plot settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.title("3D Plot of Multiple Camera Poses")
    plt.savefig("3d_vectors_plot.jpg", format="jpg", dpi=300)
    
camera_poses(Rs, ts)
    
    