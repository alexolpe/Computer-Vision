import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from scipy.optimize import least_squares
import os
from scipy.stats import gmean

# function to find the intersection of 2 lines given 2 points to define each line
def find_intersection(hline, vline):
    x1, y1 = hline[0][0], hline[0][1]
    x2, y2 = hline[1][0], hline[1][1]
    x3, y3 = vline[0][0], vline[0][1]
    x4, y4 = vline[1][0], vline[1][1]

    # Create first line
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    # Create second line
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    # Find intersections
    D = A1 * B2 - A2 * B1
    x = (C1 * B2 - C2 * B1) / D
    y = (A1 * C2 - A2 * C1) / D

    return (int(x), int(y))

# Group lines that correspond to the same true line
def group_lines(lines, part):
    clusters = []
    temp_cluster = [lines[0]]
    
    # Store rho distances between lines and set threshold in those places where distance between lines is higher (should correspond to different groups)
    dist = []
    for k in range(len(lines) - 1):
        dist.append(lines[k + 1][0] - lines[k][0])
    threshold = np.partition(dist, -part)[-part]
    
    # Group lines depending on the threshold and distance
    for line in lines[1:]:
        rho = line[0]
        prevrho = temp_cluster[-1][0]
        if rho - prevrho < threshold:
            temp_cluster.append(line)
        else:
            clusters.append(temp_cluster)
            temp_cluster = [line]
    if temp_cluster:
        clusters.append(temp_cluster)
    
    #return found groups
    return clusters
    
# Function to given a group of lines find the true line
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
        # Compute the new rho and theta as the average of the given lines
        new_rho = gmean(rho_val)
        new_theta = np.mean(theta_val)
        new_rho, new_theta = (-new_rho, new_theta + np.pi) if new_theta < 0 else (new_rho, new_theta)
        
        pt1 = (int(math.cos(new_theta) * new_rho + 5000 * (-math.sin(new_theta))), int(math.sin(new_theta) * new_rho + 5000 * (math.cos(new_theta))))
        pt2 = (int(math.cos(new_theta) * new_rho - 5000 * (-math.sin(new_theta))), int(math.sin(new_theta) * new_rho - 5000 * (math.cos(new_theta))))
        
        # Get the points for that line and store it
        final_lines.append([pt1, pt2])
        # Draw line
        cv2.line(img, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)
        
    return final_lines

# Main function to find the corners of the calibration pattern for each image of the dataset
def get_corners(path, name):
    img = cv2.imread(path)
    # Apply canny to gray image
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 400, 300)
    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/{name}_edges.jpg', edges)

    # Use Hough transform to get the lines given the edge image. Classify images in vertical or horizontal depending on the value of rho and theta
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50, None, 0, 0)
    vlines = []
    hlines = []
    img_lines = np.copy(img)
    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            if (rho < 0 and theta > 3 * np.pi / 4) or (rho > 0 and theta < np.pi / 4):
                vlines.append((np.abs(lines[i][0][0]), lines[i][0][1], np.sign(rho)))
            else:
                hlines.append((lines[i][0][0], lines[i][0][1]))

            pt1 = (int(math.cos(theta) * rho + 5000 * (-math.sin(theta))), int(math.sin(theta) * rho + 5000 * (math.cos(theta))))
            pt2 = (int(math.cos(theta) * rho - 5000 * (-math.sin(theta))), int(math.sin(theta) * rho - 5000 * (math.cos(theta))))
            cv2.line(img_lines, pt1, pt2, (0, 255, 255), 4, cv2.LINE_AA)

    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/{name}_lines.jpg', img_lines)

    hlines = np.sort(np.array(hlines, dtype=[('', np.float32), ('', np.float32)]), axis=0)
    vlines = np.sort(np.array(vlines, dtype=[('', np.float32), ('', np.float32), ('', int)]), axis=0)

    # Create clusters of the found lines
    real_hlines = group_lines(hlines, part = 9)
    real_vlines = group_lines(vlines, part = 7)

    # Get the true line for each cluster of lines
    img_final_lines = np.copy(img)
    assert len(real_hlines) == 10
    assert len(real_vlines) == 8
    hoz_lines = get_line(real_hlines, img_final_lines, "h")
    ver_lines = get_line(real_vlines, img_final_lines, "v")
    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/{name}_all_lines.jpg', img_final_lines)

    # Find the intersection of lines and plot it in the original image
    intersect = []
    img_intersec = np.copy(img)
    for hoz_line in hoz_lines:
        for ver_line in ver_lines:
            pt = find_intersection(hoz_line, ver_line)
            intersect.append(pt)
            x, y = pt
            color = (0, 0, 255)
            thickness = 1
            cv2.line(img_intersec, (x - 5, y - 5), (x + 5, y + 5), color, thickness)
            cv2.line(img_intersec, (x - 5, y + 5), (x + 5, y - 5), color, thickness)
            number = str(len(intersect))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1
            text_position = (x + 7, y + 7)
            cv2.putText(img_intersec, number, text_position, font, font_scale, color, text_thickness)

    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/{name}_final_intersec.jpg', img_intersec)
    return intersect

# Find homography from domain and range points
def get_homography(d_pts, r_pts):
    mat_A = []
    for i in range(len(r_pts)):
        mat_A.append([0, 0, 0, -d_pts[i][0], -d_pts[i][1], -1, r_pts[i][1] * d_pts[i][0], r_pts[i][1] * d_pts[i][1], r_pts[i][1]])
        mat_A.append([d_pts[i][0], d_pts[i][1], 1, 0, 0, 0, -r_pts[i][0] * d_pts[i][0], -r_pts[i][0] * d_pts[i][1], -r_pts[i][0]])
    mat_A = np.array(mat_A)
    # Homography given by the last column vector of the matrix V after doing SVD decomposition
    _, _, v = np.linalg.svd(mat_A.T @ mat_A)
    return np.reshape(v[-1], (3, 3))

# Function to get the homographies and intersection points of all the images in the given dataset
def get_homographies(data_path, world_coord):
    jpg_files = [f for f in os.listdir(data_path) if f.lower().endswith('.jpg')]
    print("Num images in dataset: ", len(jpg_files))
    jpg_files.sort()
    
    homographies = []
    intersecs_total = []
    for i, file in enumerate(jpg_files, start=1):
        path = os.path.join(data_path, file)
        intersec_points = get_corners(path, name=f'Pic_{i}')
        intersecs_total.append(intersec_points)
        homographies.append(get_homography(world_coord, intersec_points))
    return homographies, intersecs_total

# Function to define the matrices needed to estimate w
def calculate_w_matrix_coefficients(h):
    h1, h2 = h[:, 0], h[:, 1]
    
    # Matrices written following the equations explained in the report
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

# Estimate w given all the homographies
def estimate_w(homographies):
    lhs = []
    for h in homographies:
        lhs.append(calculate_w_matrix_coefficients(h)[0])  
        lhs.append(calculate_w_matrix_coefficients(h)[1])

    lhs = np.asarray(lhs, dtype=np.float64)
    
    # Use last vector of V in SVD to find w
    _, _, v = np.linalg.svd(lhs)
    w_solution = v[-1, :] 
    return w_solution

# Estimate K given w. First calculate all the coefficients following the equations from the report and then form matrix K
def estimate_k(w):
    w11, w12, w13, w22, w23, w33 = w

    y0 = (w12 * w13 - w11 * w23) / (w11 * w22 - w12 ** 2)
    lam = w33 - (w13 ** 2 + y0 * (w12 * w13 - w11 * w23)) / w11
    alphax = np.sqrt(lam / w11)
    alphay = np.sqrt(lam * w11 / (w11 * w22 - w12 ** 2))
    s = -(w12 * alphax ** 2 * alphay) / lam
    x0 = s * y0 / alphay - (w13 * alphax ** 2) / lam

    K = np.array([[alphax, s, x0],
                  [0, alphay, y0],
                  [0, 0, 1]])
    return K

# Estimate the extrinsic parameters for each image given the homography and K. Compute parameters following equations from the report
def estimate_extrinsic_param(homographies, K):
    rot = []
    trans = []
    K_inv = np.linalg.inv(K)

    for H in homographies:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        r1 = K_inv @ h1 / np.linalg.norm(K_inv @ h1)
        r2 = K_inv @ h2 / np.linalg.norm(K_inv @ h1)
        r3 = np.cross(r1, r2)
        t = K_inv @ h3 / np.linalg.norm(K_inv @ h1)
        R = np.stack([r1,r2,r3], axis=1)

        # Enforce orthogonality
        u, _, v = np.linalg.svd(R)
        R = u @ v
        
        rot.append(R)
        trans.append(t)

    return rot, trans

# Create vector with all the parameters for each image in the dataset. This is needed for the optimization algorithm
def param_cam(K, rots, trans):
    p = [K[0, 0], K[0, 1], K[0, 2], K[1, 1], K[1, 2]]
    # Use Rodrigues Representation for R
    for R, t in zip(rots, trans):
        p.extend(np.hstack(((np.arccos((np.trace(R) - 1) / 2) / (2 * np.sin(np.arccos((np.trace(R) - 1) / 2)))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]), t)))
    return p

# Given the flettened vector p, reconstruct the parameters K, R and t
def reconstruct_p(p):
    K = np.array([[p[0], p[1], p[2]],
                  [0, p[3], p[4]],
                  [0, 0, 1]])

    rotation_matrices, translation_vectors = [], []
    step_size = 6
    for idx in range(5, len(p), step_size):
        rot_vec = p[idx:idx+3]
        trans_vec = p[idx+3:idx+6]
        
        # Undo Rodrigues Representation for R
        rot_angle = np.linalg.norm(rot_vec)
        skew_matrix = np.array([
            [0, -rot_vec[2], rot_vec[1]],
            [rot_vec[2], 0, -rot_vec[0]],
            [-rot_vec[1], rot_vec[0], 0]
        ])
        identity_matrix = np.identity(3)
        R_matrix = identity_matrix + (np.sin(rot_angle) / rot_angle) * skew_matrix + ((1 - np.cos(rot_angle)) / (rot_angle ** 2)) * (skew_matrix @ skew_matrix)
        
        rotation_matrices.append(R_matrix)
        translation_vectors.append(trans_vec)
    
    return K, rotation_matrices, translation_vectors

# Get the error mean and variance of the projected corners for a specific image
def error(diff, idx):
    diff = diff.reshape((-1 , 2))
    start = idx * 80
    end = start + 80
    diff_norm = np.linalg.norm(diff[start:end], axis =1)
    return np.average(diff_norm), np.var(diff_norm)

# Cost function for the optimization algorithm. Also used to get quantitative evaluation of the refinements done
def cost(p, full_corners, world_coord, radial=False, img_idx=-1, name=None):
    K, Rs, ts = reconstruct_p(p[:-2] if radial else p)

    all_projected_points = []
    for k, (R, t) in enumerate(zip(Rs, ts)):
        # Project points
        H = np.matmul(K, np.column_stack((R[:, 0], R[:, 1], t)))
        pts_h = np.hstack([world_coord, np.ones((len(world_coord), 1))])  # Convert to homogeneous coordinates
        transf_pts = H @ pts_h.T  # Apply homography
        transf_pts = transf_pts.T
        projected_points = transf_pts[:, :2] / transf_pts[:, 2, np.newaxis]  # Normalize by the last row
        
        # Use radial distortion parameters if desired
        if radial:
            x, y = projected_points[:, 0], projected_points[:, 1]
            k1, k2, x0, y0 = p[-2], p[-1], p[2], p[4]
            r_squared = (x - x0)**2 + (y - y0)**2
            x_corrected = x + (x - x0) * (k1 * r_squared + k2 * r_squared**2)
            y_corrected = y + (y - y0) * (k1 * r_squared + k2 * r_squared**2)
            projected_points = np.vstack((x_corrected, y_corrected)).T
            
        all_projected_points.append(projected_points)
        # Draw reprojected corners
        if k == img_idx:
            reproject(all_projected_points, img_idx, name)

    # Calculated the distance between projected corners and ground truth corners
    all_projected_points = np.concatenate(all_projected_points, axis=0)
    full_corners = np.concatenate(full_corners, axis=0)
    diff = full_corners - all_projected_points
    return diff.flatten()
    
# Function to draw the projected images onto an image in whic ground truth corners are already drawn
def reproject(all_projected_points, img_idx, name):
    path = f"/home/aolivepe/Computer-Vision/HW8/output/Pic_{img_idx + 1}_final_intersec.jpg"
    img = cv2.imread(path)
    for point in all_projected_points[img_idx]:
        x, y = int(point[0]), int(point[1])  
        color = (0, 255, 0)
        thickness = 1
        cv2.line(img, (x - 5, y - 5), (x + 5, y + 5), color, thickness)
        cv2.line(img, (x - 5, y + 5), (x + 5, y - 5), color, thickness)
    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/Pic_{img_idx + 1}{name}reproject.jpg', img)

# Function to plot the camera poses for each image of the dataset
def camera_poses(Rs, ts):
    # Calculate the camera centers based on rotations and translations
    camera_centers = [-R.T @ t for R, t in zip(Rs, ts)]

    # Define the axes for each camera
    axis_x = [R.T @ np.array([1, 0, 0]) + center for R, center in zip(Rs, camera_centers)]
    axis_y = [R.T @ np.array([0, 1, 0]) + center for R, center in zip(Rs, camera_centers)]
    axis_z = [R.T @ np.array([0, 0, 1]) + center for R, center in zip(Rs, camera_centers)]

    # Set up the 3D plot
    vector_length = 35
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each camera's x, y, z axes with color-coded quivers
    for center, x, y, z in zip(camera_centers, axis_x, axis_y, axis_z):
        ax.quiver(center[0], center[1], center[2], x[0]-center[0], x[1]-center[1], x[2]-center[2], color="r", length=vector_length, normalize=True)
        ax.quiver(center[0], center[1], center[2], y[0]-center[0], y[1]-center[1], y[2]-center[2], color="g", length=vector_length, normalize=True)
        ax.quiver(center[0], center[1], center[2], z[0]-center[0], z[1]-center[1], z[2]-center[2], color="b", length=vector_length, normalize=True)

    # Plot planes based on camera orientation
    for center, z_axis in zip(camera_centers, axis_z):
        x_vals, y_vals = np.meshgrid(range(int(center[0] - vector_length), int(center[0] + vector_length)), range(int(center[1] - vector_length), int(center[1] + vector_length)))
        z_vals = -((x_vals - center[0]) * z_axis[0] + (y_vals - center[1]) * z_axis[1]) / z_axis[2] + center[2]
        ax.plot_surface(x_vals, y_vals, z_vals, alpha=0.3)

    # Plot calibration pattern as a black square
    center_x, center_y = 20, 60
    size = 50
    x_square = [center_x - size / 2, center_x + size / 2, center_x + size / 2, center_x - size / 2]
    y_square = [center_y - size / 2, center_y - size / 2, center_y + size / 2, center_y + size / 2]
    z_square = [0, 0, 0, 0]
    ax.plot_trisurf(x_square, y_square, z_square, color='black')
    
    ax.set_ylim([-1, 200])
    ax.set_zlim([-300, 0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Set orientation of the plot
    elev = -20
    azim = 85 
    ax.view_init(elev=elev, azim=azim)

    plt.savefig("3d_vectors_plot.jpg", format="jpg", dpi=300)
    
##############################################
#                  MAIN                      #
##############################################

index_img_1 = 4
index_img_2 = 10

# dataset_path = "/home/aolivepe/Computer-Vision/HW8/Dataset2"
dataset_path = "/home/aolivepe/Computer-Vision/HW8/HW8-Files/Dataset1"

# Get world coordinates
x_coords = 10 * np.arange(8)
y_coords = 10 * np.arange(10)
y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
world_coord = np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)

# Get homogrphies and ground truth corners
Hs, full_corners = get_homographies(dataset_path, world_coord)

# Get parameters
w = estimate_w(Hs)
K = estimate_k(w)
print("K: ", K)
Rs, ts = estimate_extrinsic_param(Hs, K)
print("Rs[index_img_1]: ", Rs[index_img_1])
print("ts[index_img_1]: ", ts[index_img_1])
print("Rs[index_img_2]: ", Rs[index_img_2])
print("ts[index_img_2]: ", ts[index_img_2])

# Prepare parameters for refinement
p = param_cam(K, Rs, ts)
# Quantitative metrics for evaluation
mean_init_1, var_init_1 = error(cost(np.array(p), full_corners, world_coord, img_idx=index_img_1, name="_original"), idx=index_img_1)
mean_init_2, var_init_2 = error(cost(np.array(p), full_corners, world_coord, img_idx=index_img_2, name="_original"), idx=index_img_2)

# Refine and project and quantitative metrics of projection after refinement
p_lm = least_squares(cost, np.array(p), method='lm', args=[full_corners, world_coord])
mean_refined_1, var_refined_1 = error(cost(p_lm.x, full_corners, world_coord, img_idx=index_img_1, name="_lm"), idx=index_img_1)
mean_refined_2, var_refined_2 = error(cost(p_lm.x, full_corners, world_coord, img_idx=index_img_2, name="_lm"), idx=index_img_2)

K_refined, Rs_refined, ts_refined = reconstruct_p(p_lm.x)
print("K_refined: ", K_refined)
print("Rs_refined[index_img_1]: ", Rs_refined[index_img_1])
print("ts_refined[index_img_1]: ", ts_refined[index_img_1])
print("Rs_refined[index_img_2]: ", Rs_refined[index_img_2])
print("ts_refined[index_img_2]: ", ts_refined[index_img_2])

# Incorporate radial distortion parameters, refine and quantitative metrics of projection after refinement
p_rad = param_cam(K, Rs, ts)
p_rad.extend([0, 0])
p_lm_rad = least_squares(cost, np.array(p_rad), method='lm', args=[full_corners, world_coord, True])
mean_radial_1, var_radial_1 = error(cost(p_lm_rad.x, full_corners, world_coord, radial=True, img_idx=index_img_1, name="lm_w_rad"), idx=index_img_1)
mean_radial_2, var_radial_2 = error(cost(p_lm_rad.x, full_corners, world_coord, radial=True, img_idx=index_img_2, name="lm_w_rad"), idx=index_img_2)

K_refined_rad, Rs_refined_rad, ts_refined_rad = reconstruct_p(p_lm_rad.x[:-2])
print("K_refined_rad: ", K_refined_rad)
print("Rs_refined_rad[index_img_1]: ", Rs_refined_rad[index_img_1])
print("ts_refined_rad[index_img_1]: ", ts_refined_rad[index_img_1])
print("Rs_refined_rad[index_img_2]: ", Rs_refined_rad[index_img_2])
print("ts_refined_rad[index_img_2]: ", ts_refined_rad[index_img_2])

print(f"--------------Image {index_img_1}----------------")
print(f"|Init mean: {mean_init_1} Init var: {var_init_1}")
print(f"|Refined mean: {mean_refined_1} Refined var: {var_refined_1}")
print(f"|Radial mean: {mean_radial_1} Radial var: {var_radial_1}")
print("[k1 k2] ", p_lm_rad.x[-2:])
print("-------------------------------------")

print(f"--------------Image {index_img_2}----------------")
print(f"|Init mean: {mean_init_2} Init var: {var_init_2}")
print(f"|Refined mean: {mean_refined_2} Refined var: {var_refined_2}")
print(f"|Radial mean: {mean_radial_2} Radial var: {var_radial_2}")
print("[k1 k2] ", p_lm_rad.x[-2:])
print("-------------------------------------")

# Get the camera poses plot
camera_poses(Rs, ts)