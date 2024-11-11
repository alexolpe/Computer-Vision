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
        cv2.line(img, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)
        
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
            color = (0, 0, 255)  # Green color in BGR
            thickness = 1  # Thickness of the line
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
    mat_A = []
    for i in range(len(r_pts)):
        mat_A.append([0, 0, 0, -d_pts[i][0], -d_pts[i][1], -1, r_pts[i][1] * d_pts[i][0], r_pts[i][1] * d_pts[i][1], r_pts[i][1]])
        mat_A.append([d_pts[i][0], d_pts[i][1], 1, 0, 0, 0, -r_pts[i][0] * d_pts[i][0], -r_pts[i][0] * d_pts[i][1], -r_pts[i][0]])
    mat_A = np.array(mat_A)
    _, _, v = np.linalg.svd(mat_A.T @ mat_A)
    return np.reshape(v[-1], (3, 3))

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
    _, _, v = np.linalg.svd(lhs)
    w_solution = v[-1, :]  # Last row of V matrix provides the solution
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
    K = np.array([[alphax, s, x0],
                  [0, alphay, y0],
                  [0, 0, 1]])
    return K

def get_extrinsic_param(homographies, K):
    rot = []
    trans = []
    K_inv = np.linalg.inv(K)

    for H in homographies:
        # Extract columns from the homography and scale
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        r1 = K_inv @ h1 / np.linalg.norm(K_inv @ h1)
        r2 = K_inv @ h2 / np.linalg.norm(K_inv @ h1)
        r3 = np.cross(r1, r2)
        t = K_inv @ h3 / np.linalg.norm(K_inv @ h1)
        R = np.stack([r1,r2,r3], axis=1)

        # Enforce orthogonality constraint on R using SVD
        u, _, v = np.linalg.svd(R)
        R = u @ v
        
        rot.append(R)
        trans.append(t)

    return rot, trans

def param_cam(K, rots, trans):
    # Camera intrinsics
    p = [K[0, 0], K[0, 1], K[0, 2], K[1, 1], K[1, 2]]
    # Append extrinsics for each rotation and translation pair
    for R, t in zip(rots, trans):
        p.extend(np.hstack(((np.arccos((np.trace(R) - 1) / 2) / (2 * np.sin(np.arccos((np.trace(R) - 1) / 2)))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]), t)))
    return p

def reconstruct_params(p):
    K = np.array([[p[0], p[1], p[2]],
                  [0, p[3], p[4]],
                  [0, 0, 1]])
    # Extrinsics
    rotation_matrices, translation_vectors = [], []
    step_size = 6

    for idx in range(5, len(p), step_size):
        # Extract rotation and translation components
        rot_vec = p[idx:idx+3]
        trans_vec = p[idx+3:idx+6]
        
        # Calculate rotation angle from rotation vector
        rot_angle = np.linalg.norm(rot_vec)
        skew_matrix = np.array([
            [0, -rot_vec[2], rot_vec[1]],
            [rot_vec[2], 0, -rot_vec[0]],
            [-rot_vec[1], rot_vec[0], 0]
        ])
        
        # Construct rotation matrix using Rodrigues' formula
        identity_matrix = np.identity(3)
        R_matrix = identity_matrix + (np.sin(rot_angle) / rot_angle) * skew_matrix + ((1 - np.cos(rot_angle)) / (rot_angle ** 2)) * (skew_matrix @ skew_matrix)
        
        rotation_matrices.append(R_matrix)
        translation_vectors.append(trans_vec)
    
    return K, rotation_matrices, translation_vectors

def error(diff, idx):
    diff = diff.reshape((-1 , 2))
    start = idx * 80
    end = start + 80
    diff_norm = np.linalg.norm(diff[start:end], axis =1)
    return np.average(diff_norm), np.var(diff_norm)

def cost_function(params, all_intersec_points, world_coord, radio_distortion=False, img_idx=-1, name=None):
    K, Rs, ts = reconstruct_params(params[:-2] if radio_distortion else params)

    all_projected_points = []
    for k, (R, t) in enumerate(zip(Rs, ts)):
        H = np.matmul(K, np.column_stack((R[:, 0], R[:, 1], t)))
        
        # Apply homography
        pts_h = np.hstack([world_coord, np.ones((len(world_coord), 1))])  # Convert to homogeneous coordinates
        transf_pts = H @ pts_h.T  # Apply homography
        transf_pts = transf_pts.T  # Transpose back to Nx3
        projected_points = transf_pts[:, :2] / transf_pts[:, 2, np.newaxis]  # Normalize by the last row
        
        if radio_distortion:
            x, y = projected_points[:, 0], projected_points[:, 1]
            k1, k2, x0, y0 = params[-2], params[-1], params[2], params[4]
            r_squared = (x - x0)**2 + (y - y0)**2
            x_corrected = x + (x - x0) * (k1 * r_squared + k2 * r_squared**2)
            y_corrected = y + (y - y0) * (k1 * r_squared + k2 * r_squared**2)
            projected_points = np.vstack((x_corrected, y_corrected)).T
        all_projected_points.append(projected_points)
        if k == img_idx:
            reproject(all_projected_points, img_idx, name)

    all_projected_points = np.concatenate(all_projected_points, axis=0)
    all_intersec_points = np.concatenate(all_intersec_points, axis=0)
    diff = all_intersec_points - all_projected_points
    return diff.flatten()
    
    # else:
    #     diff = all_intersec_points[img_idx] - all_projected_points[img_idx]
    #     dx = diff[:, 0]
    #     dy = diff[:, 1]
    #     distance = np.sqrt(dx**2 + dy**2)
    #     mean = np.mean(distance)
    #     var = np.var(distance)
    #     reproject(all_projected_points, img_idx, name)
    #     return mean, var
    
def reproject(all_projected_points, img_idx, name):
    path = f"/home/aolivepe/Computer-Vision/HW8/output/Pic_{img_idx + 1}_final_intersec.jpg"
    img = cv2.imread(path)
    for point in all_projected_points[img_idx]:
        x, y = int(point[0]), int(point[1])  # Change these values to the pixel location you want
        # Draw the "X" by drawing two diagonal lines that cross at (x, y)
        color = (0, 255, 0)  # Green color in BGR
        thickness = 1  # Thickness of the line
        cv2.line(img, (x - 5, y - 5), (x + 5, y + 5), color, thickness)
        cv2.line(img, (x - 5, y + 5), (x + 5, y - 5), color, thickness)
        # cv2.circle(img, (int(point[0]), int(point[1])), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(f'/home/aolivepe/Computer-Vision/HW8/output/Pic_{img_idx + 1}{name}reproject.jpg', img)

##############################################
#                  MAIN                      #
##############################################

index_img_1 = 4
index_img_2 = 10

dataset_path = "/home/aolivepe/Computer-Vision/HW8/Dataset2"
# dataset_path = "/home/aolivepe/Computer-Vision/HW8/HW8-Files/Dataset1"

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
print("Rs[index_img_1]: ", Rs[index_img_1])
print("ts[index_img_1]: ", ts[index_img_1])
print("Rs[index_img_2]: ", Rs[index_img_2])
print("ts[index_img_2]: ", ts[index_img_2])

# Define camera parameters
params = param_cam(K, Rs, ts)
mean_init_0, var_init_0 = error(cost_function(np.array(params), all_intersec_points, world_coord, img_idx=index_img_1, name='init'), idx=index_img_1)
mean_init_7, var_init_7 = error(cost_function(np.array(params), all_intersec_points, world_coord, img_idx=index_img_2, name='init'), idx=index_img_2)

# Perform least squares optimization

# Refine parameters
# K_refined, Rs_refined, ts_refined = reconstruct_params(res_ls.x)
# print(K_refined)
# print(Rs_refined[index_img_1], ts_refined[index_img_1])
# print(Rs_refined[index_img_2], ts_refined[index_img_2])

# Project and print mean and variance
res_ls = least_squares(cost_function, np.array(params), method='lm', args=[all_intersec_points, world_coord])
mean_refined_0, var_refined_0 = error(cost_function(res_ls.x, all_intersec_points, world_coord, img_idx=index_img_1, name='refined'), idx=index_img_1)
mean_refined_7, var_refined_7 = error(cost_function(res_ls.x, all_intersec_points, world_coord, img_idx=index_img_2, name='refined'), idx=index_img_2)

K_refined, Rs_refined, ts_refined = reconstruct_params(res_ls.x)
print("K_refined: ", K_refined)
print("Rs_refined[index_img_1]: ", Rs_refined[index_img_1])
print("ts_refined[index_img_1]: ", ts_refined[index_img_1])
print("Rs_refined[index_img_2]: ", Rs_refined[index_img_2])
print("ts_refined[index_img_2]: ", ts_refined[index_img_2])

# Define camera parameters with radial distortion
params_rd = param_cam(K, Rs, ts)
params_rd.extend([0, 0])
# Perform least squares optimization with radial distortion
res_ls_rd = least_squares(cost_function, np.array(params_rd), method='lm', args=[all_intersec_points, world_coord, True])
mean_radial_0, var_radial_0 = error(cost_function(res_ls_rd.x, all_intersec_points, world_coord, radio_distortion=True, img_idx=index_img_1, name='radio_dis'), idx=index_img_1)
mean_radial_7, var_radial_7 = error(cost_function(res_ls_rd.x, all_intersec_points, world_coord, radio_distortion=True, img_idx=index_img_2, name='radio_dis'), idx=index_img_2)

K_refined_rad, Rs_refined_rad, ts_refined_rad = reconstruct_params(res_ls_rd.x[:-2])
print("K_refined_rad: ", K_refined_rad)
print("Rs_refined_rad[index_img_1]: ", Rs_refined_rad[index_img_1])
print("ts_refined_rad[index_img_1]: ", ts_refined_rad[index_img_1])
print("Rs_refined_rad[index_img_2]: ", Rs_refined_rad[index_img_2])
print("ts_refined_rad[index_img_2]: ", ts_refined_rad[index_img_2])

print(f"--------------Image {index_img_1}----------------")
print(f"|Init mean: {mean_init_0} Init var: {var_init_0}")
print(f"|Refined mean: {mean_refined_0} Refined var: {var_refined_0}")
print(f"|Radial mean: {mean_radial_0} Radial var: {var_radial_0}")
print("[k1 k2] ", res_ls_rd.x[-2:])
print("-------------------------------------")

print(f"--------------Image {index_img_2}----------------")
print(f"|Init mean: {mean_init_7} Init var: {var_init_7}")
print(f"|Refined mean: {mean_refined_7} Refined var: {var_refined_7}")
print(f"|Radial mean: {mean_radial_7} Radial var: {var_radial_7}")
print("[k1 k2] ", res_ls_rd.x[-2:])
print("-------------------------------------")

def camera_poses(Rs, ts):
    Cs=[-np.matmul(np.transpose(Rs[i]),ts[i]) for i in range(len(Rs))]
    Xx=[np.matmul(np.transpose(Rs[i]),[Cs[i][0] + 1,0,0])+Cs[i] for i in range(len(Rs))]
    Xy=[np.matmul(np.transpose(Rs[i]),[0,Cs[i][1] + 1,0])+Cs[i] for i in range(len(Rs))]
    Xz=[np.matmul(np.transpose(Rs[i]),[0,0,Cs[i][2] + 1])+Cs[i] for i in range(len(Rs))]
    ### length of vectors
    length=35
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(Cs)):
        ax.quiver(Cs[i][0],Cs[i][1],Cs[i][2],
                  Xx[i][0],Xx[i][1],Xx[i][2],
                  color="r",length=length,normalize=True)
        if i==0:
            print(Xx[i])
    for i in range(len(Cs)):
        ax.quiver(Cs[i][0],Cs[i][1],Cs[i][2],
                  Xy[i][0],Xy[i][1],Xy[i][2],
                  color="g",length=length,normalize=True)
        if i==0:
            print(Xy[i])
    for i in range(len(Cs)):
        ax.quiver(Cs[i][0],Cs[i][1],Cs[i][2],
                  Xz[i][0],Xz[i][1],Xz[i][2],
                  color="b",length=length,normalize=True)
        if i==0:
            print(Xz[i])
    for i in range(len(Cs)):
        xx, yy = np.meshgrid(range(int(Cs[i][0]-length),int(Cs[i][0]+length)),
                             range(int(Cs[i][1]-length),int(Cs[i][1]+length)))
        z = -((xx-Cs[i][0])*Xz[i][0]+(yy-Cs[i][1])*Xz[i][1])/Xz[i][2]+Cs[i][2]
        ax.plot_surface(xx, yy, z, alpha=0.3)
        
    # Create calibration pattern on the Z=0 plane
    grid_size = 10  # Size of each square in the grid
    num_squares_h = 5  # Number of squares along each axis
    num_squares_v = 4
    gap = 5  # Space between squares

    for i in range(-num_squares_v // 2, num_squares_v // 2):
        for j in range(-num_squares_h // 2, num_squares_h // 2):
            x = [i * (grid_size + gap), (i + 1) * (grid_size + gap) - gap, (i + 1) * (grid_size + gap) - gap, i * (grid_size + gap)]
            y = [j * (grid_size + gap), j * (grid_size + gap), (j + 1) * (grid_size + gap) - gap, (j + 1) * (grid_size + gap) - gap]
            z = [0, 0, 0, 0]
            ax.plot_trisurf(x, y, z, color='black', shade=False)
    
    ## show origin with limit setting
    ax.set_ylim([-1, 200])
    ax.set_zlim([-300, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("3d_vectors_plot.jpg", format="jpg", dpi=300)
    
camera_poses(Rs, ts)