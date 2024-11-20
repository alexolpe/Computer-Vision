import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import proj3d

# Function to resize images
def resizeImages(img1, img2, width=450, height=700):
    resized_img1 = cv2.resize(img1, (width, height))
    resized_img2 = cv2.resize(img2, (width, height))
    cv2.imwrite("/home/aolivepe/Computer-Vision/HW9/HW9_images/resized_img1.jpg", resized_img1)
    cv2.imwrite("/home/aolivepe/Computer-Vision/HW9/HW9_images/resized_img2.jpg", resized_img2)
    return resized_img1, resized_img2


# Function to manually collect corresponding points
def getCorrespondingPoints(image1, image2):
    pixel_points1 = []
    pixel_points2 = []
    window_name1 = "Image 1 - Click Points"
    window_name2 = "Image 2 - Click Points"

    def click_event_image1(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel_points1.append((x, y))
            cv2.circle(image1_display, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow(window_name1, image1_display)

    def click_event_image2(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel_points2.append((x, y))
            cv2.circle(image2_display, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow(window_name2, image2_display)

    image1_display = image1.copy()
    image2_display = image2.copy()
    cv2.imshow(window_name1, image1_display)
    cv2.imshow(window_name2, image2_display)
    cv2.setMouseCallback(window_name1, click_event_image1)
    cv2.setMouseCallback(window_name2, click_event_image2)

    print("Click on corresponding points in both images. Press 'q' to finish.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    return np.array(pixel_points1), np.array(pixel_points2)


# Function to estimate the fundamental matrix
def estimateF(points1, points2):
    def normalizePts(points):
        mean = np.mean(points, axis=0)
        d_bar = np.mean([np.linalg.norm(point - mean) for point in points])
        scale = np.sqrt(2) / d_bar
        T = np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]])
        normalized_points = np.array([T @ np.append(point, 1) for point in points])
        normalized_points /= normalized_points[:, 2][:, None]
        return normalized_points[:, :2], T

    def constructA(points1, points2):
        return np.array([
            [p2[0] * p1[0], p2[0] * p1[1], p2[0], p2[1] * p1[0], p2[1] * p1[1], p2[1], p1[0], p1[1], 1]
            for p1, p2 in zip(points1, points2)
        ])

    norm_points1, T1 = normalizePts(points1)
    norm_points2, T2 = normalizePts(points2)
    A = constructA(norm_points1, norm_points2)
    _, _, Vt = np.linalg.svd(A)
    F_hat = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, D, Vt = np.linalg.svd(F_hat)
    D[-1] = 0
    F_hat = U @ np.diag(D) @ Vt

    # Denormalize F
    F = T2.T @ F_hat @ T1
    return F / F[-1, -1]

def refine_H_prime(H_prime, points_img1, points_img2, H, max_iterations=100, tol=1e-6):
    def objective(H_prime_flat):
        H_prime = H_prime_flat.reshape(3, 3)
        transformed_points_img1 = np.dot(H, np.column_stack((points_img1, np.ones(len(points_img1)))).T).T
        transformed_points_img2 = np.dot(H_prime, np.column_stack((points_img2, np.ones(len(points_img2)))).T).T
        
        transformed_points_img1 /= transformed_points_img1[:, 2][:, np.newaxis]
        transformed_points_img2 /= transformed_points_img2[:, 2][:, np.newaxis]
        
        errors = transformed_points_img1[:, 1] - transformed_points_img2[:, 1]
        return np.sum(errors**2)

    result = least_squares(objective, H_prime.ravel(), method='trf', max_nfev=max_iterations, ftol=tol)
    return result.x.reshape(3, 3)


# Function to find epipoles
def findEs(F):
    U, D, Vt = np.linalg.svd(F)
    e = Vt[-1]
    e = e / e[-1]
    e_prime = U[:, -1]
    e_prime = e_prime / e_prime[-1]
    return e, e_prime


# Function to compute rectifying homographies
def compute_rectifying_homographies(img, e):
    h, w = img.shape[:2]
    T1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
    e = e / e[-1]
    theta = np.arctan2(-e[1], e[0])
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    f = np.sqrt(e[0]**2 + e[1]**2)
    G = np.array([[1, 0, 0], [0, 1, 0], [-1 / f, 0, 1]])
    return np.linalg.inv(T1) @ G @ R @ T1


# Function to warp and plot rectified images
def apply_rectification(image1, image2, H, H_prime, width=800, height=600):
    rectified1 = cv2.warpPerspective(image1, H, (width, height))
    rectified2 = cv2.warpPerspective(image2, H_prime, (width, height))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(rectified1, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Rectified Left Image")
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(rectified2, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Rectified Right Image")
    ax[1].axis("off")
    plt.show()
    return rectified1, rectified2

def plot_rectified_images_with_correspondences_fixed(img1, img2, points_img1, points_img2, H, H_dash, width=3000, height=5000):
    # Warp the images using the provided homographies
    rectified1 = cv2.warpPerspective(img1, H, (width, height))
    rectified2 = cv2.warpPerspective(img2, H_dash, (width, height))
    
    # Transform the points using the homographies
    def transform_points(points, H):
        transformed = []
        for p in points:
            p_h = np.append(p, 1)
            p_transformed = H @ p_h
            p_transformed /= p_transformed[2]
            transformed.append(p_transformed[:2])
        return np.array(transformed)

    rectified_points_img1 = transform_points(points_img1, H)
    rectified_points_img2 = transform_points(points_img2, H_dash)

    # Create a combined canvas
    combined_image = np.zeros((height, 2 * width, 3), dtype=np.uint8)
    combined_image[:, :width] = rectified1
    combined_image[:, width:] = rectified2

    # Shift the right points to match their positions on the combined canvas
    shifted_points_img2 = rectified_points_img2.copy()
    shifted_points_img2[:, 0] += width

    # Plot the correspondences on the combined image
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    for p1, p2 in zip(rectified_points_img1, shifted_points_img2):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=np.random.rand(3,), linewidth=1.5)
        plt.scatter(p1[0], p1[1], color='red', s=15)
        plt.scatter(p2[0], p2[1], color='blue', s=15)

    plt.title("Keypoint Correspondences After Rectification")
    plt.axis("off")
    
    cv2.imwrite("rectified_img1_img2_task1.jpg", rectified_img2)
    
    plt.show()


def extract_interest_points(image, low_threshold=100, high_threshold=350):
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    # Get the coordinates of the edge points
    y_coords, x_coords = np.where(edges > 0)
    interest_points = np.stack((x_coords, y_coords), axis=-1)

    return interest_points, edges


def compute_canvas_size(image, H):
    """
    Computes the required canvas size for warpPerspective to ensure the entire image fits.
    """
    h, w = image.shape[:2]
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1]
    ])  # Four corners of the original image

    # Transform corners using H
    transformed_corners = H @ corners.T
    transformed_corners /= transformed_corners[2]  # Normalize to homogeneous coordinates
    transformed_corners = transformed_corners[:2].T  # Extract (x, y) coordinates

    # Calculate the bounding box
    x_min, y_min = np.min(transformed_corners, axis=0)
    x_max, y_max = np.max(transformed_corners, axis=0)

    # Compute the required canvas size
    width = int(np.ceil(x_max - x_min))
    height = int(np.ceil(y_max - y_min))
    return width, height, x_min, y_min

def apply_rectification_dynamic(image1, image2, H, H_prime):
    # Compute canvas size for both images
    width1, height1, x_min1, y_min1 = compute_canvas_size(image1, H)
    width2, height2, x_min2, y_min2 = compute_canvas_size(image2, H_prime)

    # Compute the canvas size as the maximum width and height
    width = max(width1, width2)
    height = max(height1, height2)

    # Translation matrices to shift the images to the positive coordinate space
    T1 = np.array([[1, 0, -x_min1],
                   [0, 1, -y_min1],
                   [0, 0, 1]])
    T2 = np.array([[1, 0, -x_min2],
                   [0, 1, -y_min2],
                   [0, 0, 1]])

    # Apply rectification
    rectified1 = cv2.warpPerspective(image1, T1 @ H, (width, height))
    rectified2 = cv2.warpPerspective(image2, T2 @ H_prime, (width, height))

    return rectified1, rectified2

def get_matrix_cross_product_representation(e):
    return np.array([[0, -e[2], e[1]],
                     [e[2], 0, -e[0]],
                     [-e[1], e[0], 0]])

def compute_Ps(F, e, e_prime):
    P = np.hstack([np.eye(3), np.zeros((3, 1))])
    e_prime = e_prime.reshape(-1, 1) if e_prime.ndim == 1 else e_prime
    P_prime = np.hstack([
        np.matmul(get_matrix_cross_product_representation(e_prime.ravel()), F),
        e_prime
    ])
    return P, P_prime

def apply_homography(pts, H):
    """Apply a homography to a set of points."""
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))  # Convert to homogeneous
    pts_rectified_h = H @ pts_h.T  # Apply homography
    pts_rectified = (pts_rectified_h[:2] / pts_rectified_h[2]).T  # Normalize
    return pts_rectified

def triangulate_point(P1, P2, x1, x2):
    # Ensure homogeneous coordinates
    if len(x1) == 2:
        x1 = np.append(x1, 1)
    if len(x2) == 2:
        x2 = np.append(x2, 1)
    
    # Construct the matrix A
    A = np.vstack([
        x1[0] * P1[2, :] - P1[0, :],
        x1[1] * P1[2, :] - P1[1, :],
        x2[0] * P2[2, :] - P2[0, :],
        x2[1] * P2[2, :] - P2[1, :]
    ])
    
    # Solve AX = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    # Convert from homogeneous to Cartesian coordinates
    X = X / X[3]
    return X[:3]

def threeD_plot(points_img1, points_img2, P, P_prime, H, H_prime):
    # homogeneous_points_1 = [(x, y, 1) for x, y in points_img1]
    # homogeneous_points_2 = [(x, y, 1) for x, y in points_img2]
    # world_points = cv2.triangulatePoints(P, P_prime, np.array(points_img1).T, np.array(points_img2).T)

    X = [triangulate_point(P, P_prime, x1, x2) for x1, x2 in zip(points_img1, points_img2)]
    
    # Points to plot in 2D (left), 3D, and 2D (right)
    points2D_left = points_img1
    points3D = X
    points2D_right = points_img2
    
    points2D_left = apply_homography(np.array(points2D_left), H)
    points2D_right = apply_homography(np.array(points2D_right), H_prime)

    print("points2D_left: ", len(points2D_left), points2D_left[0])
    print("points3D: ", len(points3D), points3D[0])
    print("points2D_right: ", len(points2D_right), points2D_right[0])
    
    # Create figure and axes
    fig = plt.figure(figsize=(10, 10), dpi=100)

    # Overlay axes for annotations
    ax0 = plt.axes([0., 0., 1., 1.])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis('off')

    # Subplots: 3D in the first row (center), 2D plots in the second row
    ax2 = fig.add_subplot(2, 1, 1, projection='3d')  # Top center 3D plot
    ax1 = fig.add_subplot(2, 2, 3)  # Bottom left 2D plot (image)
    ax3 = fig.add_subplot(2, 2, 4)  # Bottom right 2D plot (image)

    # Create dummy images for 2D plots
    image_left = cv2.imread("/home/aolivepe/Computer-Vision/HW9/rectified_img1_task1.jpg")  # Random grayscale image
    image_right = cv2.imread("/home/aolivepe/Computer-Vision/HW9/rectified_img2_task1.jpg")  # Random grayscale image

    # Display images
    h_left, w_left = image_left.shape[:2]
    h_right, w_right = image_right.shape[:2]
    ax1.imshow(image_left, cmap='gray', extent=(0, w_left, h_left, 0))
    ax3.imshow(image_right, cmap='gray', extent=(0, w_right, h_right, 0))

    # Plot points and lines in the left 2D plot (on top of image)
    for i, p in enumerate(points2D_left):
        ax1.plot(p[0], p[1], 'go')  # Points
        if i > 0:  # Connect points with a line
            ax1.plot([points2D_left[i - 1][0], p[0]], [points2D_left[i - 1][1], p[1]], 'g--')

    # Plot points and lines in the 3D plot
    for i, p in enumerate(points3D):
        ax2.plot([p[0]], [p[1]], [p[2]], 'ro')  # Points
        if i > 0:  # Connect points with a line
            ax2.plot([points3D[i - 1][0], p[0]], [points3D[i - 1][1], p[1]], [points3D[i - 1][2], p[2]], 'r--')

    # Plot points and lines in the right 2D plot (on top of image)
    for i, p in enumerate(points2D_right):
        ax3.plot(p[0], p[1], 'bo')  # Points
        if i > 0:  # Connect points with a line
            ax3.plot([points2D_right[i - 1][0], p[0]], [points2D_right[i - 1][1], p[1]], 'b--')

    # Set axes limits
    margin=0.2
    ax2.set_xlim(np.min(np.array(points3D)[:, 0]) - margin, np.max(np.array(points3D)[:, 0]) + margin)
    ax2.set_ylim(np.min(np.array(points3D)[:, 1]) - margin, np.max(np.array(points3D)[:, 1]) + margin)
    ax2.set_zlim(np.min(np.array(points3D)[:, 2]) - margin, np.max(np.array(points3D)[:, 2]) + margin)
    ax1.set_xlim(0, w_left)
    ax1.set_ylim(h_left, 0)  # Flip the y-axis to match image orientation
    ax3.set_xlim(0, w_right)
    ax3.set_ylim(h_right, 0)  # Flip the y-axis

    # Connect corresponding points between plots
    fig.canvas.draw()
    dpi = fig.get_dpi()
    height = fig.get_figheight() * dpi
    width = fig.get_figwidth() * dpi

    for p2D, p3D, p2D_right in zip(points2D_left, points3D, points2D_right):
        # Project left 2D points into figure coordinates
        x1, y1 = ax1.transData.transform(p2D)
        x1 = x1 / width
        y1 = y1 / height

        # Project 3D points into figure coordinates
        x2, y2, _ = proj3d.proj_transform(p3D[0], p3D[1], p3D[2], ax2.get_proj())
        [x2, y2] = ax2.transData.transform((x2, y2))
        x2 = x2 / width
        y2 = y2 / height

        # Project right 2D points into figure coordinates
        x3, y3 = ax3.transData.transform(p2D_right)
        x3 = x3 / width
        y3 = y3 / height

        # Transform coordinates for lines
        transFigure = fig.transFigure.inverted()
        coord1 = transFigure.transform(ax0.transData.transform([x1, y1]))
        coord2 = transFigure.transform(ax0.transData.transform([x2, y2]))
        coord3 = transFigure.transform(ax0.transData.transform([x3, y3]))

        # Add lines connecting the points
        line1 = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]), transform=fig.transFigure, linestyle='dashed', color='blue')
        line2 = matplotlib.lines.Line2D((coord2[0], coord3[0]), (coord2[1], coord3[1]), transform=fig.transFigure, linestyle='dashed', color='purple')
        fig.lines.extend([line1, line2])

    # Add annotation
    ax0.text(0.5, 0.95, "3D Plot Connecting Two 2D Images with Points", fontsize=20, ha='center')

    # Save the figure
    plt.savefig("3dplot.jpg")
    
    # plt.figure(figsize=(15, 8))

    # h1, w1, _ = image_left.shape
    # h2, w2, _ = image_right.shape

    # # Combine images
    # combined_image = np.hstack((image_left, image_right))

    # # Adjust the points from the second image
    # pts2_shifted = points2D_right.copy()
    # pts2_shifted[:, 0] += w1  # Shift x-coordinates by the width of the first image

    # # Plot the combined image
    # plt.figure(figsize=(10, 5))
    # plt.imshow(combined_image)
    # plt.axis("off")

    # # Draw correspondences
    # for (x1, y1), (x2, y2) in zip(points2D_left, pts2_shifted):
    #     plt.plot([x1, x2], [y1, y2], 'r-', lw=1)  # Line connecting points
    #     plt.scatter([x1, x2], [y1, y2], c='yellow', s=30)  # Points

    # cv2.imwrite("rectified_img1_img2_task1.jpg", rectified_img2)
    
# Main driver script
img1_path = "/home/aolivepe/Computer-Vision/HW9/HW9_images/img1.jpg"
img2_path = "/home/aolivepe/Computer-Vision/HW9/HW9_images/img2.jpg"
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
resized_img1, resized_img2 = resizeImages(img1, img2)
'''points_img1, points_img2 = getCorrespondingPoints(resized_img1, resized_img2)
np.save("points_img1.npy", points_img1)
np.save("points_img2.npy", points_img2)'''

# Original points
# points_img1= [(504, 729), (511, 410), (821, 415), (942, 737), (498, 789), (903, 811), (683, 457), (692, 589)]
# points_img2= [(230, 747), (291, 422), (603, 411), (675, 727), (243, 810), (657, 803), (463, 459), (450, 590)]

#Resized points
points_img1= [(239, 225), (384, 228), (441, 405), (236, 399), (232, 431), (422, 443), (322, 250), (324, 321)]
points_img2= [(137, 230), (283, 226), (316, 400), (107, 408), (113, 443), (309, 439), (217, 252), (210, 322)]

F = estimateF(points_img1, points_img2)
e, e_prime = findEs(F)
#Get P and P'
P, P_prime = compute_Ps(F, e, e_prime)

H = compute_rectifying_homographies(resized_img1, e)
H_prime = compute_rectifying_homographies(resized_img2, e_prime)
H_prime = refine_H_prime(H_prime, points_img1, points_img2, H)
rectified_img1, rectified_img2 = apply_rectification(resized_img1, resized_img2, H, H_prime)
# plot_rectified_images_with_correspondences_fixed(resized_img1, resized_img2, points_img1, points_img2, H, H_prime, width=1000, height=1000)

rectified_img1, rectified_img2 = apply_rectification_dynamic(resized_img1, resized_img2, H, H_prime)
cv2.imwrite("rectified_img1_task1.jpg", rectified_img1)
cv2.imwrite("rectified_img2_task1.jpg", rectified_img2)

threeD_plot(points_img1, points_img2, P, P_prime, H, H_prime)