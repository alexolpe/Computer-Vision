import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import matplotlib
import numpy as np
import cv2

# Function to update the figure with new correspondences
def update_lines():
    # Clear previous lines
    fig.lines = []
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
        line1 = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                        transform=fig.transFigure, linestyle='dashed', color='blue')
        line2 = matplotlib.lines.Line2D((coord2[0], coord3[0]), (coord2[1], coord3[1]),
                                        transform=fig.transFigure, linestyle='dashed', color='purple')
        fig.lines.extend([line1, line2])

    # Redraw the figure
    fig.canvas.draw()

# Callback function for mouse motion events
def on_mouse_move(event):
    if event.button == 1:  # Left mouse button for rotating
        ax2.view_init(elev=ax2.elev + event.step, azim=ax2.azim + event.step)
        update_lines()

# Create figure and axes
fig = plt.figure(figsize=(10, 10), dpi=100)
ax0 = plt.axes([0., 0., 1., 1.])
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.axis('off')

# Subplots: 3D in the first row (center), 2D plots in the second row
ax2 = fig.add_subplot(2, 1, 1, projection='3d')  # Top center 3D plot
ax1 = fig.add_subplot(2, 2, 3)  # Bottom left 2D plot (image)
ax3 = fig.add_subplot(2, 2, 4)  # Bottom right 2D plot (image)

# Load images and display them
image_left = cv2.imread("/home/aolivepe/Computer-Vision/HW9/rectified_img1_task1.jpg")
image_right = cv2.imread("/home/aolivepe/Computer-Vision/HW9/rectified_img2_task1.jpg")
h_left, w_left = image_left.shape[:2]
h_right, w_right = image_right.shape[:2]
ax1.imshow(image_left, cmap='gray', extent=(0, w_left, h_left, 0))
ax3.imshow(image_right, cmap='gray', extent=(0, w_right, h_right, 0))

# Points to plot in 2D (left), 3D, and 2D (right)
points2D_left = [[250, 120], [100, 100], [750, 121]]
points3D = [[0, 200, 1], [1, 1, 0.6], [2, 1, 0.3]]
points2D_right = [[200, 50], [600, 300], [100, 80]]

# Plot points and lines in the plots
for i, p in enumerate(points2D_left):
    ax1.plot(p[0], p[1], 'go')
    if i > 0:
        ax1.plot([points2D_left[i - 1][0], p[0]], [points2D_left[i - 1][1], p[1]], 'g--')

for i, p in enumerate(points3D):
    ax2.plot([p[0]], [p[1]], [p[2]], 'ro')
    if i > 0:
        ax2.plot([points3D[i - 1][0], p[0]], [points3D[i - 1][1], p[1]], [points3D[i - 1][2], p[2]], 'r--')

for i, p in enumerate(points2D_right):
    ax3.plot(p[0], p[1], 'bo')
    if i > 0:
        ax3.plot([points2D_right[i - 1][0], p[0]], [points2D_right[i - 1][1], p[1]], 'b--')

# Set limits
margin = 0.2
ax2.set_xlim(np.min(np.array(points3D)[:, 0]) - margin, np.max(np.array(points3D)[:, 0]) + margin)
ax2.set_ylim(np.min(np.array(points3D)[:, 1]) - margin, np.max(np.array(points3D)[:, 1]) + margin)
ax2.set_zlim(np.min(np.array(points3D)[:, 2]) - margin, np.max(np.array(points3D)[:, 2]) + margin)
ax1.set_xlim(0, w_left)
ax1.set_ylim(h_left, 0)
ax3.set_xlim(0, w_right)
ax3.set_ylim(h_right, 0)

# Initialize figure dimensions
fig.canvas.draw()
dpi = fig.get_dpi()
height = fig.get_figheight() * dpi
width = fig.get_figwidth() * dpi

# Add annotation
ax0.text(0.5, 0.95, "3D Plot Connecting Two 2D Images with Points", fontsize=20, ha='center')

# Connect mouse motion event to rotation
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Draw initial lines
update_lines()

# Show the plot
plt.show()

# Save the figure
plt.savefig("prova.jpg", dpi=100)
