import os
import cv2
import time
import numpy as np
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imsave, imread
from scipy.optimize import least_squares
from copy import deepcopy

np.set_printoptions(suppress=True)

class twoD_HC_representation():
	def __init__(self, point, mode = 'pixel'):
		self.mode = mode
		self.HC_representation = None
		self.pixel_representation = None
		if mode=='pixel':
			self.pixel_representation = point
			self.convert_points_to_HC(point)
		else:
			self.HC_representation = point
			self.convert_HC_to_points(point)

	def convert_points_to_HC(self, points):
		"""
		Input:
			1. points: A list of points [x,y]
		Output:
			1. out: a matrix of shape (3xn) in HC representation.
		"""
		out = []
		for i in range(len(points)):
			out.append(np.array(points[i]+[1.0]).reshape(3,1))
		self.HC_representation = np.hstack(out)

	def convert_HC_to_points(self, np_array):
		"""
		Input: 
			1. A matrix of shape (3xn) in HC representation.
		Output:
			1. points: A list of points [x,y]
		"""
		points = []
		for i in range(np.shape(np_array)[1]):
			if np_array[:,i][2] != 0:
				points.append([int(np_array[:,i][0]/np_array[:,i][2]), int(np_array[:,i][1]/np_array[:,i][2])])
			else:
				points.append([np.inf, np.inf])
		self.pixel_representation = points

	def normalize_HC_representation(self):
		"""
		Input: 
			1. Reads the HC representation and normalizes it, if the last coordinate is zero then it keeps it unchanged
		Output:
			1. points: A list of points [x,y]
		"""
		for i in range(np.shape(self.HC_representation)[1]):
			if self.HC_representation[:,i][2] != 0:
				self.HC_representation[:,i] = np.array([self.HC_representation[:,i][0]/self.HC_representation[:,i][2], self.HC_representation[:,i][1]/self.HC_representation[:,i][2], 1])
			else:
				self.HC_representation[:,i] = np.array([self.HC_representation[:,i][0], self.HC_representation[:,i][1],0])
    



def apply_homography_v1(img1_path, H):
	img1 = io.imread(img1_path, pilmode='RGB')

	height, width, _ = img1.shape
	boundaries = [[0,0],[width,height],[width,0],[0,height]]
	print(boundaries)
	HC = twoD_HC_representation(boundaries)
	HC.HC_representation = np.matmul(H,HC.HC_representation)
	HC.normalize_HC_representation()
	print(HC.HC_representation)
	maximum = np.max(HC.HC_representation,1)
	minimum = np.min(HC.HC_representation,1)
	img2 = np.ones((int(np.ceil(maximum[1])),int(np.ceil(maximum[0])),3), dtype= np.uint8)*255
	print(img2.shape)

	for width in tqdm(range(img1.shape[1]-1)):
		for height in range(img1.shape[0]-1):
			HC = twoD_HC_representation([[width,height]])
			HC.HC_representation = np.matmul(H,HC.HC_representation)
			HC.normalize_HC_representation()
			x,y = [HC.HC_representation[0][0],HC.HC_representation[1][0]]

			try:
				# if round(y) <1280 and round(x) <1280 and round(x) >0 and round(y) >0:
				img2[round(y),round(x),:] = img1[height,width,:]
				img2[round(y)-1,round(x)-1,:] = img1[height,width,:]
				img2[round(y)-1,round(x)+1,:] = img1[height,width,:]
				img2[round(y)+1,round(x)-1,:] = img1[height,width,:]
				img2[round(y)+1,round(x)+1,:] = img1[height,width,:]
			except Exception as e:
				continue

	return img2

def apply_homography_v2(img1_path, H):
	img1 = io.imread(img1_path, pilmode='RGB')
	
	height, width, _ = img1.shape
	boundaries = [[0,0],[width,height],[width,0],[0,height]]
	print(boundaries)
	HC = twoD_HC_representation(boundaries)
	HC.HC_representation = np.matmul(H,HC.HC_representation)
	HC.normalize_HC_representation()
	print(HC.HC_representation)

	maximum = np.max(HC.HC_representation,1)

	
	img2 = np.ones((int(np.ceil(maximum[1])),int(np.ceil(maximum[0])),3), dtype= np.uint8)*255

	H_inv = np.linalg.inv(H)
	for width in tqdm(range(img2.shape[1]-1)):
		for height in range(img2.shape[0]-1):
			HC = twoD_HC_representation([[width,height]])
			HC.HC_representation = np.matmul(H_inv,HC.HC_representation)
			HC.normalize_HC_representation()
			x,y = [HC.HC_representation[0][0],HC.HC_representation[1][0]]
			if int(np.floor(y))>0 and int(np.floor(x))>0 and int(np.floor(x))< img1.shape[1] and int(np.floor(y)<img1.shape[0]):
				img2[height, width,:] = img1[int(np.floor(y)),int(np.floor(x)),:]
	return img2

def apply_homography_v3(img1_path, H):
	img1 = io.imread(img1_path, pilmode='RGB')
	
	height, width, _ = img1.shape
	boundaries = [[0,0],[width,height],[width,0],[0,height]]
	HC = twoD_HC_representation(boundaries)
	HC.HC_representation = np.matmul(H,HC.HC_representation)
	HC.normalize_HC_representation()

	maximum = np.max(HC.HC_representation,1)
	minimum = np.min(HC.HC_representation,1)
	
	# img2 = np.ones((int(np.ceil(maximum[1])),int(np.ceil(maximum[0])),3), dtype= np.uint8)*255
	img2 = np.ones((int(np.ceil(maximum[1])-np.floor(minimum[1])),int(np.ceil(maximum[0])-np.floor(minimum[0])),3), dtype= np.uint8)*255
	# img2 = np.ones((3000,3000), dtype= np.uint8)*255
	H_inv = np.linalg.inv(H)
	print("min:", minimum)
	for width in tqdm(range(img2.shape[1]-1)):
		for height in range(img2.shape[0]-1):
			HC = twoD_HC_representation([[width+np.floor(minimum[0]),height+np.floor(minimum[1])]])
			# HC = twoD_HC_representation([[width,height]])
			HC.HC_representation = np.matmul(H_inv,HC.HC_representation)
			HC.normalize_HC_representation()
			x,y = [HC.HC_representation[0][0],HC.HC_representation[1][0]]
			# if int(np.floor(y))>0 and int(np.floor(x))>0 and int(np.floor(x))< img1.shape[1] and int(np.floor(y)<img1.shape[0]):
			try:
				if int(np.floor(y))>0 and int(np.floor(x))>0 and int(np.floor(x))< img1.shape[1] and int(np.floor(y)<img1.shape[0]):
					img2[height, width,:] = img1[int(np.floor(y)),int(np.floor(x)),:]
			except:
				continue
	return img2

def compute_fundamental_matrix(correspondences):
	"""
	Given a set of correspondences from the Left and the Right cameras. It returns the fundamental matrix assuming the L camera as reference.
	"""
	A = []	
	T1, T2 = compute_t(np.array(correspondences))
	for correspondence in correspondences:
		L,R = correspondence
		L = np.dot(T1, L+[1])
		R = np.dot(T2, R+[1])
		A.append([R[0]*L[0],
				  R[0]*L[1],
				  R[0],
				  R[1]*L[0],
				  R[1]*L[1],
				  R[1],
				  L[0],
				  L[1],1])
	A = np.vstack(A)
	U,D,V = np.linalg.svd(A)
	F = V[-1].reshape(3,3)
	U,D,V = np.linalg.svd(F)
	D[-1] = 0 
	F = np.matmul(np.matmul(U,np.diag(D)),V)
	F = T2.T@F@T1
	F = F/F[2,2]
	return F

def compute_t(correspondences):
	X = correspondences[:,0,:]
	X_mean = np.mean(X,0)
	d = np.mean([np.linalg.norm(i) for i in X-X_mean])
	S = np.sqrt(2)/d
	X_prime = correspondences[:,1,:]
	X_prime_mean = np.mean(X_prime,0)
	d_prime = np.mean([np.linalg.norm(i) for i in X_prime-X_prime_mean])
	S_prime = np.sqrt(2)/d_prime

	T1 = np.eye(3)
	T2 = np.eye(3)

	T1[0:2,0:2] = S*np.eye(2)
	T1[0:2,2] = -S*X_mean

	T2[0:2,0:2] = S_prime*np.eye(2)
	T2[0:2,2] = -S_prime*X_prime_mean

	return T1, T2

def compute_epipoles(F):
	U,D,V = np.linalg.svd(F)
	e = V[-1]/V[-1][-1]

	U,D,V = np.linalg.svd(F.T)
	e_prime = V[-1]/V[-1][-1]
	return e.reshape(3,1), e_prime.reshape(3,1)

def get_matrix_cross_product_representation(e):
	return np.array([[0,-e[2][0],e[1][0]],[e[2][0],0,-e[0][0]],[-e[1][0],e[0][0],0]])

def compute_Ps(F,e,e_prime):
	P = np.hstack([np.eye(3),np.zeros((3,1))])
	x = np.hstack([F,e_prime])
	P_prime = np.hstack([np.matmul(get_matrix_cross_product_representation(e_prime), F),e_prime])
	return P,P_prime

def triangulate_world_points(correspondences, P, P_prime):
	"""
	returns a list of world points corresponding to the list of input points
	"""
	world_points =[]
	for correspondence in correspondences:
		A=[]
		L,R = correspondence
		A.append(np.array(L[0] * P[2,:]- P[0,:]).reshape(1,4))
		A.append(np.array(L[1] * P[2,:]- P[1,:]).reshape(1,4))
		A.append(np.array(R[0] * P_prime[2,:]- P_prime[0,:]).reshape(1,4))
		A.append(np.array(R[1] * P_prime[2,:]- P_prime[1,:]).reshape(1,4))
		A = np.vstack(A)
		U,D,V = np.linalg.svd(np.matmul(A.T,A))
		# U,D,V = np.linalg.svd(A)
		world_point  = V[-1]/V[-1][-1]
		world_points.append(world_point.reshape(4,1))

	return world_points

def cost_function(params, correspondences, P):
	P_prime = params.reshape(3,4)
	world_points = triangulate_world_points(correspondences,P, P_prime)
	preds = []
	target = []
	for i, world_point in enumerate(world_points):
		left_reproj = np.matmul(P,world_point)
		right_reproj = np.matmul(P_prime,world_point)
		left_reproj = left_reproj/left_reproj[-1]
		right_reproj = right_reproj/right_reproj[-1]
		preds.extend([left_reproj[0], left_reproj[1],right_reproj[0], right_reproj[1]])
		target.extend([correspondences[i][0][0],correspondences[i][0][1], correspondences[i][1][0],correspondences[i][1][1]])

	return np.array(target).reshape(-1) - np.array(preds).reshape(-1)

def compute_refined_F(P, P_prime):
	F = np.matmul(get_matrix_cross_product_representation(P_prime[:,3].reshape(3,1)), np.matmul(P_prime,np.matmul(P.T,np.linalg.inv(np.matmul(P,P.T)))))
	return F/F[2,2]

def get_rotation_matrix(theta):
	return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

def compute_P_dagger(P):
	# print(np.shape(P))
	return np.matmul(P.T,np.linalg.inv(np.matmul(P,P.T)))

def compute_homographies(image_L_path, image_R_path, correspondences, e, e_prime, P, P_prime):
	Limg = io.imread(image_L_path, as_gray='True')
	Rimg = io.imread(image_R_path, as_gray='True')

	H,W = Rimg.shape
	translation = np.eye(3)
	translation[0:2,[2]] = [[-H/2], [-W/2]]
	rotation_angle = np.arctan(-(2*e_prime[1][0]-H)/(2*e_prime[0][0]-W))
	rotation = np.eye(3)
	rotation[0:2,0:2] = get_rotation_matrix(rotation_angle)
	inverse_translation = np.eye(3)
	inverse_translation[0:2,[2]] = [[H/2], [W/2]]
	distance = np.linalg.norm([e_prime[1][0] -H/2,e_prime[0][0]-W/2])
	G = np.eye(3)
	G[2,0] = -1/distance
	H2 = inverse_translation @ G @ rotation @ translation
	M = np.matmul(P,compute_P_dagger(P))
	H0 = np.matmul(H2,M)
	A = []
	B = []
	for i in range(len(correspondences)):
		A.append([correspondences[i][0][0], correspondences[i][0][1],1])
		B.append([correspondences[i][1][0]])
	A = np.vstack(A)
	B = np.vstack(B)
	H_temp = np.matmul(np.linalg.pinv(A),B)
	H = np.eye(3)
	H[0,:] = H_temp[0,:]
	H1 = np.matmul(H,H0)
	return H1/H1[2,2], H2/H2[2,2]

def apply_homographies_and_save_images(image_L_path, H1, image_R_path, H2, out_path1, out_path2):
	img1 = apply_homography_v3(image_L_path,H1)
	img2 = apply_homography_v3(image_R_path,H2)
	imsave(out_path1, img1)
	imsave(out_path2, img2)


def apply_homography_to_points(points, H, rect_image_path, image_path, offset =0):
	img1 = io.imread(image_path, pilmode='RGB')	
	height, width, _ = img1.shape
	boundaries = [[0,0],[width,height],[width,0],[0,height]]
	HC = twoD_HC_representation(boundaries)
	HC.HC_representation = np.matmul(H,HC.HC_representation)
	HC.normalize_HC_representation()
	maximum = np.max(HC.HC_representation,1)
	minimum = np.min(HC.HC_representation,1)
	out_points = []
	for point in points:
		HC = twoD_HC_representation([[point[0],point[1]]], mode = "pixel")#
		HC.HC_representation = np.matmul(H,HC.HC_representation)
		HC.normalize_HC_representation()
		x,y = [HC.HC_representation[0][0],HC.HC_representation[1][0]]
		out_points.append([x-minimum[0]+offset,y -minimum[1] ])

	return out_points
	
def show_matches_for_rectified_images(L_rectified,R_rectified, correspondences, HL, HR, image_L_path, image_R_path, fig_name="default.png"):
	correspondences = np.array(correspondences)
	Limg = io.imread(L_rectified)
	Rimg = io.imread(R_rectified)	
	full_img = np.hstack([Limg,Rimg])
	H,W, _ = Limg.shape
	newL = apply_homography_to_points(correspondences[:,0,:],HL,L_rectified, image_L_path, offset = 0)
	newR = apply_homography_to_points(correspondences[:,1,:],HR,R_rectified, image_R_path, offset = W);plt.figure();
	plt.imshow(full_img)
	colours= ["r", "g", "b", "c", "m", "y", "k"]
	for i in range(len(correspondences)):
		x1,y1 = newL[i]
		x2,y2 = newR[i]
		plt.plot((x1,x2), (y1,y2), marker = "o", color = colours[i%len(colours)])
	plt.axis('off')
	plt.savefig(fig_name, dpi =200)

def plot_img_and_correspondances(L_image,R_image, correspondences,fig_name):
	Limg = io.imread(L_image)
	Rimg = io.imread(R_image)
	full_img = np.hstack([Limg,Rimg])
	H,W, _ = Limg.shape;plt.figure()
	plt.imshow(full_img)
	colours= ["r", "g", "b", "c", "m", "y", "k"]
	for i in range(len(correspondences)):
		x1,y1 = correspondences[i][0]; y1 = y1 + 215
		x2,y2 = correspondences[i][1]
		x2+=W;y2 = y2 + 215
		plt.plot((x1,x2), (y1,y2), marker = "o", color = colours[i%len(colours)], linewidth=1, markersize=6)
	plt.axis("off")
	plt.savefig(fig_name, dpi=200)


def compute_matches(point, array, i, j):
	min_distance = 9999
	min_distance_match = None

	half_win_size = 3
	for l in range(1,np.shape(array)[1]-1):

		try:
			dist = np.linalg.norm(point[i-half_win_size:i+half_win_size+1, j-half_win_size:j+half_win_size+1]-array[i-half_win_size:i+half_win_size+1, l-half_win_size:l+half_win_size+1])
			if dist< min_distance:
				min_distance = dist
				min_distance_match = [i,l]
		except:
			continue

	return min_distance_match

	return None
def filter_matches(row_matches):
	return row_matches

def compute_multiple_correspondences(L_rectified, R_rectified):
    img = cv2.imread(L_rectified)
    L_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.imread(R_rectified)
    R_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    L_edge = cv2.Canny(L_gray, 300, 200, 5)
    R_edge = cv2.Canny(R_gray, 300, 200, 5)
    np.random.seed(25)
    L_matches = np.argwhere(L_edge > 0)
    rand_matches = L_matches[np.random.randint(0, len(L_matches), (40))]
    all_matches = []
    row_matches = []
    for rand_match in rand_matches:
        i, j = rand_match
        if L_edge[i, j]:
            x = compute_matches(L_gray, R_gray, i, j)
            if x is None:
                continue
            newi, newj = x
            row_matches.append([[j, i], [newj, newi]])  

    all_matches.append(filter_matches(row_matches))
    return all_matches


def threeD_plot(correspondences,P,P_prime):
	world_points = triangulate_world_points(correspondences,P,P_prime)
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	new_world_points = np.zeros((len(correspondences),3))

	count = 0
	invalid_points = [43,51,64,73]
	for point in world_points:
		if count not in invalid_points:		
			xs= point[0]
			ys= point[1]
			zs= point[2]
			new_world_points[count,:]= [xs,ys,zs]
			count+=1
			ax.scatter(xs, ys, zs, marker="o")
			ax.text(xs[0], ys[0], zs[0], str(count), 'x')

	pairs = [[0,2],
	[2,4],
	[4,6],
	[6,0],
	[7,3],
	[3,1],
	[1,5],
	[5,7]]

	for pair in pairs:
		ax.plot([new_world_points[pair[0]][0],new_world_points[pair[1]][0]],\
		[new_world_points[pair[0]][1],new_world_points[pair[1]][1]],\
		[new_world_points[pair[0]][2],new_world_points[pair[1]][2]])
	plt.show()


def resizeImageToMatch(img1, img2):
    height1, _, _ = img1.shape
    height2, _, _ = img2.shape
    if height1 > height2:
        img2 = cv2.resize(img2, (img2.shape[1], height1), interpolation=cv2.INTER_LINEAR)
    elif height2 > height1:
        img1 = cv2.resize(img1, (img1.shape[1], height2), interpolation=cv2.INTER_LINEAR)
    return img1, img2


def applyHomographyToPoints(points, H):
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    transformed_points = np.dot(H, points_homogeneous.T).T
    transformed_points /= transformed_points[:, 2][:, np.newaxis]  # Normalize
    return transformed_points[:, :2]  # Return only x, y

def plotImageAndCorrespondences(l_image, r_image, correspondences, fig_name):
    l_img = io.imread(l_image)
    r_img = io.imread(r_image)
    l_img, r_img = resizeImageToMatch(l_img, r_img)
    full_img = np.hstack([l_img, r_img])

    h, w, _ = l_img.shape
    plt.figure(figsize=(12, 8))
    plt.imshow(full_img)
    colours = ["r", "g", "b", "c", "m", "y", "k"]

    for i, (point_l, point_r) in enumerate(correspondences):
        x1, y1 = point_l
        x2, y2 = point_r
        x2 += w  # Adjust x2 for the right image
        plt.plot([x1, x2], [y1, y2], color=colours[i % len(colours)], linewidth=1)
        plt.scatter([x1, x2], [y1, y2], color=colours[i % len(colours)], s=10)
    plt.axis("off")
    plt.savefig(fig_name, dpi=300)
    plt.close()
    
def apply_homography_to_points_v2(points, H):
    """
    Transforms points using the provided homography matrix H.
    """
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    transformed_points = np.dot(H, points_homogeneous.T).T
    transformed_points /= transformed_points[:, 2][:, np.newaxis]  # Normalize
    return transformed_points[:, :2]  # Return only x, y

def transform_correspondences(correspondences, HL, HR):
    def apply_homography_to_point(point, H):
        """Apply a homography H to a single point."""
        point_hc = np.array([point[0], point[1], 1])  # Convert to homogeneous coordinates
        transformed_point_hc = np.dot(H, point_hc)
        transformed_point_hc /= transformed_point_hc[2]  # Normalize
        return [int(round(transformed_point_hc[0])), int(round(transformed_point_hc[1]))]  # Convert to integer

    transformed_correspondences = []
    for correspondence in correspondences:
        left_point, right_point = correspondence
        transformed_left_point = apply_homography_to_point(left_point, HL)
        transformed_right_point = apply_homography_to_point(right_point, HR)
        transformed_correspondences.append([transformed_left_point, transformed_right_point])

    return transformed_correspondences
    

points_img1 = [(349, 672), (707, 732), (594, 1321), (204, 1243), (324, 939), (521, 976), (479, 1172), (274, 1135)]
points_img2 = [(648, 694), (1017, 747), (950, 1348), (544, 1274), (643, 965), (848, 966), (820, 1197), (610, 1162)]

# Combine the points into the required correspondence format
correspondences = [[[int(x1), int(y1)], [int(x2), int(y2)]] 
                   for (x1, y1), (x2, y2) in zip(points_img1, points_img2)]

force_recompute = True
verbose = True
image_L_path = "/home/aolivepe/Computer-Vision/HW9/HW9_images/img1.jpg"
image_R_path = "/home/aolivepe/Computer-Vision/HW9/HW9_images/img2.jpg"

if force_recompute:

	F = compute_fundamental_matrix(correspondences)
	e, e_prime = compute_epipoles(F)
	P,P_prime = compute_Ps(F,e, e_prime)
	world_points = triangulate_world_points(correspondences, P, P_prime)

	refined_P_prime = least_squares(cost_function, P_prime.flatten() , method ='lm' , args =[correspondences , P])
	refined_P_prime = refined_P_prime.x.reshape(3,4)
	refined_P_prime = refined_P_prime/refined_P_prime[2,3]

	refined_F = compute_refined_F(P,refined_P_prime)
	refined_e, refined_e_prime = compute_epipoles(refined_F)

	HL,HR = compute_homographies(image_L_path, image_R_path, correspondences, e, e_prime, P, P_prime)

	apply_homographies_and_save_images(image_L_path, HL, image_R_path, HR, "L_rectified.png", "R_rectified.png")
 

L_rectified="./L_rectified.png"
R_rectified="./R_rectified.png"

# HL = [[3.210640965049326, 0.1437904404884618, -6.486219888302003e-06], [3.295628626995771, 1.947838369522805, 2.289949600253749e-05], [0.00175593721840173, -6.249992099630879e-06, 1]]
# HR = [[2.105557587619295, 0.17625540198054, 0], [2.251598438714073, 1.500461919087117, 607.3189697265625], [0.001212079457401081, 7.172431933101178e-05, 1]]



# Example usage:
# Assuming correspondences, HL, HR are already defined
transformed_correspondences = transform_correspondences(correspondences, HL, HR)
#show_matches_for_rectified_images(L_rectified,R_rectified, transformed_correspondences, HL, HR, image_L_path, image_R_path,"Task1_rectified_images.png")


all_matches = compute_multiple_correspondences(L_rectified,R_rectified)[0]


newlist = []
newlist.extend(correspondences)
newlist.extend(all_matches)
plotImageAndCorrespondences(L_rectified, R_rectified, all_matches, "Task1_multiple_correspondences.png");
# threeD_plot(correspondences, P, P_prime)
# threeD_plot(all_matches, P, P_prime)

