import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from more_itertools import locate
import pandas as pd
from matplotlib import cm
from copy import deepcopy 
import pandas as pd
from scipy.spatial import cKDTree
import time
import threading
import os
import copy

# ------------------------------------
# Point Cloud Processing
# ------------------------------------

class VoxelProcessing():
    
    def __init__ (self):
        self.visFlag = True
        self.ICPFlag = False
        self.KDTreeVisFlag = False
        self.PP_System = True
        self.pre_segment_flag = True

        parameters_file_path = '/home/miguel-cruz/Documents/RAN-3D/Voxel_Processing_Parameters'
        # Create a dictionary to store the parameters
        parameters = {}

        # Read and parse the parameter file
        try:
            with open(parameters_file_path, "r") as file:
                for line in file:
                    line = line.strip()
                    if line and ":" in line:
                        key, value = line.split(":", 1)  # Split only once
                        parameters[key.strip()] = value.strip()
        except FileNotFoundError:
            print(f"Parameter file not found: {parameters_file_path}")
        except Exception as e:
            print(f"An error occurred while reading the parameter file: {e}")

        ## -Parameters- ##
        # PCD LOAD
        self.max_pcd_points = int(parameters.get("MaxPCDPoints"))
        self.voxel_size = int(parameters.get("VoxelSize"))
      
        # GT Extraction
        self.GT_distance_threshold = float(parameters.get("DifThresholdGT"))
        self.GT_k = float(parameters.get("kGT"))
        
        # Plane Segmentation
        self.distance_threshold = float(parameters.get("DistThreshold"))
        self.ransac_n = int(parameters.get("Ranscac_n"))
        self.num_iterations = int(parameters.get("Num_iterations"))
        self.min_points_Percentage = float(parameters.get("Min_points_break"))

        # Clustering
        self.eps = float(parameters.get("Eps"))
        self.min_points = int(parameters.get("Min_points"))
        
        # Detection Validation
        self.validation_k = float(parameters.get("KValidation"))
        self.validation_distance_threshold = float(parameters.get("DifThreshold"))
        # pass

    def loadpcd (self, filepath):

        # Load Mesh (File .STL)
        self.mesh = o3d.io.read_triangle_mesh(filepath)
        self.mesh = self.mesh.compute_vertex_normals()

        if self.visFlag == True:
            # You can visualize both the mesh and the sampled point cloud together
            VoxelProcessing.visualization([self.mesh], 'Original Mesh')

        

        self.original_pcd = self.mesh.sample_points_poisson_disk(self.max_pcd_points)


        ########################## Visualization ######################### 
         
        if self.visFlag == True:
            VoxelProcessing.visualization([self.original_pcd], 'Original PCD')
        
        ##################################################################
        
        
        # PCD Copy
        self.pcd = deepcopy(self.original_pcd)
        self.pcd.paint_uniform_color([1, 0.5, 0.3])

        # # Create a voxel grid to downsample the point cloud
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, self.voxel_size)  #self.voxel_size=1

        # # # Get the downsampled point cloud
        downsampled_points = voxel_grid.get_voxels()  # returns list of voxels
        indices = np.stack(list(vx.grid_index for vx in downsampled_points))
        downsampled_point_cloud = o3d.geometry.PointCloud()
        downsampled_point_cloud.points = o3d.utility.Vector3dVector(indices)
        self.damaged_pcd = downsampled_point_cloud
        self.damaged_pcd.paint_uniform_color([1, 0.5, 0.3])
              


        ########################## Visualization ######################### 
         
        if self.visFlag == True:
            
            frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=20, origin=np.array([0, 0, 0]))
            
            # PCD
            geometry_list=[]
            geometry_list.append(self.pcd)
            # geometry_list.append(frame)
            VoxelProcessing.visualization(geometry_list, 'PCD')

            # Damaged Model 
            geometry_list=[]
            geometry_list.append(self.damaged_pcd)
            # geometry_list.append(frame)
            VoxelProcessing.visualization(geometry_list, 'Damaged Model')

        ################################################################# 


    def GTExtraction(self, filepath):
   
        self.not_detected_GT = 0

        # Load Mesh (File .STL)
        mesh = o3d.io.read_triangle_mesh(filepath)
        mesh = mesh.compute_vertex_normals()

        # self.undamaged_pcd = mesh.sample_points_poisson_disk(self.max_pcd_points)
        self.undamaged_pcd = mesh.sample_points_poisson_disk(20000)
        self.undamaged_pcd.paint_uniform_color([1, 0.5, 0.3])

        # # Create a voxel grid to downsample the point cloud
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.undamaged_pcd, self.voxel_size)

        # # # Get the downsampled point cloud
        downsampled_points = voxel_grid.get_voxels()  # returns list of voxels
        indices = np.stack(list(vx.grid_index for vx in downsampled_points))
        downsampled_point_cloud = o3d.geometry.PointCloud()
        downsampled_point_cloud.points = o3d.utility.Vector3dVector(indices)
        self.undamaged_pcd = downsampled_point_cloud
        self.undamaged_pcd.paint_uniform_color([0, 0, 1])
     
        

        ########################## Visualization ######################### 
         
        # if self.visFlag == True:
            
        #     frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=20, origin=np.array([0, 0, 0]))
            
        #     # PCD
        #     geometry_list=[]
        #     geometry_list.append(self.undamaged_pcd)
        #     # geometry_list.append(frame)
        #     VoxelProcessing.visualization(geometry_list, 'Undamaged Model')

        ################################################################# 



        ########################## Alignemnt #########################

        if self.ICPFlag == True:
            # Perform manual 90-degree rotation (if needed)
            angle = np.radians(90)  # Rotation angle in degrees
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle), 0],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1]
            ])

            # Apply the rotation to the damaged point cloud
            self.damaged_pcd.transform(rotation_matrix)

            # Apply translation to align the models (adjust the values as needed)
            translation_vector = [0, 0, 20]  # Translate 20mm along the X-axis (example)
            translation_matrix = np.eye(4)  # Identity matrix
            translation_matrix[:3, 3] = translation_vector  # Add translation to the matrix

            self.damaged_pcd.transform(translation_matrix)

            # Perform ICP alignment
            threshold = 0.5  # Distance threshold for point matching
            trans_init = np.identity(4)  # Initial transformation (identity matrix)

            icp_result = o3d.pipelines.registration.registration_icp(
                self.damaged_pcd, self.undamaged_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            # Apply the transformation to align the damaged point cloud
            self.damaged_pcd.transform(icp_result.transformation)

            # Visualize the aligned point clouds
            o3d.visualization.draw_geometries([self.undamaged_pcd, self.damaged_pcd], 
                                            window_name="Aligned Point Clouds")
            

        ################################################################# 

        # # Extract the coordinates of the point clouds
        undamaged_points = np.asarray(self.undamaged_pcd.points)
        damaged_points = np.asarray(self.damaged_pcd.points)
        
        
        # Build a KD-Tree for the undamaged point cloud
        undamaged_tree = cKDTree(undamaged_points)

        
        if self.KDTreeVisFlag == True:
        
            # Function to create a bounding box from min and max corners
            def create_bounding_box(min_corner, max_corner, color=[1, 0, 0]):
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
                bbox.color = color
                return bbox

            # Function to recursively plot the KD tree with bounding boxes up to a specified depth
            def plot_kdtree(node_indices, depth=0, max_depth=12, min_corner=[0, 0, 0], max_corner=[100, 100, 100]):
                if len(node_indices) == 0 or depth > max_depth:
                    return []

                k = depth % 3  # 3D
                median_index = len(node_indices) // 2
                node_indices.sort(key=lambda index: undamaged_points[index][k])
                median_point = undamaged_points[node_indices[median_index]]

                # Create the bounding box for the current node
                bbox = create_bounding_box(min_corner, max_corner, color=[depth % 2, (depth + 1) % 2, (depth + 2) % 2])

                # Split the indices into left and right nodes
                left_indices = node_indices[:median_index]
                right_indices = node_indices[median_index + 1:]

                # Update the bounding box dimensions
                if k == 0:  # x-axis
                    left_max_corner = [median_point[0], max_corner[1], max_corner[2]]
                    right_min_corner = [median_point[0], min_corner[1], min_corner[2]]
                    left_min_corner = min_corner
                    right_max_corner = max_corner
                elif k == 1:  # y-axis
                    left_max_corner = [max_corner[0], median_point[1], max_corner[2]]
                    right_min_corner = [min_corner[0], median_point[1], min_corner[2]]
                    left_min_corner = min_corner
                    right_max_corner = max_corner
                else:  # z-axis
                    left_max_corner = [max_corner[0], max_corner[1], median_point[2]]
                    right_min_corner = [min_corner[0], min_corner[1], median_point[2]]
                    left_min_corner = min_corner
                    right_max_corner = max_corner

                # Recursively create bounding boxes for left and right nodes, up to max_depth
                left_boxes = plot_kdtree(left_indices, depth + 1, max_depth, left_min_corner, left_max_corner)
                right_boxes = plot_kdtree(right_indices, depth + 1, max_depth, right_min_corner, right_max_corner)

                return [bbox] + left_boxes + right_boxes

            # Get bounding boxes from KD tree, limited to 6 levels
            bounding_boxes = plot_kdtree(list(range(len(undamaged_points))), max_depth=3)

            # Visualization
            frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=20, origin=np.array([0, 0, 0]))
            # geometry_list = [self.undamaged_pcd] + bounding_boxes + [frame]
            geometry_list = [self.undamaged_pcd] + bounding_boxes 

            lookat = np.array([60.59954383327868, 33.962101398113305, 26.664280558723334])
            up = np.array([-0.43442394124106293, 0.65595421213830563, -0.6172519022689168])
            front = np.array([0.36140234882980843, 0.75465514998715477, 0.54761660571805393])
            zoom = 1.5

            if self.visFlag:
                o3d.visualization.draw_geometries(
                    geometry_list,
                    window_name='Nominal Model with KD Tree Bounding Boxes',
                    width=960,
                    height=540,
                    left=50,
                    top=50,
                    point_show_normal=False,
                    mesh_show_wireframe=False,
                    mesh_show_back_face=False,
                    lookat=lookat,
                    up=up,
                    front=front,
                    zoom=zoom
                )
        
        
        
        # Define a distance threshold for matching points
        # self.GT_distance_threshold = 10 # para 5k pontos
        # self.GT_distance_threshold = 3 # para 20k pontos
        # self.GT_distance_threshold = 1.75 # para 50k pontos
        # print(self.GT_distance_threshold)

        
        missing_indices = []
        lines=[]
        for i, point in enumerate(damaged_points):
            # Query the KD-Tree to find the nearest neighbor in the undamaged cloud
            _, index = undamaged_tree.query(point, self.GT_k)  #self.GT_k=1

            # Check if the distance is below the threshold
            if np.linalg.norm(point - undamaged_points[index]) > self.GT_distance_threshold:
                missing_indices.append(i)
                # index_mapping.append((i, index))

            line = [damaged_points[i], undamaged_points[index]]
            lines.append(line)

        #print(lines)

        # # Convert lines to Open3D LineSet
        # lines_flat = [line for pair in lines for line in pair]  # Flatten the nested arrays
        # line_set_points = np.vstack(lines_flat)
        # line_set_lines = np.arange(len(lines_flat)).reshape(-1, 2)

        # line_set = o3d.geometry.LineSet()
        # line_set.points = o3d.utility.Vector3dVector(line_set_points)
        # line_set.lines = o3d.utility.Vector2iVector(line_set_lines)

        # # Visualize point clouds and KD-Tree connections
        # o3d.visualization.draw_geometries([self.undamaged_pcd])
        # o3d.visualization.draw_geometries([self.damaged_pcd, line_set])
        # o3d.visualization.draw_geometries([self.undamaged_pcd, self.damaged_pcd, line_set])

        # # Create a new PointCloud with the unique points (difference)
        difference_points = damaged_points[missing_indices]
        self.GTdamage_pcd = o3d.geometry.PointCloud()
        self.GTdamage_pcd.points = o3d.utility.Vector3dVector(difference_points)

        # # Color the difference points in red
        red_color = [1, 0, 0] 
        self.GTdamage_pcd.paint_uniform_color(red_color)
        
        # Perform DBSCAN clustering
        cluster_idx = np.array(self.GTdamage_pcd.cluster_dbscan(12, 8, print_progress=False))

        # Get unique cluster indices
        objects_idx = list(set(cluster_idx))
        # print('gt clusters: ' + str(objects_idx))

        try:

            # Find the cluster with the most points
            largest_cluster_idx = max(objects_idx, key=lambda idx: np.sum(cluster_idx == idx))
            
            
            # Filter the point cloud to keep only the largest cluster
            largest_cluster = self.GTdamage_pcd.select_by_index(np.where(cluster_idx == largest_cluster_idx)[0])
            
            # Convert the largest cluster to a new PointCloud for visualization
            largest_cluster_pcd = o3d.geometry.PointCloud()
            largest_cluster_pcd.points = largest_cluster.points

            # Now, 'largest_cluster_pcd' contains the largest cluster, and you can visualize it.
            # o3d.visualization.draw_geometries([largest_cluster_pcd])

            self.GTdamage_pcd = largest_cluster_pcd
            self.GTdamage_pcd.paint_uniform_color(red_color)

            ########################## Visualization ######################### 
         
            
            if self.visFlag == False:
                # Frame
                frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=20, origin=np.array([0, 0, 0]))
                
                geometry_list=[]
                geometry_list.append(self.damaged_pcd)
                geometry_list.append(self.GTdamage_pcd)
                # geometry_list.append(frame)
                VoxelProcessing.visualization(geometry_list, 'Ground Thruth (RED) + Damaged Model')
            
            if self.visFlag == True:
                # Damaged Model 
                geometry_list=[]
                geometry_list.append(self.pcd)
                geometry_list.append(self.undamaged_pcd)
                # geometry_list.append(frame)
                VoxelProcessing.visualization(geometry_list, 'Undamaged + Damaged Model')

            ######################################################################
    
            return(self.GTdamage_pcd)
        
        except Exception as e:

            self.not_detected_GT = 1

            return(self.GTdamage_pcd)

    # def DamageDetection(self):
        # ----------------------------
        # Plane Segmentation
        # ----------------------------

        # # Plane Segmentation Parameters
        # # distance_threshold = 1         #0.0875
        # # ransac_n = 3
        # # num_iterations = 800

        # pcd_to_segment = self.damaged_pcd
        # self.detected_planes = []
        # self.detected_eq_planes = []

        # # Use a pastel colormap (up to 20 unique colors)
        # colormap = cm.get_cmap("Pastel2", 20)

        # while True:
        #     # Segment a plane
        #     plane_model, inliers = pcd_to_segment.segment_plane(
        #         self.distance_threshold, self.ransac_n, self.num_iterations
        #     )
        #     [a, b, c, d] = plane_model
        #     print(f"Plane model: {plane_model}")
        #     self.detected_eq_planes.append(plane_model)

        #     # Extract plane and remaining points
        #     inlier_cloud = pcd_to_segment.select_by_index(inliers)
        #     outlier_cloud = pcd_to_segment.select_by_index(inliers, invert=True)
        #     pcd_to_segment = outlier_cloud

        #     # Assign a pastel color to this plane
        #     color = colormap(len(self.detected_planes) % 20)[0:3]
        #     inlier_cloud.paint_uniform_color(color)
        #     self.detected_planes.append(inlier_cloud)

        #     # Stop if the remaining point cloud is too small
        #     if len(pcd_to_segment.points) < self.max_pcd_points * self.min_points_Percentage:
        #         break
 
    def segment_planes(self, input_pcd, max_planes=1):
        print("[Segmenting Planes...]")
        planes = []
        plane_models = []

        remaining_pcd = input_pcd
        colormap = cm.get_cmap("Pastel2", max_planes)

        for i in range(max_planes):
            if len(remaining_pcd.points) < self.max_pcd_points * self.min_points_Percentage:
                break

            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold= 2,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
            )

            inlier_cloud = remaining_pcd.select_by_index(inliers)
            outlier_cloud = remaining_pcd.select_by_index(inliers, invert=True)

            inlier_cloud.paint_uniform_color(colormap(i % max_planes)[0:3])

            planes.append(inlier_cloud)
            plane_models.append(plane_model)

            remaining_pcd = outlier_cloud

        return planes, plane_models
    

    def select_plane_by_click(self, planes):
        print("\n[Plane Selection by Clicking]")
        print("Click a point on the desired plane using [Shift + Left Click]")
        print("Press [Q] to confirm")

        import copy
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()

        for plane in planes:
            vis.add_geometry(copy.deepcopy(plane))

        vis.run()
        picked_indices = vis.get_picked_points()
        vis.destroy_window()

        if not picked_indices:
            print("No point selected!")
            return None

        picked_point = np.asarray(planes[0].points)[picked_indices[0]]

        # Find which plane contains the picked point (within tolerance)
        for plane in planes:
            points = np.asarray(plane.points)
            if any(np.allclose(picked_point, pt, atol=1.0) for pt in points):
                print("Selected plane found.")
                return plane

        print("No matching plane found.")
        return None



    def DamageDetection(self):


        # # Step 1: Segment planes
        # planes, models = self.segment_planes(self.damaged_pcd)

        # # Step 2: User selects which plane to keep
        # selected_plane = self.select_plane_by_click(planes)

        # # Step 3: Replace damaged_pcd with just the selected plane for further processing
        # if selected_plane is not None:
        #     self.damaged_pcd = selected_plane
        # else:
        #     print("Plane selection failed. Using full point cloud.")


        # # ----------------------------
        # # Plane Segmentation
        # # ----------------------------
        # pcd_to_segment = self.damaged_pcd
        # self.detected_planes = []
        # self.detected_eq_planes = []

        # colormap = cm.get_cmap("Pastel2", 8)

        # while True:
        #     plane_model, inliers = pcd_to_segment.segment_plane(
        #         self.distance_threshold, self.ransac_n, self.num_iterations
        #     )
        #     [a, b, c, d] = plane_model
        #     print(f"Plane model: {plane_model}")
        #     self.detected_eq_planes.append(plane_model)

        #     inlier_cloud = pcd_to_segment.select_by_index(inliers)
        #     outlier_cloud = pcd_to_segment.select_by_index(inliers, invert=True)
        #     pcd_to_segment = outlier_cloud

        #     color = colormap(len(self.detected_planes) % 20)[0:3]
        #     inlier_cloud.paint_uniform_color(color)
        #     self.detected_planes.append(inlier_cloud)

        #     if len(pcd_to_segment.points) < self.max_pcd_points * self.min_points_Percentage:
        #         break
            

        
        # # Save remaining non-planar points (defects)
        # self.defects_pcd = outlier_cloud

        # # Color the defects (outliers) in red
        # self.defects_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        # def visualize_with_custom_view(geometries):
        #     vis = o3d.visualization.Visualizer()
        #     vis.create_window()

        #     for g in geometries:
        #         vis.add_geometry(g)

        #     # Custom camera parameters
        #     ctr = vis.get_view_control()
        #     parameters = o3d.camera.PinholeCameraParameters()

        #     parameters.extrinsic = np.array([
        #         [0.36140235, 0.75465515, 0.54761661, 0.0],
        #         [-0.43442394, 0.65595421, -0.6172519, 0.0],
        #         [-0.824, -0.045, 0.564, 0.0],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ])  # optional custom extrinsic, can be identity if unknown

        #     parameters.intrinsic = ctr.convert_to_pinhole_camera_parameters().intrinsic

        #     # Setup lookat, up, front, zoom manually
        #     ctr.set_front([0.36140235, 0.75465515, 0.54761661])
        #     ctr.set_up([-0.43442394, 0.65595421, -0.6172519])
        #     ctr.set_lookat([60.59954383, 33.9621014, 26.66428056])
        #     ctr.set_zoom(10)

        #     vis.run()
        #     vis.destroy_window()



        # #  Show result
        # visualize_with_custom_view(self.detected_planes + [self.defects_pcd])
        # # o3d.visualization.draw_geometries(self.detected_planes + [self.defects_pcd])

        # ----------------------------
        # Plane Segmentation
        # ----------------------------
        pcd_to_segment = self.damaged_pcd
        self.detected_planes = []
        self.detected_eq_planes = []

        colormap = cm.get_cmap("Pastel2", 8)

        iteration = 0
        while True:
            plane_model, inliers = pcd_to_segment.segment_plane(
                self.distance_threshold, self.ransac_n, self.num_iterations
            )
            [a, b, c, d] = plane_model
            print(f"Plane model: {plane_model}")
            self.detected_eq_planes.append(plane_model)

            inlier_cloud = pcd_to_segment.select_by_index(inliers)
            outlier_cloud = pcd_to_segment.select_by_index(inliers, invert=True)
            pcd_to_segment = outlier_cloud

            color = colormap(len(self.detected_planes) % 8)[0:3]
            inlier_cloud.paint_uniform_color(color)
            self.detected_planes.append(inlier_cloud)

            # -------------------------------
            # Optional: Show segmentation step-by-step
            # -------------------------------
            print(f"[Segmentation Step {iteration}] Showing current plane")
            o3d.visualization.draw_geometries(
                self.detected_planes + [pcd_to_segment],
                window_name=f"Plane Segmentation Step {iteration}",
                width=1280, height=720
            )
            iteration += 1

            if len(pcd_to_segment.points) < self.max_pcd_points * self.min_points_Percentage:
                break

        # Save remaining non-planar points (defects)
        self.defects_pcd = outlier_cloud
        self.defects_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red

        # -------------------------------------
        # Final Visualization (Custom Camera)
        # -------------------------------------
        def visualize_with_custom_view(geometries):
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            for g in geometries:
                vis.add_geometry(g)

            ctr = vis.get_view_control()
            parameters = o3d.camera.PinholeCameraParameters()
            parameters.extrinsic = np.identity(4)  # Or custom if you have one
            parameters.intrinsic = ctr.convert_to_pinhole_camera_parameters().intrinsic

            # Apply viewpoint
            ctr.set_front([0.36140235, 0.75465515, 0.54761661])
            ctr.set_up([-0.43442394, 0.65595421, -0.6172519])
            ctr.set_lookat([60.59954383, 33.9621014, 26.66428056])
            ctr.set_zoom(10)

            vis.run()
            vis.destroy_window()

        # Show final state: segmented planes + red defect cloud
        visualize_with_custom_view(self.detected_planes + [self.defects_pcd])
        

        # # ----------------------------
        # # Clustering
        # # ----------------------------

        # # Clustering Parameters
        # # eps = 15                                                                                                        # Parameter !
        # # min_points = 35                                                                                                    # Parameter !
        

        # # Clustering 
        # cluster_idx = np.array(self.defects_pcd.cluster_dbscan(self.eps, self.min_points, print_progress=False))                           # Parameter !
        
        # # Clusters Index
        # objects_idx = list(set(cluster_idx))

        # print('clusters: ' + str(objects_idx))

        # for object_idx in objects_idx:
        #     if object_idx == -1:
        #         objects_idx.remove(-1)
        #         print('cluster -1 removed')

        # # objects_idx.remove(-1)

        # # Defects colormap
        # colormap = cm.tab10(list(range(0,len(objects_idx))))
        # self.objects=[]


        
        # # ----------------------------------------------------------------------------
        # if self.PP_System == True:
        #     # For cycle to analyze each defect/possible defect    
        #     picked_points = []

        #     print("")
        #     print(
        #         "1) Please pick at least three correspondences using [shift + left click]"
        #     )
        #     print("   Press [shift + right click] to undo point picking")
        #     print("2) Afther picking points, press q for close the window")
        #     vis = o3d.visualization.VisualizerWithEditing()
        #     vis.create_window()
        #     vis.add_geometry(self.defects_pcd)
        #     vis.run()  # user picks points
        #     vis.get_picked_points()
        #     vis.destroy_window()
        #     print("")
        
        #     # Selection of non damage point 
        #     picked_points = np.asarray(self.defects_pcd.points)[vis.get_picked_points()]
        # # ----------------------------------------------------------------------------


        
        # for object_idx in objects_idx:
            

        #     object_point_idx = list(locate(cluster_idx, lambda X: X== object_idx))
        #     object_points = self.defects_pcd.select_by_index(object_point_idx)

        #     # Create a dictionary to represent objects (damages)
        #     self.d = {}
        #     self.d['idx'] = str(object_idx)
        #     self.d['points'] = object_points
        #     # self.d['color'] = colormap[object_idx, 0:3]
        #     # self.d['color'] = (0,1,0)
        #     # self.d['points'].paint_uniform_color(self.d['color'])
        #     color = colormap[object_idx % len(colormap)][0:3]  # Normalize for RGB
        #     self.d['color'] = color
        #     self.d['points'].paint_uniform_color(color)
            
            
        #     # Selection of non defects
        #     num_points = len(np.asarray(object_points.points))
        #     points = []
        #     not_defect_flag = False
            
        #     ########################## Pick Point False Damage Delete ######################### 

        #     if self.PP_System == True:  
        #         # If using picked points
        #         for num in range(num_points):    
        #             points.append(np.asarray(object_points.points[num]))
                
        #         for point in points:
        #             for picked_point in picked_points:
        #                 if picked_point[0] == point[0] and picked_point[1] == point[1] and picked_point[2] == point[2] :
        #                     not_defect_flag = True

        #         if not_defect_flag == True:
        #             print('Detection: ' + str(object_idx) + ' discarded \n')
        #             self.d['type'] = str('non damage')
                    
        #         else:
        #             self.d['type'] = str('damage')
        #             self.objects.append(self.d)  # Add Defect
        #     # ----------------------------------------------------------------------------
        #     else:
        #         self.d['type'] = str('damage')
        #         self.objects.append(self.d)  # Add Defect

        #     ################################################################################## 

        # ########################## Visualization ######################### 
         
        #     if self.visFlag == True:
            
        #         # Frame
        #         frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=20, origin=np.array([0, 0, 0]))
                
        #         geometry_list=[]
        #         geometry_list.append(self.damaged_pcd)

        #         # Select each object to draw
        #         for object in self.objects:
        #             geometry_list.append(object['points'])
        #         geometry_list.append(frame)
        #         VoxelProcessing.visualization(geometry_list, 'Damages + Damaged Model')

        #         # Damage isoalted
        #         geometry_list=[]
        #         geometry_list.append(frame)
        #         # Select each object to draw
        #         for object in self.objects:
        #             geometry_list.append(object['points'])
        #         VoxelProcessing.visualization(geometry_list, 'Detected Damages Isolated')

        # ######################################################################

        # self.damages=[]
        # for object in self.objects:
        #     self.damages.append(object['points'])

       

        # # print(self.damages)
        # return(self.damages)

        # ----------------------------
        # Clustering
        # ----------------------------

        # DBSCAN Clustering
        cluster_idx = np.array(self.defects_pcd.cluster_dbscan(self.eps, self.min_points, print_progress=False))

        # Extract unique cluster labels (exclude noise: -1)
        objects_idx = list(set(cluster_idx))
        if -1 in objects_idx:
            objects_idx.remove(-1)
            print('cluster -1 removed')

        print('Detected clusters:', objects_idx)

        # Generate a color map for the clusters
        colormap = cm.get_cmap('tab10', len(objects_idx))
        self.objects = []

        # Prepare list of colored clusters for visualization
        colored_clusters = []

        for object_idx in objects_idx:
            # Select points for this cluster
            indices = np.where(cluster_idx == object_idx)[0]
            cluster_pcd = self.defects_pcd.select_by_index(indices)

            # Assign unique color
            color = colormap(object_idx % len(objects_idx))[0:3]  # RGB triplet
            cluster_pcd.paint_uniform_color(color)

            # Store for picking and display
            colored_clusters.append(cluster_pcd)

        # -----------------------------------------
        # Interactive Picking (if enabled)
        # -----------------------------------------
        picked_points = []

        if self.PP_System == True:
            # For cycle to analyze each defect/possible defect    
            picked_points = []

            print("")
            print(
                "1) Please pick at least three correspondences using [shift + left click]"
            )
            print("   Press [shift + right click] to undo point picking")
            print("2) Afther picking points, press q for close the window")
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            vis.add_geometry(self.defects_pcd)
            vis.run()  # user picks points
            vis.get_picked_points()
            vis.destroy_window()
            print("")
        
            # Selection of non damage point 
            picked_points = np.asarray(self.defects_pcd.points)[vis.get_picked_points()]
            
        # -----------------------------------------
        # Damage Classification & Storage
        # -----------------------------------------
        for object_idx in objects_idx:
            indices = np.where(cluster_idx == object_idx)[0]
            cluster_pcd = self.defects_pcd.select_by_index(indices)
            color = colormap(object_idx % len(objects_idx))[0:3]

            # Cluster dictionary
            obj = {
                'idx': str(object_idx),
                'points': cluster_pcd,
                'color': color
            }

            cluster_pcd.paint_uniform_color(color)

            # Check against picked points (if applicable)
            is_defect = True
            if self.PP_System:
                cluster_points = np.asarray(cluster_pcd.points)
                for point in cluster_points:
                    if any(np.allclose(point, picked_point) for picked_point in picked_points):
                        print(f'Cluster {object_idx} marked as non-damage')
                        obj['type'] = 'non damage'
                        is_defect = False
                        break

            if is_defect:
                obj['type'] = 'damage'
                self.objects.append(obj)

        # -----------------------------------------
        # Visualization (optional)
        # -----------------------------------------
        if self.visFlag:
            frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=20, origin=[0, 0, 0])

            # Full model with defects
            # model_scene = [self.damaged_pcd] + [obj['points'] for obj in self.objects] + [frame]
            model_scene = [self.damaged_pcd] + [obj['points'] for obj in self.objects] 
            VoxelProcessing.visualization(model_scene, 'Damages + Damaged Model')

            # Isolated damage clusters
            # damage_scene = [frame] + [obj['points'] for obj in self.objects]
            damage_scene = [obj['points'] for obj in self.objects]
            VoxelProcessing.visualization(damage_scene, 'Detected Damages Isolated')

        # -----------------------------------------
        # Return damages
        # -----------------------------------------
        self.damages = [obj['points'] for obj in self.objects]



        # bbox_list = []
        # for damage in self.damages:
        #     bbox = damage.get_axis_aligned_bounding_box()
        #     bbox.color = (1, 0, 0)  # Red
        #     bbox_list.append(bbox)

        # # Add them to your visualization scene
        # scene = [self.damaged_pcd] + self.damages + bbox_list
        # VoxelProcessing.visualization(scene, 'Damages with Bounding Boxes')

        offset = 5.0  # Adjust this value for how much larger you want the box

        bbox_list = []
        for damage in self.damages:
            bbox = damage.get_axis_aligned_bounding_box()

            # Expand the bounding box
            min_bound = bbox.min_bound - offset
            max_bound = bbox.max_bound + offset

            expanded_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            expanded_bbox.color = (0, 0, 1)  # Red

            bbox_list.append(expanded_bbox)

        # Visualize damages + bounding boxes
        scene = [self.damaged_pcd] + self.damages + bbox_list
        VoxelProcessing.visualization(scene, 'Damages with Expanded Bounding Boxes')
     
    

    

        

    def calculate_metrics(self):
        N_Detected_Damages = 0
        N_False_Detected_Damages = 0
        N_Undetected_Damage = 0

        number_detected_damages = len(self.damages)

        if number_detected_damages != 0:
        
            GT_Damage_points_array = np.array(self.GTdamage_pcd.points)

            # Fazer loop para vários danos

            for i in range(len(self.damages)):
                
                # Initialize variables to keep track of true positives, false positives, and false negatives
                total_true_positives = 0
                total_false_positives = 0
                total_false_negatives = 0
                true_positives = 0
                false_positives = 0
             
                
                Damage_points_array = np.asarray(self.damages[i].points)
                
                if len(GT_Damage_points_array) != 0:

                    # print(len(GT_Damage_points_array))
                    # print(len(Damage_points_array))
                    # Build KD-Trees for both GT_Damage and Detected_Damage
                    
                 
                    T_tree = cKDTree(GT_Damage_points_array)
                    Detected_tree = cKDTree(Damage_points_array)


                    # IoU Test
                    #Convert lists of points to Open3D point clouds
                    gt_pcd = o3d.geometry.PointCloud()
                    gt_pcd.points = o3d.utility.Vector3dVector(GT_Damage_points_array)
                    # print(gt_pcd)
                    detected_pcd = o3d.geometry.PointCloud()
                    detected_pcd.points = o3d.utility.Vector3dVector(Damage_points_array)
                    # print(detected_pcd)
                    # Compute the axis-aligned bounding boxes
                    gt_bbox = gt_pcd.get_axis_aligned_bounding_box()
                    detected_bbox = detected_pcd.get_axis_aligned_bounding_box()
                    
                    gt_min_bound = gt_bbox.get_min_bound()
                    gt_max_bound = gt_bbox.get_max_bound()

                    if gt_min_bound[1] == gt_max_bound[1]:
                        gt_max_bound[1]+=1
                        gt_bbox = o3d.geometry.AxisAlignedBoundingBox(gt_min_bound, gt_max_bound)


                    # Set colors for the bounding boxes
                    gt_bbox.color = (1, 0, 0)  # Red for GT bounding box
                    detected_bbox.color = (0, 1, 0)  # Green for detected bounding box

                    ########################## Visualization ######################### 

                    if self.visFlag == True:
                  
                        # View parameters
                        lookat = np.array([ 60.59954383327868, 33.962101398113305, 26.664280558723334 ])
                        up = np.array([ -0.43442394124106293, 0.65595421213830563, -0.6172519022689168 ])
                        front = np.array([ 0.36140234882980843, 0.75465514998715477, 0.54761660571805393 ])
                        zoom = 1.5
                        o3d.visualization.draw_geometries([ gt_pcd,  gt_bbox, detected_pcd, detected_bbox],
                        window_name='BBOXes IoU',
                        # width=1920,
                        # height=1080,
                        width=960,
                        height=540,
                        left=50,
                        top=50,
                        point_show_normal=False,
                        mesh_show_wireframe=False,
                        mesh_show_back_face=False,
                        lookat=lookat,
                        up=up,
                        front=front,
                        zoom=zoom
                        )

                    ###################################################################

                    # Compute the intersection volume
                    min_corner = np.maximum(gt_bbox.min_bound, detected_bbox.min_bound)
                    max_corner = np.minimum(gt_bbox.max_bound, detected_bbox.max_bound)
                    
                    intersection_volume = np.prod(np.maximum(0, max_corner - min_corner))
                    # print(intersection_volume)
                    # Compute the volume of each box
                    volume1 = np.prod(gt_bbox.get_extent())
                    volume2 = np.prod(detected_bbox.get_extent())
                    
                    # Compute the union volume
                    union_volume = volume1 + volume2 - intersection_volume
                 
                    # Compute IoU
                    iou = intersection_volume / union_volume if union_volume > 0 else 0
                   
                    print(iou)
                    # -----------------------------------------------------------
                    # Point to point GT vs Detection 
                    # -----------------------------------------------------------
                #     for point in Damage_points_array:
                #         # Query the GT_Damage KD-Tree to find the nearest neighbor
                #         _, index = GT_tree.query(point, self.validation_k) #k=0.01

                #         # print('Ponto')
                #         # print(point)
                #         # print('corresponde:')
                #         # print(GT_Damage_points_array[index])
                #         # print('Dif =' + str(np.linalg.norm(point - GT_Damage_points_array[index])))
                    
                #         # Check if the distance to the nearest neighbor is below the threshold
                #         if np.linalg.norm(point - GT_Damage_points_array[index]) <= self.validation_distance_threshold:
                #             true_positives += 1
                #         else:
                #             false_positives += 1
                # else:
                #     not_detected_GT = not_detected_GT + 1
               

                # # Calculate the number of false negatives (GT_Damage points not matched)
                # false_negatives = len(GT_Damage_points_array) - true_positives

                # # Accumulate statistics for all positions
                # total_true_positives += true_positives
                # total_false_positives += false_positives
                # total_false_negatives += false_negatives

                # # Calculate precision, recall, and F1-score for all positions
                
                # if total_true_positives == 0:
                #     precision = 0
                #     recall = 0
                #     f1_score = 0 
                # else:

                #     precision = total_true_positives / (total_true_positives + total_false_positives)
                #     recall = total_true_positives / (total_true_positives + total_false_negatives)
                #     f1_score = 2 * (precision * recall) / (precision + recall)
                
                # # print('precision: ' + str(precision))
                # # print(precision)
                
                # if iou > 0.60:
                if iou > 0.95:
                # if precision > 0.60:
                    N_Detected_Damages = N_Detected_Damages + 1
                    # print("dano detetado")
                else:
                    N_False_Detected_Damages = N_False_Detected_Damages + 1


        else:
            precision = 0 
            recall = 0
            f1_score = 0

        # N_False_Detected_Damages = number_detected_damages - N_Detected_Damages
        N_Undetected_Damage = 1 - N_Detected_Damages 
        
        if N_Undetected_Damage < 0:
            N_Undetected_Damage0 = 0

        return (number_detected_damages, N_Detected_Damages, N_False_Detected_Damages, N_Undetected_Damage, self.not_detected_GT)


    def visualization(geoemtry_list, name):
         
        # View parameters
        lookat = np.array([ 60.59954383327868, 33.962101398113305, 26.664280558723334 ])
        up = np.array([ -0.43442394124106293, 0.65595421213830563, -0.6172519022689168 ])
        front = np.array([ 0.36140234882980843, 0.75465514998715477, 0.54761660571805393 ])
        zoom = 1.5

       
        ############################################################   
        # Call draw_geometries with the correct argument variation
        o3d.visualization.draw_geometries(
                geoemtry_list,
                window_name= name,
                # width=1920,
                # height=1080,
                width=960,
                height=540,
                left=50,
                top=50,
                point_show_normal=False,
                mesh_show_wireframe=False,
                mesh_show_back_face=False,
                lookat=lookat,
                up=up,
                front=front,
                zoom=zoom
            )
        ############################################################ 
    

    def ML3D_correspondences(self):
    

        distance_threshold = 1

        # Extract point arrays from damaged and GT point clouds
        damaged_points = np.asarray(self.damaged_pcd.points)
        gt_points = np.asarray(self.GTdamage_pcd.points)

        # Build a KD-Tree for the GT discrepancy points
        gt_tree = cKDTree(gt_points)

        # Initialize the correspondence dictionary
        self.correspondences = {}

        # print("Corresponding Points (within threshold):")
        # print(f"Distance Threshold: {distance_threshold}")
        # print("Damaged Index (Coordinates) -> GT Index (Coordinates)\n")

        for i, point in enumerate(damaged_points):
            # Find the nearest GT point and the distance to it
            distance, gt_index = gt_tree.query(point)

            # Check if the distance is below the threshold
            if distance <= distance_threshold:
                self.correspondences[i] = gt_index

                # Print the correspondence
                # print(f"{i} {point} -> {gt_index} {gt_points[gt_index]} (Distance: {distance:.2f})")

        # # Save the point clouds
        # o3d.io.write_point_cloud(damaged_pcd_file, self.damaged_pcd)
        # o3d.io.write_point_cloud(gt_pcd_file, self.GTdamage_pcd)
        # print(f"\nSaved damaged PCD to {damaged_pcd_file}")
        # print(f"Saved GT PCD to {gt_pcd_file}")
        
        # return correspondences

    def ML3D_labels_to_pcd(self, filename):

        filename = filename + '.pcd'

        output_dir = '/home/miguel-cruz/Documents/RAN-3D' 

        path = os.path.join(output_dir, filename)
        
        print (path)
        # Get the number of points in the damaged point cloud
        num_points = len(self.damaged_pcd.points)

        # Initialize labels: 0 for no damage, 1 for damage
        self.labels = np.zeros(num_points, dtype=int)

        # Assign labels based on correspondences
        for damaged_index in self.correspondences.keys():
            self.labels[damaged_index] = 1  # Mark as damaged

        # Add labels as a new dimension in the damaged PCD (optional visualization)
        colors = np.zeros((num_points, 3))  # Default color: black
        colors[self.labels == 1] = [1, 0, 0]  # Damaged points in red
        colors[self.labels == 0] = [0, 1, 0]  # Undamaged points in green
        self.damaged_pcd.colors = o3d.utility.Vector3dVector(colors)
        print((self.damaged_pcd.points))
        o3d.visualization.draw_geometries([self.damaged_pcd], window_name='Damaged PCD with Labels')

        # Save the labeled damaged PCD
        # o3d.io.write_point_cloud(path, self.damaged_pcd)
        # print(f"Labeled damaged PCD saved to {damaged_pcd_file}")
        # print (self.labels)
        # return labels

      

    
    def ML3D_print(self):
   
        # print("Printing damaged points and their corresponding labels:")

        filename = 'pointsandlabels'

        with open(filename, 'w') as f:
            f.write("Printing damaged points and their corresponding labels:\n")

            # Iterate through the damaged point cloud
            for i, point in enumerate(self.damaged_pcd.points):
                label = self.labels[i]
                
                # Print the point's index, coordinates, and the label (1 for damage, 0 for no damage)
                # print(f"Point {i} - Coordinates: {point} - Label: {'Damage' if label == 1 else 'No Damage'}")

                # Write the point's index, coordinates, and the label (1 for damage, 0 for no damage)


                f.write(f"Point {i} - Coordinates: {point} - Label: {'Damage' if label == 1 else 'Normal'}\n")
