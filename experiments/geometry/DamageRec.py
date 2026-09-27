import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree, distance
from sklearn.decomposition import PCA
import os

def extract_missing_region(nominal_path, damaged_path, voxel_size=1.0, threshold=3, k=1,
                           use_pca=True, use_cross_section=True, use_feature_matching=True, use_icp=True):
    # Load and sample point clouds from STL
    nominal_mesh = o3d.io.read_triangle_mesh(nominal_path).compute_vertex_normals()
    damaged_mesh = o3d.io.read_triangle_mesh(damaged_path).compute_vertex_normals()

    # Visualize original (misaligned) models
    print("Visualizing misaligned models...")
    o3d.visualization.draw_geometries([nominal_mesh.paint_uniform_color([0, 0, 0.9])],
                                      window_name="Nominal Model")

    # # Apply a rotation of 90 degrees around the Z-axis
    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.radians(30)])
    # damaged_mesh.rotate(rotation_matrix, center=(0, 0, 0))
    # o3d.visualization.draw_geometries([damaged_mesh.paint_uniform_color([1, 0, 0])],
    #                                   window_name="Nominal Model")

    # Apply translation to simulate misalignment for testing
    # translation_vector = np.array([250.0, 0.0, 0.0])  # Translate 150 mm in X and Y
    # damaged_mesh.translate(translation_vector)
    
    nominal_pcd = nominal_mesh.sample_points_poisson_disk(100000) #50k
    damaged_pcd = damaged_mesh.sample_points_poisson_disk(100000)

    # Visualize original (misaligned) models
    print("Visualizing misaligned models...")
    o3d.visualization.draw_geometries([nominal_pcd.paint_uniform_color([0, 0, 0.9])],
                                      window_name="Nominal Model")

    # Visualize original (misaligned) models
    print("Visualizing misaligned models...")
    # Apply a rotation of 30 degrees around the Z-axis
    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.radians(30)])
    # damaged_pcd.rotate(rotation_matrix, center=(0, 0, 0))

    o3d.visualization.draw_geometries([damaged_pcd.paint_uniform_color([1, 0, 0])],
                                      window_name="Damaged Model")

    # Visualize original (misaligned) models
    print("Visualizing misaligned models...")
    o3d.visualization.draw_geometries([nominal_pcd.paint_uniform_color([0, 0, 0.9]),
                                       damaged_pcd.paint_uniform_color([1, 0, 0])],
                                      window_name="Original Misaligned Models")
    
    # Downsample point clouds
    voxel_grid_nominal = o3d.geometry.VoxelGrid.create_from_point_cloud(nominal_pcd, voxel_size)
    voxel_grid_damaged = o3d.geometry.VoxelGrid.create_from_point_cloud(damaged_pcd, voxel_size)
    
    downsampled_nominal = np.stack([vx.grid_index for vx in voxel_grid_nominal.get_voxels()])
    downsampled_damaged = np.stack([vx.grid_index for vx in voxel_grid_damaged.get_voxels()])
    
    nominal_pcd.points = o3d.utility.Vector3dVector(downsampled_nominal)
    damaged_pcd.points = o3d.utility.Vector3dVector(downsampled_damaged)
    
    # **PCA-Based Initial Alignment**
    if use_pca:
        print("PCA Running")
        def align_pca(source, target):
            pca_source = PCA(n_components=3).fit(np.asarray(source.points))
            pca_target = PCA(n_components=3).fit(np.asarray(target.points))
            
            R = pca_target.components_.T @ pca_source.components_
            source.rotate(R, center=np.mean(np.asarray(source.points), axis=0))
        
        align_pca(nominal_pcd, damaged_pcd)
        print("PCA Complete!")

        # Visualize
        o3d.visualization.draw_geometries([nominal_pcd, damaged_pcd], window_name="Models Alignment")

    # **Cross-Section Alignment (Experimental)**
    if use_cross_section:
        print("Cross-Section Alignement Running")
        def slice_and_align(source, target, axis=2, slices=10):
            source_points = np.asarray(source.points)
            target_points = np.asarray(target.points)
            
            min_val, max_val = np.min(source_points[:, axis]), np.max(source_points[:, axis])
            step = (max_val - min_val) / slices
            
            for i in range(slices):
                z_min, z_max = min_val + i * step, min_val + (i + 1) * step
                
                source_slice = source_points[(source_points[:, axis] >= z_min) & (source_points[:, axis] < z_max)]
                target_slice = target_points[(target_points[:, axis] >= z_min) & (target_points[:, axis] < z_max)]
                
                if len(source_slice) > 0 and len(target_slice) > 0:
                    center_source = np.mean(source_slice, axis=0)
                    center_target = np.mean(target_slice, axis=0)
                    translation = center_target - center_source
                    source.translate(translation)
        
        slice_and_align(nominal_pcd, damaged_pcd)
        print("Cross-Section Alignement Complete!")

        # Visualize
        o3d.visualization.draw_geometries([nominal_pcd, damaged_pcd], window_name="Models Alignment")

    # **Feature-Based Registration**
    if use_feature_matching:
        print("Feature Matching Running")
        nominal_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
        damaged_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

        fpfh_nominal = o3d.pipelines.registration.compute_fpfh_feature(
            nominal_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=100))
        fpfh_damaged = o3d.pipelines.registration.compute_fpfh_feature(
            damaged_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=100))
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            nominal_pcd, damaged_pcd, fpfh_nominal, fpfh_damaged,
            mutual_filter=True, max_correspondence_distance=5.0,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=4, criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999))
        
        nominal_pcd.transform(result.transformation)
        print("Feature Matching Complete!")

        # Visualize
        o3d.visualization.draw_geometries([nominal_pcd, damaged_pcd], window_name="Models Alignment")

    # **Filter out the missing region before ICP**
    nominal_tree = cKDTree(np.asarray(nominal_pcd.points))
    damaged_tree = cKDTree(np.asarray(damaged_pcd.points))
    
    # distances, indices = damaged_tree.query(np.asarray(nominal_pcd.points), k=1)
    # filtered_indices = np.where(distances < threshold)[0]
    # nominal_pcd = nominal_pcd.select_by_index(filtered_indices)
    
    # **ICP Refinement with filtered points**
    if use_icp:
        print("ICP Running")
        icp_result = o3d.pipelines.registration.registration_icp(
            nominal_pcd, damaged_pcd, 0.5, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        nominal_pcd.transform(icp_result.transformation)
        print("ICP Completed!")

        # Visualize
        o3d.visualization.draw_geometries([nominal_pcd, damaged_pcd], window_name="Models Alignment")

    # Convert to numpy arrays
    nominal_points = np.asarray(nominal_pcd.points)
    damaged_points = np.asarray(damaged_pcd.points)
    
    # Find missing points in damaged model
    missing_in_damaged = [i for i, point in enumerate(nominal_points)
                           if damaged_tree.query(point, k)[1] < len(damaged_points) and 
                              np.linalg.norm(point - damaged_points[damaged_tree.query(point, k)[1]]) > threshold]
    
    # Find missing points in nominal model
    missing_in_nominal = [i for i, point in enumerate(damaged_points)
                           if nominal_tree.query(point, k)[1] < len(nominal_points) and 
                              np.linalg.norm(point - nominal_points[nominal_tree.query(point, k)[1]]) > threshold]
      
    # # Convert to numpy arrays
    # nominal_points = np.asarray(nominal_pcd.points)
    # damaged_points = np.asarray(damaged_pcd.points)

    # k = 1  # Nearest neighbor search
    # threshold = 0.001  # Adjust this value as needed

    # # Find missing points in damaged model
    # missing_in_damaged = [
    #     i for i, point in enumerate(nominal_points)
    #     if damaged_tree.query(point, k=1)[0] > 1.5  # Use distance, not index
    # ]

    # # Find missing points in nominal model
    # missing_in_nominal = [
    #     i for i, point in enumerate(damaged_points)
    #     if nominal_tree.query(point, k=1)[0] > 1.5  # Use distance, not index
    # ]

    # # Create missing points PCD
    # if missing_in_damaged:
    #     damage_surface_points = nominal_points[np.array(missing_in_damaged)]
    #     damage_pcd = o3d.geometry.PointCloud()
    #     damage_pcd.points = o3d.utility.Vector3dVector(damage_surface_points)
    #     # o3d.visualization.draw_geometries([damage_pcd], window_name="Damage Volume")
       
    #     # Cluster missing points using DBSCAN
    #     labels = np.array(damage_pcd.cluster_dbscan(eps=26, min_points=200, print_progress=False))
    #     print("DBSCAN Labels:", labels)

    #     # Ensure at least one valid cluster exists
    #     if len(labels) > 0 and np.any(labels != -1):  # Exclude noise points (-1)
    #         largest_cluster = max(set(labels) - {-1}, key=list(labels).count)  # Ignore noise (-1)
    #         damage_pcd = damage_pcd.select_by_index(np.where(labels == largest_cluster)[0])
    #         o3d.visualization.draw_geometries([damage_pcd], window_name="Damage Volume")
    #     else:
    #         print("Warning: No valid clusters found. Saving all missing points.")

    # else:
    #     print("No missing points found in damaged model.")

    # if missing_in_nominal:
    #     damage_volume_points = damaged_points[np.array(missing_in_nominal)]
    #     damage_volume_pcd = o3d.geometry.PointCloud()
    #     damage_volume_pcd.points = o3d.utility.Vector3dVector(damage_volume_points)
    #     # o3d.visualization.draw_geometries([damage_volume_pcd], window_name="Damage Surface")

    #     # Cluster missing points using DBSCAN
    #     labels = np.array(damage_volume_pcd.cluster_dbscan(eps=26.2, min_points=200, print_progress=False))
    #     print("DBSCAN Labels:", labels)

    #     # Ensure at least one valid cluster exists
    #     if len(labels) > 0 and np.any(labels != -1):  # Exclude noise points (-1)
    #         largest_cluster = max(set(labels) - {-1}, key=list(labels).count)  # Ignore noise (-1)
    #         damage_volume_pcd = damage_volume_pcd.select_by_index(np.where(labels == largest_cluster)[0])
    #         o3d.visualization.draw_geometries([damage_volume_pcd], window_name="Damage Surface")
    #     else:
    #         print("Warning: No valid clusters found. Saving all missing points.")
    # else:
    #     print("No missing points found in nominal model.")

    # o3d.visualization.draw_geometries([damage_pcd, damage_volume_pcd], window_name="Damage Volume")

     
    non_planar_surface_points = np.vstack((damaged_points[missing_in_nominal]))
    non_planar_surface = o3d.geometry.PointCloud()
    non_planar_surface.points = o3d.utility.Vector3dVector(non_planar_surface_points)
    non_planar_surface.paint_uniform_color([1, 0, 0])
    
    # Cluster missing points using DBSCAN
    labels_non_plar_surface = np.array(non_planar_surface.cluster_dbscan(eps=20, min_points=10, print_progress=False)) #23.5
    print("DBSCAN Labels:", labels_non_plar_surface)

    # Ensure at least one valid cluster exists
    if len(labels_non_plar_surface) > 0 and np.any(labels_non_plar_surface != -1):  # Exclude noise points (-1)
        largest_cluster = max(set(labels_non_plar_surface) - {-1}, key=list(labels_non_plar_surface).count)  # Ignore noise (-1)
        non_planar_surface = non_planar_surface.select_by_index(np.where(labels_non_plar_surface == largest_cluster)[0])
    else:
        print("Warning: No valid clusters found. Saving all missing points.")
    o3d.visualization.draw_geometries([non_planar_surface], window_name="Non planar Surface")

    # case_name = os.path.basename(nominal_path).split('.')[0]
    # non_planar_surface_filename = f"{case_name}_non_planar_surface.pcd"
    # o3d.io.write_point_cloud(non_planar_surface_filename, non_planar_surface)
    
    
    # Combine missing points from both models
    missing_volume_points = np.vstack((nominal_points[missing_in_damaged], damaged_points[missing_in_nominal]))

    almostvolume_points = np.vstack((nominal_points[missing_in_damaged]))
    almostvolume_pcd = o3d.geometry.PointCloud()
    almostvolume_pcd.points = o3d.utility.Vector3dVector(almostvolume_points)
    almostvolume_pcd.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([almostvolume_pcd], window_name="Almost Volume")
    
   
    missing_pcd = non_planar_surface + almostvolume_pcd
    o3d.visualization.draw_geometries([missing_pcd], window_name="Missing Volume")

    # Create Open3D PointCloud
    missing_pcd = o3d.geometry.PointCloud()
    missing_pcd.points = o3d.utility.Vector3dVector(missing_volume_points)

    # Cluster missing points using DBSCAN
    labels = np.array(missing_pcd.cluster_dbscan(eps=20, min_points=200, print_progress=False)) #23.5
    print("DBSCAN Labels:", labels)

    # Ensure at least one valid cluster exists
    if len(labels) > 0 and np.any(labels != -1):  # Exclude noise points (-1)
        largest_cluster = max(set(labels) - {-1}, key=list(labels).count)  # Ignore noise (-1)
        missing_pcd = missing_pcd.select_by_index(np.where(labels == largest_cluster)[0])
    else:
        print("Warning: No valid clusters found. Saving all missing points.")

    # Save missing volume as PCD
    case_name = os.path.basename(nominal_path).split('.')[0]
    missing_volume_filename = f"{case_name}_mv.pcd"
    # o3d.io.write_point_cloud(missing_volume_filename, missing_pcd)

    print(f"Missing volume saved as {missing_volume_filename}")

    # Final visualization of aligned models
    print("Visualizing aligned models...")
    o3d.visualization.draw_geometries([nominal_pcd.paint_uniform_color([1, 0.706, 0]),
                                       damaged_pcd.paint_uniform_color([0, 0.651, 0.929])],
                                      window_name="Aligned Models")

    # Visualize missing region again with both models
    print("Visualizing missing region with models...")
    o3d.visualization.draw_geometries([damaged_pcd.paint_uniform_color([0, 0.651, 0.929]),
                                       missing_pcd.paint_uniform_color([1, 0, 0])],
                                      window_name="Missing Region on Damaged Model")

    # Visualize missing region again with both models
    print("Visualizing missing region with models...")
    o3d.visualization.draw_geometries([missing_pcd],
                                      window_name="Missing Region on Damaged Model")
    

    # # Visualize
    # o3d.visualization.draw_geometries([nominal_pcd, damaged_pcd], window_name="Models Alignment")
    

    # Return the missing region and damaged model
    return missing_pcd, damaged_pcd

if __name__ == "__main__":

    print("Extracting missing region from damaged part...")
    missing_region, damaged_part = extract_missing_region(
        "/home/miguel-cruz/Documents/Dataset/Case1.STL", 
        "/home/miguel-cruz/Documents/Dataset/Case1 D.STL", 1, 1.5, 1, False, False, False, True)
    
    # o3d.visualization.draw_geometries([missing_region], window_name="Missing Region")

    print("Missing region extraction complete.")