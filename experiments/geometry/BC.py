import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree, distance
from sklearn.decomposition import PCA
import os
import re
import glob

def process_pair(nominal_path, damaged_path,
                 align_methods=("pca","feature","icp"),
                 vis_flags=("align","surface","volume"),
                 export_flags=("surface", "volume"),
                 voxel_size=1, dbscan_eps=20, dbscan_min=10, threshold=0.1, k=1):
    """
    Process one STL pair:
      - Registers (based on align_methods)
      - Extracts missing-surface and missing-volume
      - Visualizes selected stages (vis_flags)
      - Exports .pcd of damage surface/volume based on export_flags
    Returns Open3D PointClouds for surface & volume.
    """
    
    # Load STLs and sample large clouds
    nom = o3d.io.read_triangle_mesh(nominal_path).compute_vertex_normals()
    dmg = o3d.io.read_triangle_mesh(damaged_path).compute_vertex_normals()
    nom_pcd = nom.sample_points_uniformly(50000)
    dmg_pcd = dmg.sample_points_uniformly(50000)
    dmg_pcd.paint_uniform_color([0, 0, 1])

    # Optional alignments
    if "pca" in align_methods:
        # PCA alignment
        print("PCA alignment")

        def align_pca(source, target):
            pca_source = PCA(n_components=3).fit(np.asarray(source.points))
            pca_target = PCA(n_components=3).fit(np.asarray(target.points))
            R = pca_target.components_.T @ pca_source.components_
            source.rotate(R, center=np.mean(np.asarray(source.points), axis=0))
        
        align_pca(nom_pcd, dmg_pcd)

        print("PCA Complete!")


    if "cross-section" in align_methods:
        # Cross-section alignment
        print("Cross-Section Alignement")

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
        
        slice_and_align(nom_pcd, dmg_pcd)

        print("Cross-Section Alignement Complete!")
        
    if "feature" in align_methods:
        # Feature matching alignment
        print("Feature Matching Alignment")

        nom_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
        dmg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

        fpfh_nominal = o3d.pipelines.registration.compute_fpfh_feature(
            nom_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=100))
        fpfh_damaged = o3d.pipelines.registration.compute_fpfh_feature(
            dmg_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=100))
        
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            nom_pcd, dmg_pcd, fpfh_nominal, fpfh_damaged,
            mutual_filter=True, max_correspondence_distance=5.0,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=4, criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999))
        
        nom_pcd.transform(result.transformation)

        print("Feature Matching Complete!")
    
    # **Filter out the missing region before ICP**
    nom_tree = cKDTree(np.asarray(nom_pcd.points))
    dmg_tree = cKDTree(np.asarray(dmg_pcd.points))


    if "icp" in align_methods:
        # ICP Alignment
        # print("ICP Alignment")

        icp_result = o3d.pipelines.registration.registration_icp(
            nom_pcd, dmg_pcd, 0.5, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
        nom_pcd.transform(icp_result.transformation)

        # print("ICP Completed!")

    # Visualize aligned clouds
    if "align" in vis_flags:
        show_aligned(nom_pcd, dmg_pcd, window="Aligned Clouds")

    
    # Extract missing-volume and missing-surface
    # Use KDTree and distance threshold
    nominal_points = np.asarray(nom_pcd.points)
    damaged_points = np.asarray(dmg_pcd.points)
    
    # Find missing points in damaged model
    missing_in_damaged = [i for i, point in enumerate(nominal_points)
                           if dmg_tree.query(point, k)[1] < len(damaged_points) and 
                              np.linalg.norm(point - damaged_points[dmg_tree.query(point, k)[1]]) > threshold]
    
    # Find missing points in nominal model
    missing_in_nominal = [i for i, point in enumerate(damaged_points)
                           if nom_tree.query(point, k)[1] < len(nominal_points) and 
                              np.linalg.norm(point - nominal_points[nom_tree.query(point, k)[1]]) > threshold]
    
    non_planar_surface_points = np.vstack((damaged_points[missing_in_nominal]))
    non_planar_surface = o3d.geometry.PointCloud()
    non_planar_surface.points = o3d.utility.Vector3dVector(non_planar_surface_points)
    non_planar_surface.paint_uniform_color([1, 0, 0])

    # DBSCAN clustering to define main region

    # print("DBSCAN Clustering")

    labels_non_plar_surface = np.array(non_planar_surface.cluster_dbscan(eps=20, min_points=10, print_progress=False)) #23.5
    # print("DBSCAN Labels:", labels_non_plar_surface)

    # Ensure at least one valid cluster exists
    if len(labels_non_plar_surface) > 0 and np.any(labels_non_plar_surface != -1):  # Exclude noise points (-1)
        largest_cluster = max(set(labels_non_plar_surface) - {-1}, key=list(labels_non_plar_surface).count)  # Ignore noise (-1)
        non_planar_surface = non_planar_surface.select_by_index(np.where(labels_non_plar_surface == largest_cluster)[0])
        # print("DBSCAN Clustering Complete!")
    else:
        print("Warning: No valid clusters found. Saving all missing points.")
    
    # Create point cloud for the non-planar surface
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(np.asarray(non_planar_surface.points))
    surface_pcd.paint_uniform_color([1, 0, 0])  # Red color for missing surface


    # Combine missing points from both models to generate the missing volume
    missing_volume_points = np.vstack((nominal_points[missing_in_damaged], damaged_points[missing_in_nominal]))
    almostvolume_points = np.vstack((nominal_points[missing_in_damaged]))
    volume_pcd = o3d.geometry.PointCloud()
    volume_pcd.points = o3d.utility.Vector3dVector(almostvolume_points)
    volume_pcd.paint_uniform_color([0, 0, 1])
    missing_pcd = non_planar_surface + volume_pcd

    # Create Open3D PointCloud
    missing_pcd = o3d.geometry.PointCloud()
    missing_pcd.points = o3d.utility.Vector3dVector(missing_volume_points)

    # Cluster missing points using DBSCAN
    labels = np.array(missing_pcd.cluster_dbscan(eps=20, min_points=200, print_progress=False)) #23.5
    # print("DBSCAN Labels:", labels)

    # Ensure at least one valid cluster exists
    if len(labels) > 0 and np.any(labels != -1):  # Exclude noise points (-1)
        largest_cluster = max(set(labels) - {-1}, key=list(labels).count)  # Ignore noise (-1)
        missing_pcd = missing_pcd.select_by_index(np.where(labels == largest_cluster)[0])
    else:
        print("Warning: No valid clusters found. Saving all missing points.")


    # Visualizations
    if "surface" in vis_flags:
        show(surface_pcd, title="Damage Surface")
    if "volume" in vis_flags:
        show(missing_pcd, title="Damage Volume")

   
    # Exports
    basename = os.path.splitext(os.path.basename(damaged_path))[0]
  
    if "surface" in export_flags:
        # o3d.io.write_point_cloud(f"{basename}_surface.pcd", surface_pcd)
        return dmg_pcd, surface_pcd
    if "volume" in export_flags:
        # o3d.io.write_point_cloud(f"{basename}_volume.pcd", volume_pcd)
        return dmg_pcd, volume_pcd

def show_aligned(nominal, damaged, title="Aligned Models"):
    """
    Visualize aligned nominal and damaged point clouds.
    """
    o3d.visualization.draw_geometries([nominal, damaged], window_name=title)

def show(pcd, title="Point Cloud"):
    """
    Visualize a single point cloud.
    """
    o3d.visualization.draw_geometries([pcd], window_name=title)

def save_to_location_and_volume_dirs(
    dmg_pcd, damage_points,
    file_nr, location, volume,
    loc_base="cubes_loc", vol_base="cubes_vol",
    num_points=2048, match_radius=1.0
    ):

    def label_points(pc, damage_pcd, radius):
        pts = np.asarray(pc.points)
        nrm = np.asarray(pc.normals)
        dmg_pts = np.asarray(damage_pcd.points)  # <-- convert properly
        tree = cKDTree(dmg_pts)
        _, idx = tree.query(pts, distance_upper_bound=radius)
        lbl = (idx != len(dmg_pts)).astype(np.int32)
        return pts, nrm, lbl

    def save_data(out_dir, fname, pts, nrms, lbls):
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, f"{fname}.txt"),
                   np.hstack([pts, nrms, lbls[:, None]]), fmt="%.6f")
        np.savez(os.path.join(out_dir, f"{fname}.npz"),
                 points=pts, normals=nrms, labels=lbls)

    # Sample point cloud
    sampled = dmg_pcd.farthest_point_down_sample(num_points)
    points, normals, labels = label_points(sampled, damage_points, match_radius)
    fname = f"Part{file_nr}"

    # --- Location-based folder ---
    loc_map = {'C': 'corner', 'E': 'edge', 'S': 'surface'}
    loc_dir = os.path.join(loc_base, loc_map.get(location.upper(), "unknown"))
    save_data(loc_dir, fname, points, normals, labels)

    # --- Volume-based folder ---
    volume = float(str(volume).replace(" ", "").replace(",", ""))
    vol_group = "under_100" if volume <= 100 else "above_100"
    vol_dir = os.path.join(vol_base, vol_group)
    save_data(vol_dir, fname, points, normals, labels)

    print(f"Saved Part{file_nr} to:\n - {loc_dir}\n - {vol_dir}")



if __name__ == "__main__":

    print("Extracting missing region from damaged part:")
    

    path = '/home/miguel-cruz/Documents/RAN-3D/Dataset_500_DamageModels/'
    nominal_path = '/home/miguel-cruz/Documents/RAN-3D/Dataset_500_DamageModels/Part G .STL'
    file_nr = 0
    file_extension = ".STL"
    files = glob.glob(os.path.join(path, f"*{file_extension}"))
    count = len(files)
    count = count - 1

    
    while file_nr != 500:
        file_nr = file_nr + 1

        word = "Part" + str(file_nr) + " "
        for file in os.listdir(path):
            if file.startswith(word):
                filename = os.path.join(file)
        parts = filename.split()
        
        if len(parts) == 5:
            volume = parts[2] + parts[3]  # The volume is the 4th part (index 3)
            location = parts[4].split('.')[0]  # The location is the 5th part (index 4)

        elif len(parts) == 4:
            volume = parts[2] # The volume is the 4th part (index 3)
            location = parts[3].split('.')[0]  # The location is the 5th part (index 4)

        print(filename)
        damaged_path = path + filename
        
        try:
            # Check if the damaged file exists

            dmg_pcd, damage_points = process_pair(nominal_path, damaged_path, "icp", "", "surface", 1, 20, 10, 3, 1) #th = 3
            # show_aligned(dmg_pcd, damage_points, title=f"Damage Points {filename}")

            save_to_location_and_volume_dirs(
                dmg_pcd, damage_points,
                file_nr, location, volume,
                loc_base="/home/miguel-cruz/Documents/3DL-Based /data/Cubes_Loc", vol_base="/home/miguel-cruz/Documents/3DL-Based /data/Cubes_Vol",
                num_points=2048, match_radius=1.0
            )

            print("Missing region extraction complete.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue


    