import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree, distance
from sklearn.decomposition import PCA
import os
import re
import glob
import matplotlib



def process_pair(nominal_path, damaged_path,
                 align_methods=("pca","feature",""),
                 vis_flags=("align","surface","volume"),
                 export_flags=("surface", "volume"),
                 voxel_size=1, dbscan_eps=20, dbscan_min=10, threshold=2.5, k=1, system=False):
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
    nom_pcd = nom.sample_points_uniformly(200000)
    dmg_pcd = dmg.sample_points_uniformly(200000)
    dmg_pcd.paint_uniform_color([0, 0, 1])
    nom_pcd.paint_uniform_color([0, 1, 0.2])

    o3d.visualization.draw_geometries([nom_pcd], window_name="Nom Cloud")
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    o3d.visualization.draw_geometries([dmg_pcd, axis], window_name="Dmg Cloud")

    # Optional: Apply manual transform to damaged cloud
    dmg_pcd = manual_perturbation(
        dmg_pcd,
        rotation_deg=(22.5, 44.5, -30),     # Rotate 10 degrees around X axis
        translation=(125, 5, 35)        # Move 5mm on X axis
    )
    # translation part 1 (28, 12, 39)
    #part 3 rotation 50, 45, -30
    # rotation for part 2 is (2, 1, 1.5) and translation is (74.5, 18.5, 59.5)

    show_aligned(nom_pcd, dmg_pcd, title="Initial Clouds")

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

        def slice_and_align(source, target, axis=2, slices=40):
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
        #  Alignment
        print("ICP Alignment")

        icp_result = o3d.pipelines.registration.registration_icp(
            nom_pcd, dmg_pcd, 5.5, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

        print("Alignement Result:", icp_result.fitness)
        nom_pcd.transform(icp_result.transformation)


        print("ICP Completed!")

    # Visualize aligned clouds
    if "align" in vis_flags:
        show_aligned(nom_pcd, dmg_pcd, title="Aligned Clouds")

    
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

    print(f"Points missing in damaged model: {len(missing_in_damaged)}")
    print(f"Points missing in nominal model: {len(missing_in_nominal)}")
    
    non_planar_surface_points = np.vstack((damaged_points[missing_in_nominal]))
    non_planar_surface = o3d.geometry.PointCloud()
    non_planar_surface.points = o3d.utility.Vector3dVector(non_planar_surface_points)
    non_planar_surface.paint_uniform_color([1, 0, 0])

    

    # DBSCAN clustering to define main region

    # print("DBSCAN Clustering")

    labels_non_plar_surface = np.array(non_planar_surface.cluster_dbscan(dbscan_eps, dbscan_min, print_progress=False)) #23.5
    print("DBSCAN Labels:", labels_non_plar_surface)


    assert len(labels_non_plar_surface) == len(non_planar_surface.points), \
    "DBSCAN labels and point cloud size mismatch!"
    
    # # Ensure at least one valid cluster exists
    # if len(labels_non_plar_surface) > 0 and np.any(labels_non_plar_surface != -1):  # Exclude noise points (-1)
    #     largest_cluster = max(set(labels_non_plar_surface) - {-1}, key=list(labels_non_plar_surface).count)  # Ignore noise (-1)
    #     non_planar_surface = non_planar_surface.select_by_index(np.where(labels_non_plar_surface == largest_cluster)[0])
    #     # print("DBSCAN Clustering Complete!")
    # else:
    #     print("Warning: No valid clusters found. Saving all missing points.")

    # Keep all clusters except noise (-1)
    valid_indices = np.where(labels_non_plar_surface != -1)[0]

    if system:

        clusters_idx = np.unique(labels_non_plar_surface[valid_indices])
        objects_idx = clusters_idx[clusters_idx != -1]  # Remove noise label if present

        # Generate a color map for the clusters
        colormap = matplotlib.colormaps.get_cmap('tab10')

        # Color points directly instead of rebuilding the cloud
        colors = np.zeros((len(labels_non_plar_surface), 3))

        for object_idx in objects_idx:
            color = colormap(object_idx % len(objects_idx))[0:3]
            colors[labels_non_plar_surface == object_idx] = color

        non_planar_surface.colors = o3d.utility.Vector3dVector(colors)

                
        # -----------------------------------------
        # Interactive Picking (if enabled)
        # -----------------------------------------

        picked_points = []

        print("")
        print("[INTERACTIVE PICKING]")
        print("1) Please pick points that are NOT damage using [shift + left click]")
        print("   (e.g., noise, false positives)")
        print("2) Press [shift + right click] to undo point picking")
        print("3) After picking, press 'q' to close the window")
        
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(non_planar_surface)
        vis.run()  # user picks points
        picked_indices = vis.get_picked_points()
        vis.destroy_window()
        print("")

        # Get picked points coordinates
        picked_points = np.asarray(non_planar_surface.points)[picked_indices] if picked_indices else np.array([])
        print(f"[INFO] Picked {len(picked_points)} points as non-damage")

        # If the user picked points, map each picked coordinate back to the original
        # non_planar_surface indices (using a KDTree) to find its DBSCAN cluster label.
        # Then remove all points that belong to any of the selected clusters.
        if len(picked_points) > 0:
            orig_pts = np.asarray(non_planar_surface.points)
            tree_orig = cKDTree(orig_pts)

            picked_cluster_labels = set()

            for p in picked_points:
                _, idx = tree_orig.query(p)
                lbl = labels_non_plar_surface[idx]
                if lbl != -1:
                    picked_cluster_labels.add(int(lbl))

            if picked_cluster_labels:
                mask = ~np.isin(labels_non_plar_surface, list(picked_cluster_labels))
                non_planar_surface = non_planar_surface.select_by_index(
                    np.where(mask)[0]
                )
                labels_non_plar_surface = labels_non_plar_surface[mask]
                

                print(f"[INFO] Removed {np.sum(~mask)} points from clusters {sorted(picked_cluster_labels)}")

                # -------------------------------------------------
                # REMOVE DBSCAN NOISE (-1) — ALWAYS
                # -------------------------------------------------
                noise_mask = labels_non_plar_surface != -1

                removed_noise = np.sum(~noise_mask)

                non_planar_surface = non_planar_surface.select_by_index(
                    np.where(noise_mask)[0]
                )
                labels_non_plar_surface = labels_non_plar_surface[noise_mask]

                print(f"[INFO] Removed {removed_noise} DBSCAN noise points")

            else:
                print("[INFO] Picked points were noise only. Nothing removed.")

        else:
            if len(valid_indices) > 0:
                non_planar_surface = non_planar_surface.select_by_index(valid_indices)
            else:
                print("Warning: No valid clusters found. Saving all missing points.")

    
    # Create point cloud for the non-planar surface
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(np.asarray(non_planar_surface.points))
    surface_pcd.paint_uniform_color([1, 0, 0])  # Red color for missing surface

    try: 
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

    except Exception as e:
        print(f"Error creating missing volume: {e}")    

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

    # # Create coordinate frames at origin and at the center of damaged model
    # axis_nom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    # axis_dmg = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    # axis_dmg.translate(damaged.get_center())

    # o3d.visualization.draw_geometries(
    #     [nominal, damaged, axis_nom, axis_dmg],
    #     window_name=title
    # )

def show(pcd, title="Point Cloud"):
    """
    Visualize a single point cloud.
    """
    o3d.visualization.draw_geometries([pcd], window_name=title)

def save_labeled_point_cloud(dmg_pcd, damage_pcd, full_id, out_dir, match_radius=1.0):
    """
    Save the damaged point cloud with labels: 1 for damaged points, 0 for normal points.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create KDTree for damaged region
    tree = cKDTree(np.asarray(damage_pcd.points))
    pts = np.asarray(dmg_pcd.points)
    nrms = np.asarray(dmg_pcd.normals)

    # Label: 1 if point is close to damage region, else 0
    _, idx = tree.query(pts, distance_upper_bound=match_radius)
    labels = (idx != len(damage_pcd.points)).astype(np.int32)

    # Save as .txt
    np.savetxt(os.path.join(out_dir, f"{full_id}.txt"),
               np.hstack([pts, nrms, labels[:, None]]),
               fmt="%.6f")

    # Save as .npz
    np.savez(os.path.join(out_dir, f"{full_id}.npz"),
             points=pts, normals=nrms, labels=labels)

    print(f"Saved labeled point cloud for {full_id} to {out_dir}")

def partition_labeled_point_cloud_spatially(dmg_pcd, labels, full_id, out_dir, grid_size=10, partition_size=2048, visualize=False):
    """
    Spatially partitions the point cloud into 3D grid cells and saves each cell (if it has points).
    Each saved partition has 2048 points, padded if necessary.
    """

    # Color the point cloud based on labels
    colors = np.zeros_like(np.asarray(dmg_pcd.points))  # Default color
    colors[labels == 1] = [1, 0, 0]  # Red for label 1
    # colors[labels == 0] = [0, 1, 0.2]  # Green for label 0
    colors[labels == 0] = [0.6, 0.6, 0.6]  # Green for label 0
    dmg_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the labeled point cloud
    o3d.visualization.draw_geometries([dmg_pcd], window_name="Labeled Point Cloud")

    os.makedirs(out_dir, exist_ok=True)

    points = np.asarray(dmg_pcd.points)
    normals = np.asarray(dmg_pcd.normals)

    # Get spatial bounds
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    grid_step = (max_bound - min_bound) / grid_size  # (dx, dy, dz)

    partition_id = 0

    for xi in range(grid_size):
        for yi in range(grid_size):
            for zi in range(grid_size):
                # Compute bounds for this grid cell
                min_corner = min_bound + [xi, yi, zi] * grid_step
                max_corner = min_corner + grid_step

                # Find points inside this box
                mask = np.all((points >= min_corner) & (points < max_corner), axis=1)
                cell_points = points[mask]
                cell_normals = normals[mask]
                cell_labels = labels[mask]

                if len(cell_points) == 0:
                    continue  # Skip empty boxes

                # Downsample or pad to partition_size
                if len(cell_points) >= partition_size:
                    idx = np.random.choice(len(cell_points), partition_size, replace=False)
                else:
                    idx = np.random.choice(len(cell_points), partition_size, replace=True)

                pts_part = cell_points[idx]
                nrms_part = cell_normals[idx]
                lbls_part = cell_labels[idx]

                # Set label flag
                damage_flag = "01" if np.any(lbls_part == 1) else "00"
                partition_id_str = f"{partition_id:04d}"

                filename = f"{full_id}{partition_id_str}{damage_flag}"

                print("Number of partition points:", pts_part.shape[0])

                assert pts_part.shape[0] == 2048, "Partition does not have 2048 points!"

                np.savetxt(os.path.join(out_dir, f"{filename}.txt"),
                           np.hstack([pts_part, nrms_part, lbls_part[:, None]]),
                           fmt="%.6f")

                np.savez(os.path.join(out_dir, f"{filename}.npz"),
                         points=pts_part, normals=nrms_part, labels=lbls_part)

                partition_id += 1

                if visualize:
                    print(f"Visualizing partition {partition_id}...")
                    pcd_partition = o3d.geometry.PointCloud()
                    pcd_partition.points = o3d.utility.Vector3dVector(cell_points)
                    pcd_partition.normals = o3d.utility.Vector3dVector(cell_normals)

                    # Color by label
                    colors = np.zeros_like(cell_points)
                    colors[cell_labels == 1] = [1, 0, 0]  # Red for damage
                    colors[cell_labels == 0] = [0, 1, 0]  # Green for normal
                    pcd_partition.colors = o3d.utility.Vector3dVector(colors)


                    o3d.visualization.draw_geometries([pcd_partition], window_name=f"Partition {partition_id}")


    print(f"Saved {partition_id} spatial partitions for {full_id}")

def manual_perturbation(mesh, rotation_deg=(0, 0, 0), translation=(0, 0, 0)):
    """
    Applies manual rotation (in degrees) and translation to a mesh.
    rotation_deg: tuple of 3 floats (rx, ry, rz) in degrees
    translation: tuple of 3 floats (tx, ty, tz)
    """
    # Convert degrees to radians
    rotation_rad = np.radians(rotation_deg)
    
    # Rotation matrix
    R = mesh.get_rotation_matrix_from_xyz(rotation_rad)
    mesh.rotate(R, center=np.mean(np.asarray(mesh.points), axis=0))
    
    # Apply translation
    mesh.translate(translation)
    
    return mesh


def visualize_damage_on_damaged_cloud(dmg_pcd, damage_pcd, radius=2.0, title="Damaged Cloud with Damage Highlighted"):
    """
    Visualize damaged point cloud with damage points highlighted in red.
    """
    # Copy to avoid modifying original
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(np.asarray(dmg_pcd.points))
    vis_pcd.normals = o3d.utility.Vector3dVector(np.asarray(dmg_pcd.normals))

    # Default color: gray
    colors = np.tile([0.6, 0.6, 0.6], (len(vis_pcd.points), 1))

    # Build KDTree on damage points
    damage_tree = cKDTree(np.asarray(damage_pcd.points))
    pts = np.asarray(vis_pcd.points)

    # Find damaged points
    _, idx = damage_tree.query(pts, distance_upper_bound=radius)
    damaged_mask = idx != len(damage_pcd.points)

    # Color damaged points red
    colors[damaged_mask] = [1.0, 0.0, 0.0]

    vis_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries(
        [vis_pcd],
        window_name=title
    )

if __name__ == "__main__":
    print("Starting batch processing...")

    # Paths
    path = '/home/miguelcruz/Documents/AS4RCSD/Data/BC_models/CaseStudy_03'
    nominal_path = '/home/miguelcruz/Documents/AS4RCSD/Data/BC_models/CaseStudy_03/C3G_01.STL'
    output_dir = "/home/miguelcruz/Documents/AS4RCSD/Data/BC_models/CaseStudy_03/CaseStudy_Labeled"

    # Get all files like '001001.STL'
    files = sorted(glob.glob(os.path.join(path, "[0-9][0-9][0-9][0-9][0-9][0-9].STL")))

    for filepath in files:      
        filename = os.path.basename(filepath)  # e.g., 001001.STL
        full_id = os.path.splitext(filename)[0]  # e.g., 001001

        # filename = "001001.STL" # for testing
        # full_id = "001001"
        damaged_path = os.path.join(path, filename)
        
        print(f"Processing {filename}...")

        try:
            dmg_pcd, damage_points = process_pair(
                nominal_path, damaged_path,
                align_methods=("icp"),
                vis_flags=("align", "surface"),
                export_flags=("surface",),
                voxel_size=1, dbscan_eps=2.35, dbscan_min=45, threshold=0.32, k=1, system = True
            ) 
            
            # 2.5 , 20, 0.45
            visualize_damage_on_damaged_cloud(
                dmg_pcd, damage_points,
                radius=2.0,
                title=f"Damaged Cloud with Damage Highlighted: {full_id}"
            )

            save_labeled_point_cloud(
                dmg_pcd, damage_points,
                full_id=full_id,
                out_dir=output_dir,
                match_radius=2.0  #1.0
            )

            partition = input("Partition the point cloud? (y/n): ").strip().lower() == 'y'
            
            if partition:
                partition_labeled_point_cloud_spatially(
                dmg_pcd, 
                labels=np.load(os.path.join(output_dir, f"{full_id}.npz"))["labels"],
                full_id=full_id,
                out_dir=os.path.join(output_dir, "partitions"),
                grid_size=6,  # 5x5x5 = 125 boxes, adjust to get ~200
                partition_size=2048, 
                visualize=False
                )


        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

