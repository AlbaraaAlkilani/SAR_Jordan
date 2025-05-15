# Libraries Loading
import logging
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import geopandas as gpd
import pandas as pd
import shapely.affinity
from shapely.geometry import Point, LineString, MultiLineString, Polygon
import matplotlib.pyplot as plt

# AOI LOADING

def load_and_describe_aoi() -> Polygon:
    """
    Prompt for GeoJSON path, load AOI, reproject to metric CRS,
    display area & perimeter, and plot boundary.
    """
    geojson = input("Enter path to your GeoJSON AOI file: ").strip()
    gdf = gpd.read_file(geojson)
    if gdf.crs is None or gdf.crs.to_string() in ["EPSG:4326", "WGS84"]:
        center = gdf.geometry.centroid.iloc[0]
        zone = int((center.x + 180) / 6) + 1
        epsg = 32600 + zone if center.y >= 0 else 32700 + zone
        logging.info(f"Reprojecting to EPSG:{epsg}")
        gdf = gdf.to_crs(epsg=epsg)
    poly = gdf.geometry.unary_union
    # Compute metrics
    area_km2 = poly.area / 1e6
    peri_km = poly.length / 1000
    print(f"AOI Area: {area_km2:.3f} km²")
    print(f"AOI Perimeter: {peri_km:.3f} km")
    # Plot boundary
    fig, ax = plt.subplots(figsize=(6,6))
    if isinstance(poly, Polygon):
        x, y = poly.exterior.xy
        ax.plot(x, y, 'blue', lw=2)
    else:
        for part in poly.geoms:
            x, y = part.exterior.xy
            ax.plot(x, y, 'blue', lw=2)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.show()
    return poly
poly = load_and_describe_aoi()

#Drone Specifications
def get_drone_specs() -> Tuple[float, float, float, float]:
    """
    Prompt for spacing (computed or fixed), sweep rotation angle,
    per-drone time limit (s), and speed (m/s).
    """
    if input("Compute spacing from FOV? (y/n): ").strip().lower() == 'y':
        fov = float(input("Camera FOV (deg): "))
        alt = float(input("Flight altitude (m): "))
        ov = float(input("Overlap fraction (0-1): "))
        fov_rad = math.radians(fov)
        width = 2 * alt * math.tan(fov_rad / 2)
        spacing = width * (1 - ov)
        print(f"Computed spacing: {spacing:.2f} m")
    else:
        spacing = float(input("Fixed sweep spacing (m): "))
    angle = float(input("Rotation angle for sweeps (deg): "))
    time_limit = float(input("Time limit per drone (s): "))
    speed = float(input("Drone speed (m/s): "))
    return spacing, angle, time_limit, speed
spacing, angle, time_limit, speed = get_drone_specs()

#Launch Station Selection
def choose_launch_station(poly: Polygon) -> Tuple[float, float]:
    """
    Options: centroid, custom coords, edge midpoint.
    """
    print("Launch station options: 1) Centroid 2) Custom 3) Edge")
    ch = input("Option: ").strip()
    xmin, ymin, xmax, ymax = poly.bounds
    if ch == '1':
        return (poly.centroid.x, poly.centroid.y)
    if ch == '2':
        x = float(input("Custom launch X: "))
        y = float(input("Custom launch Y: "))
        return (x, y)
    if ch == '3':
        e = input("Edge (top/bottom/left/right): ").strip().lower()
        if e == 'top':    return ((xmin+xmax)/2, ymax)
        if e == 'bottom': return ((xmin+xmax)/2, ymin)
        if e == 'left':   return (xmin, (ymin+ymax)/2)
        if e == 'right':  return (xmax, (ymin+ymax)/2)
    print("Invalid, defaulting to centroid.")
    return (poly.centroid.x, poly.centroid.y)
station = choose_launch_station(poly)
print(station)

#Overall Routes & CSV Export
def plan_and_export_all(
    poly: Polygon,
    station: Tuple[float, float],
    spacing: float,
    angle: float,
    speed: float,
    time_limit: float,
    outdir: Path
) -> Tuple[List[List[Tuple[float, float]]], List[Dict]]:
    """
    Generate full coverage path, split into missions,
    merge last two if possible, plot combined, export CSV.
    """
    # Master path
    full_path = lawnmower_path(poly, spacing, angle, station)
    missions, infos = compute_missions(full_path, station, speed, time_limit)
    # Merge last two
    if len(infos) >= 2:
        lim_min = time_limit/60.0
        if infos[-2]['Time_min'] + infos[-1]['Time_min'] <= lim_min:
            print(f"Merging drones {infos[-2]['DroneID']} & {infos[-1]['DroneID']}")
            merged = missions[-2][:-1] + missions[-1][1:]
            total = sum(math.hypot(x2-x1, y2-y1)
                        for (x1,y1),(x2,y2) in zip(merged[:-1], merged[1:]))
            dout = math.hypot(merged[1][0]-station[0], merged[1][1]-station[1])
            dret = math.hypot(merged[-2][0]-station[0], merged[-2][1]-station[1])
            dcov = total - dout - dret
            infos[-2] = {
                'DroneID': infos[-2]['DroneID'],
                'Outbound_km': dout/1000,
                'Coverage_km': dcov/1000,
                'Return_km': dret/1000,
                'Total_km': total/1000,
                'Time_min': total/(speed*60)
            }
            missions[-2] = merged
            missions.pop()
            infos.pop()
    # Add effective area to each info
    for info in infos:
        info['Area_km2'] = info['Coverage_km'] * (spacing / 1000.0)
    # Prepare output folder
    outdir.mkdir(parents=True, exist_ok=True)
    # Plot combined
    plot_all(missions, infos, poly, station, outdir)
    # Export CSV
    pd.DataFrame(infos).to_csv(outdir/'time_mode_summary.csv', index=False)
    print(f"Summary CSV at {outdir/'time_mode_summary.csv'}")
    return missions, infos
outdir = Path("output")
missions, infos = plan_and_export_all(
        poly, station, spacing, angle, speed, time_limit, outdir    )

# Individual Drone Plots

def export_individual_plots(
    missions: List[List[Tuple[float, float]]],
    infos: List[Dict],
    outdir: Path
) -> None:
    """
    Generate and save figure per drone mission.
    """
    for info, path in zip(infos, missions):
        i = info['DroneID']
        fig, ax = plt.subplots(figsize=(8,6))
        xs, ys = zip(*path)
        ax.plot(xs, ys, marker='o')
        ax.scatter(xs[0], ys[0], s=100, marker='*', color='red', label='Launch')
        ax.set_title(f"Drone {i} Details")
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        txt = (f"Drone {i}\n"
                f"Outbound: {info['Outbound_km']:.2f} km\n"
                f"Coverage: {info['Coverage_km']:.2f} km\n"
                f"Area: {info['Area_km2']:.3f} km²\n"
                f"Return: {info['Return_km']:.2f} km\n"
                f"Total: {info['Total_km']:.2f} km\n"
                f"Time: {info['Time_min']:.1f} min"
        )
        ax.text(1.02, 0.5, txt, transform=ax.transAxes,
                verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        fig.tight_layout(rect=[0,0,0.85,1])
        fig.savefig(outdir/f'drone_{i}_mission.png', bbox_inches='tight')
        plt.show()
export_individual_plots(missions, infos, outdir)

