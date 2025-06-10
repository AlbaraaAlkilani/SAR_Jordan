
import logging
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import geopandas as gpd
import pandas as pd
import shapely.affinity
from shapely.geometry import Point, LineString, MultiLineString, Polygon
import matplotlib.pyplot as plt


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger.
    """
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to an appropriate UTM metric CRS.
    """
    if gdf.crs is None or gdf.crs.to_string() in ["EPSG:4326", "WGS84"]:
        center = gdf.geometry.centroid.iloc[0]
        zone = int((center.x + 180) / 6) + 1
        epsg = 32600 + zone if center.y >= 0 else 32700 + zone
        logging.info(f"Reprojecting to EPSG:{epsg} for metric calculations.")
        return gdf.to_crs(epsg=epsg)
    return gdf


def compute_spacing(fov_deg: float, altitude: float, overlap: float) -> float:
    """
    Compute sweep spacing from camera FOV (deg), flight altitude (m), and overlap fraction.
    """
    fov_rad = math.radians(fov_deg)
    ground_width = 2 * altitude * math.tan(fov_rad / 2)
    spacing = ground_width * (1 - overlap)
    logging.info(
        f"Computed spacing: {spacing:.2f} m (ground width {ground_width:.2f} m, overlap {overlap*100:.0f}%%)"
    )
    return spacing


def lawnmower_path(
    poly: Polygon,
    spacing: float,
    angle: float,
    station: Optional[Tuple[float, float]] = None
) -> List[Tuple[float, float]]:
    """
    Generate an ordered list of sweep waypoints over the polygon.
    If station is given, align one sweep through its y-coordinate for continuity.
    """
    # Rotate if needed
    if angle != 0:
        poly = shapely.affinity.rotate(poly, -angle, origin='centroid')

    minx, miny, maxx, maxy = poly.bounds
    # Determine starting y
    if station:
        st_pt = Point(station)
        if angle != 0:
            st_pt = shapely.affinity.rotate(st_pt, -angle, origin=poly.centroid)
        start_y = st_pt.y
    else:
        start_y = miny + spacing / 2

    # Collect sweep line y's
    ys = []
    y = start_y
    while y >= miny:
        ys.append(y)
        y -= spacing
    y = start_y + spacing
    while y <= maxy:
        ys.append(y)
        y += spacing
    ys.sort()

    # Intersect with polygon
    segments: List[LineString] = []
    for y in ys:
        sweep = LineString([(minx, y), (maxx, y)])
        inter = poly.intersection(sweep)
        if inter.is_empty:
            continue
        if isinstance(inter, LineString):
            segments.append(inter)
        elif isinstance(inter, MultiLineString):
            segments.extend(inter.geoms)

    # Build waypoint sequence
    waypoints: List[Tuple[float, float]] = []
    reverse = False
    for seg in segments:
        coords = list(seg.coords)
        if reverse:
            coords.reverse()
        waypoints.extend(coords)
        reverse = not reverse

    # Rotate back if needed
    if angle != 0:
        ctr = poly.centroid
        waypoints = [
            shapely.affinity.rotate(Point(x, y), angle, origin=ctr).coords[0]
            for x, y in waypoints
        ]
    return waypoints


def compute_missions(
    full_path: List[Tuple[float, float]],
    station: Tuple[float, float],
    speed: float,
    time_limit: float
) -> Tuple[List[List[Tuple[float, float]]], List[Dict]]:
    """
    Partition the master path into sequential drone missions under time_limit.
    Each mission: launch -> coverage -> return, starting where last ended.
    """
    # Cumulative distance along full path
    cum = [0.0]
    for a, b in zip(full_path[:-1], full_path[1:]):
        cum.append(cum[-1] + math.hypot(b[0]-a[0], b[1]-a[1]))

    missions: List[List[Tuple[float, float]]] = []
    infos: List[Dict] = []
    idx = 0
    drone_id = 1
    sx, sy = station

    while idx < len(full_path):
        fx, fy = full_path[idx]
        dist_out = math.hypot(fx-sx, fy-sy)
        t_out = dist_out / speed
        j_stop = idx
        # Extend until time budget exhausted
        for j in range(idx, len(full_path)):
            cov = cum[j] - cum[idx]
            t_cov = cov / speed
            rx, ry = full_path[j]
            t_ret = math.hypot(rx-sx, ry-sy) / speed
            if t_out + t_cov + t_ret <= time_limit:
                j_stop = j
            else:
                break
        # Stop if no progress
        if j_stop == idx:
            logging.warning(f"Drone {drone_id} made no forward progress; halting.")
            break
        segment = full_path[idx:j_stop+1]
        path = [station] + segment + [station]
        cov = cum[j_stop] - cum[idx]
        ret = math.hypot(full_path[j_stop][0]-sx, full_path[j_stop][1]-sy)
        total = dist_out + cov + ret
        # Record info in km/min
        infos.append({
            'DroneID': drone_id,
            'Outbound_km': dist_out/1000.0,
            'Coverage_km': cov/1000.0,
            'Return_km': ret/1000.0,
            'Total_km': total/1000.0,
            'Time_min': total/(speed*60.0)
        })
        missions.append(path)
        idx = j_stop
        drone_id += 1

    return missions, infos


def plot_all(
    missions: List[List[Tuple[float, float]]],
    infos: List[Dict],
    poly: Polygon,
    station: Tuple[float, float],
    outdir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8,6))
    # Plot area boundary
    if isinstance(poly, Polygon):
        x, y = poly.exterior.xy
        ax.plot(x, y, 'k-', lw=2)
    else:
        for part in poly.geoms:
            x, y = part.exterior.xy
            ax.plot(x, y, 'k-', lw=2)
    # Plot each mission
    colors = ['red','blue','green','orange','purple']
    for info, path in zip(infos, missions):
        label = f"Drone {info['DroneID']}"
        xs, ys = zip(*path)
        col = colors[(info['DroneID']-1) % len(colors)]
        ax.plot(xs, ys, label=label, color=col)
        ax.scatter(xs, ys, s=10, color=col)
    ax.scatter(*station, s=100, marker='*', color='red', label='Launch')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('All Drone Routes')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(outdir/'all_routes.png', bbox_inches='tight')
    plt.show()


def plot_individual(
    missions: List[List[Tuple[float, float]]],
    infos: List[Dict],
    outdir: Path
) -> None:
    for info, path in zip(infos, missions):
        i = info['DroneID']
        fig, ax = plt.subplots(figsize=(8,6))
        xs, ys = zip(*path)
        ax.plot(xs, ys, marker='o')
        ax.scatter(xs[0], ys[0], s=100, marker='*', color='red', label='Launch')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f"Drone {i} Mission Details")
        # Info text
        txt = (
            f"Drone {i}\n"
            f"Outbound: {info['Outbound_km']:.2f} km\n"
            f"Coverage: {info['Coverage_km']:.2f} km\n"
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


def export_csv(
    missions: List[List[Tuple[float, float]]],
    infos: List[Dict],
    outdir: Path
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Summary and individual paths
    pd.DataFrame(infos).to_csv(outdir/'time_mode_summary.csv', index=False)
    for info, path in zip(infos, missions):
        rid = info['DroneID']
        pd.DataFrame(path, columns=['x','y']).to_csv(outdir/f'drone_{rid}_path.csv', index=False)
    print(f"Exports written to {outdir}")


def main() -> None:
    configure_logging()
    geojson = input("GeoJSON file path: ").strip()
    gdf = gpd.read_file(geojson)
    gdf = to_metric(gdf)
    poly = gdf.geometry.iloc[0]

    if input("Compute spacing from FOV? (y/n): ").strip().lower() == 'y':
        fov = float(input("Camera FOV (deg): "))
        alt = float(input("Flight altitude (m): "))
        ov = float(input("Overlap fraction (0-1): "))
        spacing = compute_spacing(fov, alt, ov)
    else:
        spacing = float(input("Fixed spacing between sweeps (m): "))
    angle = float(input("Rotation angle for sweeps (deg): "))
    time_limit = float(input("Time limit per drone (s): "))
    speed = float(input("Drone speed (m/s): "))

    print("Launch station options: 1) Centroid  2) Custom  3) Edge")
    ch = input("Option: ").strip()
    if ch == '1':
        station = (poly.centroid.x, poly.centroid.y)
    elif ch == '2':
        x = float(input("Custom launch X: "))
        y = float(input("Custom launch Y: "))
        station = (x, y)
    elif ch == '3':
        e = input("Edge (top/bottom/left/right): ").strip().lower()
        xmin, ymin, xmax, ymax = poly.bounds
        station = {
            'top':((xmin+xmax)/2, ymax),
            'bottom':((xmin+xmax)/2, ymin),
            'left':(xmin, (ymin+ymax)/2),
            'right':(xmax, (ymin+ymax)/2)
        }.get(e, (poly.centroid.x, poly.centroid.y))
    else:
        station = (poly.centroid.x, poly.centroid.y)

    outdir = Path("output")
    # ensure output directory exists before saving figures
    outdir.mkdir(parents=True, exist_ok=True)
    full_path = lawnmower_path(poly, spacing, angle, station)
    missions, infos = compute_missions(full_path, station, speed, time_limit)

    # Try merging last two if possible
    if len(infos) >= 2:
        limit_min = time_limit / 60.0
        t1 = infos[-2]['Time_min']
        t2 = infos[-1]['Time_min']
        if t1 + t2 <= limit_min:
            print(f"Merging Drone {infos[-2]['DroneID']} and {infos[-1]['DroneID']}")
            merged = missions[-2][:-1] + missions[-1][1:]
            total = sum(math.hypot(x2-x1, y2-y1)
                        for (x1,y1),(x2,y2) in zip(merged[:-1], merged[1:]))
            dout = math.hypot(merged[1][0]-station[0], merged[1][1]-station[1])
            dret = math.hypot(merged[-2][0]-station[0], merged[-2][1]-station[1])
            dcov = total - dout - dret
            infos[-2] = {
                'DroneID': infos[-2]['DroneID'],
                'Outbound_km': dout/1000.0,
                'Coverage_km': dcov/1000.0,
                'Return_km': dret/1000.0,
                'Total_km': total/1000.0,
                'Time_min': total/(speed*60.0)
            }
            missions[-2] = merged
            missions.pop()
            infos.pop()

    plot_all(missions, infos, poly, station, outdir)
    plot_individual(missions, infos, outdir)
    export_csv(missions, infos, outdir)

    # Full coverage summary
    area_km2 = poly.area / 1e6
    cov_km = sum(info['Coverage_km'] for info in infos)
    cov_area = cov_km * (spacing / 1000.0)
    pct = (cov_area / area_km2 * 100) if area_km2 > 0 else 0
    print(f"Full coverage: {cov_area:.2f} km² ({pct:.1f}% of polygon)")


if __name__ == '__main__':
    main()
