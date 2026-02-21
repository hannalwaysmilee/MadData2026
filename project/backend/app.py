
from pydantic import BaseModel
import math
import os
import requests
from fastapi import HTTPException
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import math
from functools import lru_cache
import networkx as nx
import osmnx as ox
from pyproj import Transformer


app = FastAPI(title="Low-Stress Bike Routing API")

@app.get("/health")
def health():
    return {"ok": True}

class Point(BaseModel):
    lat: float
    lng: float

class RouteRequest(BaseModel):
    start: Point
    end: Point
    mode: str = "shortest"


# --- LTS graph builder (from local GeoJSON) ---

def _snap_xy(x: float, y: float, grid_m: float = 2.0) -> tuple[float, float]:
    return (round(x / grid_m) * grid_m, round(y / grid_m) * grid_m)

def _edge_cost(length_m: float, lts: int, beta: float) -> float:
    # 低压力：更讨厌高 LTS
    return float(length_m) * (1.0 + beta * (max(1, min(4, int(lts))) - 1))

@lru_cache(maxsize=1)
def get_lts_graph(beta: float = 2.0) -> nx.Graph:
    path = "/data/bike_lts.geojson"
    try:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Missing {path}. Put bike_lts.geojson into ./data and mount to /data.")

    # CRS84 -> 先投影到 UTM 16N（Madison 大体在 zone 16）
    # 如果你们后面要更严谨，可自动挑 zone；但这对 demo 足够稳。
    tf = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)

    G = nx.Graph()

    feats = gj.get("features", [])
    for feat in feats:
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        if geom.get("type") != "LineString":
            continue

        coords = geom.get("coordinates", [])
        if not coords or len(coords) < 2:
            continue

        # LTS 字段：优先用 LTS，其次 LTS_F
        lts = props.get("LTS", props.get("LTS_F", None))
        if lts is None:
            continue
        try:
            lts_i = int(lts)
        except Exception:
            continue
        lts_i = max(1, min(4, lts_i))

        # 线段拆成“相邻点”边，连通性更好
        for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
            x1, y1 = tf.transform(lon1, lat1)
            x2, y2 = tf.transform(lon2, lat2)

            x1, y1 = _snap_xy(x1, y1, grid_m=2.0)
            x2, y2 = _snap_xy(x2, y2, grid_m=2.0)

            if (x1, y1) == (x2, y2):
                continue

            length_m = math.hypot(x2 - x1, y2 - y1)
            if length_m <= 0.01:
                continue

            cost = _edge_cost(length_m, lts_i, beta)

            # 多条边重复时取更“便宜”的（更短或更低 LTS）
            if G.has_edge((x1, y1), (x2, y2)):
                if cost < G[(x1, y1)][(x2, y2)]["cost"]:
                    G[(x1, y1)][(x2, y2)].update(length_m=length_m, lts=lts_i, cost=cost)
            else:
                G.add_edge((x1, y1), (x2, y2), length_m=length_m, lts=lts_i, cost=cost)

    if G.number_of_edges() == 0:
        raise RuntimeError("LTS graph has 0 edges. Check bike_lts.geojson content/format.")

    return G

def _nearest_node_xy(G: nx.Graph, x: float, y: float) -> tuple[float, float]:
    # 简单最近邻：demo 足够（后面可用 scipy KDTree 加速）
    best = None
    best_d2 = float("inf")
    for (nx_, ny_) in G.nodes:
        d2 = (nx_ - x) ** 2 + (ny_ - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (nx_, ny_)
    return best

def _path_stats(G: nx.Graph, path_nodes: list[tuple[float, float]]) -> dict:
    dist = 0.0
    max_lts = 0
    breakdown = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        data = G[u][v]
        dist += float(data["length_m"])
        lts = int(data["lts"])
        max_lts = max(max_lts, lts)
        breakdown[lts] += float(data["length_m"])
    return {"distance_m": dist, "max_lts": max_lts, "lts_breakdown_m": breakdown}

# --- existing: shortest graph via osmnx (keep as-is) ---
@lru_cache(maxsize=2)
def get_graph_pair():
    place = "Madison, Wisconsin, USA"
    G_ll = ox.graph_from_place(place, network_type="bike", simplify=True)
    G_p = ox.project_graph(G_ll)
    return G_ll, G_p

@app.post("/api/route")
def route(req: RouteRequest):
    mode = (req.mode or "shortest").lower()

    if mode == "shortest":
        # 你现在已经跑通的版本（略）——保持你现有 shortest 实现即可
        G_ll, G_p = get_graph_pair()

        transformer = Transformer.from_crs("EPSG:4326", G_p.graph["crs"], always_xy=True)
        sx, sy = transformer.transform(req.start.lng, req.start.lat)
        ex, ey = transformer.transform(req.end.lng, req.end.lat)

        orig = ox.nearest_nodes(G_p, sx, sy)
        dest = ox.nearest_nodes(G_p, ex, ey)

        path = nx.shortest_path(G_p, orig, dest, weight="length")
        polyline = [[G_ll.nodes[n]["x"], G_ll.nodes[n]["y"]] for n in path]

        dist_m = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edges = G_p.get_edge_data(u, v)
            dist_m += float(min(attr.get("length", 0.0) for attr in edges.values()))

        return {"route_id": "shortest", "polyline": polyline, "stats": {"distance_m": dist_m, "num_nodes": len(path)}}

    if mode == "low_stress":
        # ✅ 走本地 LTS 图
        beta = 2.0  # 你们后面可以做成参数
        G = get_lts_graph(beta=beta)

        tf = Transformer.from_crs("EPSG:4326", "EPSG:26916", always_xy=True)
        sx, sy = tf.transform(req.start.lng, req.start.lat)
        ex, ey = tf.transform(req.end.lng, req.end.lat)

        start_node = _nearest_node_xy(G, *_snap_xy(sx, sy, 2.0))
        end_node   = _nearest_node_xy(G, *_snap_xy(ex, ey, 2.0))

        try:
            path_nodes = nx.shortest_path(G, start_node, end_node, weight="cost")
        except nx.NetworkXNoPath:
            raise HTTPException(status_code=400, detail="No path in LTS graph (try closer points / check coverage).")

        # 输出 polyline 用经纬度（把 UTM 16N 反投影回 WGS84）
        inv = Transformer.from_crs("EPSG:26916", "EPSG:4326", always_xy=True)
        polyline = []
        for x, y in path_nodes:
            lon, lat = inv.transform(x, y)
            polyline.append([lon, lat])

        stats = _path_stats(G, path_nodes)
        stats["beta"] = beta
        stats["num_nodes"] = len(path_nodes)

        return {"route_id": "low_stress", "polyline": polyline, "stats": stats}

    raise HTTPException(status_code=400, detail="mode must be 'shortest' or 'low_stress'")
# ---- sampling helpers ----
def _haversine_m(lat1, lon1, lat2, lon2):
    # 快速够用：用球面距离（米）
    R = 6371000.0
    import math
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _bearing_deg(lat1, lon1, lat2, lon2):
    import math
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dl)
    brng = (math.degrees(math.atan2(y, x)) + 360) % 360
    return brng

def sample_polyline(polyline_lnglat, spacing_m=100.0, max_points=20):
    """
    polyline_lnglat: [[lng,lat], ...]
    return: list of samples {idx, lat, lng, bearing}
    """
    if not polyline_lnglat or len(polyline_lnglat) < 2:
        return []

    # 走一遍 polyline，按累计距离取样
    samples = []
    target = 0.0
    traveled = 0.0

    # 先把点展开成 (lat,lon)
    pts = [(p[1], p[0]) for p in polyline_lnglat]

    i = 0
    while i < len(pts) - 1 and len(samples) < max_points:
        lat1, lon1 = pts[i]
        lat2, lon2 = pts[i+1]
        seg = _haversine_m(lat1, lon1, lat2, lon2)
        if seg <= 0.01:
            i += 1
            continue

        # 如果下一采样点落在这个 segment 内
        while traveled + seg >= target and len(samples) < max_points:
            t = (target - traveled) / seg  # 0..1
            lat = lat1 + t*(lat2 - lat1)
            lon = lon1 + t*(lon2 - lon1)

            # bearing 用当前点指向一点点前方（用 segment 的方向）
            br = _bearing_deg(lat1, lon1, lat2, lon2)

            samples.append({
                "idx": len(samples),
                "lat": lat,
                "lng": lon,
                "bearing": br
            })
            target += float(spacing_m)

        traveled += seg
        i += 1

    return samples


# ---- endpoint ----
from pydantic import BaseModel

class SampleRequest(BaseModel):
    start: Point
    end: Point
    mode: str = "shortest"
    spacing_m: float = 100.0
    max_points: int = 20

@app.post("/api/route/samples")
def route_samples(req: SampleRequest):
    # 复用你已有的 /api/route 逻辑：直接调用 route() 函数（同进程）
    route_resp = route(RouteRequest(start=req.start, end=req.end, mode=req.mode))
    polyline = route_resp.get("polyline", [])
    samples = sample_polyline(polyline, spacing_m=req.spacing_m, max_points=req.max_points)
    return {
        "route_id": route_resp.get("route_id", req.mode),
        "mode": req.mode,
        "spacing_m": req.spacing_m,
        "samples": samples,
        "num_samples": len(samples)
    }
import os
import requests
from fastapi import HTTPException

MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN", "").strip()

def _mly_get(url, params):
    if not MAPILLARY_TOKEN:
        raise HTTPException(status_code=400, detail="MAPILLARY_TOKEN is missing in .env")
    headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail={"mapillary_status": r.status_code, "body": r.text[:500]})
    return r.json()
def _bbox_from_point(lat: float, lng: float, radius_m: float) -> str:
    # 用近似换算：1 deg lat ~ 111320m；lon 需要乘 cos(lat)
    dlat = radius_m / 111320.0
    dlng = radius_m / (111320.0 * max(0.1, math.cos(math.radians(lat))))
    min_lon = lng - dlng
    min_lat = lat - dlat
    max_lon = lng + dlng
    max_lat = lat + dlat
    return f"{min_lon},{min_lat},{max_lon},{max_lat}"

def mapillary_image_near(lat, lng, radius_m=60, bearing=None):
    if not MAPILLARY_TOKEN:
        raise HTTPException(status_code=400, detail="MAPILLARY_TOKEN is missing in .env")

    base = "https://graph.mapillary.com/images"

    # v4：用 bbox 搜附近图（closeto 在 v4 不好用/是 v3 思路）
    bbox = _bbox_from_point(lat, lng, radius_m)

    # 多要一个 geometry，方便你后面按“离采样点最近”排序
    fields = "id,thumb_1024_url,captured_at,compass_angle,creator,geometry"

    params = {
        "access_token": MAPILLARY_TOKEN,
        "fields": fields,
        "bbox": bbox,
        "limit": 10
    }

    r = requests.get(base, params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail={"mapillary_status": r.status_code, "body": r.text[:500]})

    data = r.json()
    items = data.get("data", [])
    if not items:
        return None

    # 选图策略：先按距离近，再按朝向接近 bearing（可选）
    def haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    def ang_diff(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    best = None
    best_score = 1e18

    for it in items:
        geom = it.get("geometry") or {}
        coords = (geom.get("coordinates") or [])
        if len(coords) == 2:
            ilon, ilat = coords[0], coords[1]
            dist = haversine_m(lat, lng, ilat, ilon)
        else:
            dist = 1e9

        ca = it.get("compass_angle")
        if bearing is not None and ca is not None:
            score = dist + 5.0 * ang_diff(float(ca), float(bearing))  # 5m/deg 的轻权重
        else:
            score = dist

        if score < best_score:
            best_score = score
            best = it

    creator = (best.get("creator") or {})
    creator_name = creator.get("username") or creator.get("name") or "Unknown"
    best["attribution"] = f"{creator_name} via Mapillary (CC BY-SA 4.0)"
    return best

class FetchImagesRequest(BaseModel):
    start: Point
    end: Point
    mode: str = "shortest"
    spacing_m: float = 100.0
    max_points: int = 20
    radius_m: float = 50.0

@app.post("/api/route/fetch-images")
def fetch_images(req: FetchImagesRequest):
    # 先拿 samples
    sresp = route_samples(SampleRequest(
        start=req.start, end=req.end, mode=req.mode,
        spacing_m=req.spacing_m, max_points=req.max_points
    ))
    samples = sresp["samples"]

    results = []
    for s in samples:
        img = mapillary_image_near(s["lat"], s["lng"], radius_m=req.radius_m, bearing=s.get("bearing"))
        results.append({
            "idx": s["idx"],
            "lat": s["lat"],
            "lng": s["lng"],
            "bearing": s["bearing"],
            "image": img
        })

    found = sum(1 for r in results if r["image"] is not None)
    return {
        "route_id": sresp["route_id"],
        "mode": req.mode,
        "radius_m": req.radius_m,
        "num_samples": len(samples),
        "num_images_found": found,
        "items": results
    }
