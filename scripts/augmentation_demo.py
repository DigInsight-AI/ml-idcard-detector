"""Build a deterministic, reviewable augmentation demo from training labels."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = [
    "00180987abc66573f6cb0c40b70b1f41",
    "0032488f389c5a210a98a5c709c4f29b",
    "01e7e1b2b09a681c2212371a32eee702",
    "01d4d7b13ed9057909984faff2ce567e",
]
SIZE = 768
VARIANTS = [
    ("original", "original", "Original"),
    ("large_85", "large", "Large: longest side 85%"),
    ("large_95", "large", "Large: longest side 95%"),
    ("large_98", "large", "Large: longest side 98.5%"),
    ("clip_left", "clipping", "Crop left: 10% of bbox width"),
    ("clip_right", "clipping", "Crop right: 18% of bbox width"),
    ("clip_top", "clipping", "Crop top: 12% of bbox height"),
    ("clip_bottom", "clipping", "Crop bottom: 22% of bbox height"),
    ("clip_corner", "clipping", "Crop one corner: diagonal cut"),
    ("cover_corner", "occlusion", "Occlude K0"),
    ("cover_two_corners", "occlusion", "Occlude K0 + K1"),
    ("cover_edge", "occlusion", "Occlude edge; corners visible"),
    ("cover_center", "occlusion", "Occlude interior; corners visible"),
    ("cover_heavy", "occlusion", "Heavy occlusion: stress case"),
]


def project(points, matrix):
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(1, -1, 2), matrix
    )[0]


def intersection(points):
    canvas = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
    area, polygon = cv2.intersectConvexConvex(points.astype(np.float32), canvas)
    if polygon is None or area <= 0:
        raise ValueError("card does not intersect the canvas")
    return float(area), polygon.reshape(-1, 2)


def window_matrix(x0, y0, width, height):
    scale = SIZE / max(width, height)
    return np.array([
        [scale, 0, (SIZE - width * scale) / 2 - x0 * scale],
        [0, scale, (SIZE - height * scale) / 2 - y0 * scale],
        [0, 0, 1],
    ], dtype=np.float64)


def large_matrix(points, occupancy):
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    side = float(max(hi - lo) / occupancy)
    return window_matrix(center[0] - side / 2, center[1] - side / 2, side, side)


def clipping_matrix(base_points, variant):
    rotation = np.eye(3, dtype=np.float64)
    if variant == "clip_corner":
        direction = base_points.mean(axis=0) - base_points[0]
        direction /= np.linalg.norm(direction)
        ux, uy = direction
        rotation[:2, :2] = [[ux, uy], [-uy, ux]]
        base_points = project(base_points, rotation)
    lo, hi = base_points.min(axis=0), base_points.max(axis=0)
    width, height = hi - lo
    margin = 0.04 * max(width, height)
    if variant in ("clip_left", "clip_corner"):
        if variant == "clip_corner":
            ordered = np.sort(base_points[:, 0])
            cut = ordered[0] + 0.55 * (ordered[1] - ordered[0])
        else:
            cut = lo[0] + 0.10 * width
        side = max(height + 2 * margin, hi[0] - cut + margin)
        x0, y0 = cut, (lo[1] + hi[1] - side) / 2
    elif variant == "clip_right":
        cut = hi[0] - 0.18 * width
        side = max(height + 2 * margin, cut - lo[0] + margin)
        x0, y0 = cut - side, (lo[1] + hi[1] - side) / 2
    elif variant == "clip_top":
        cut = lo[1] + 0.12 * height
        side = max(width + 2 * margin, hi[1] - cut + margin)
        x0, y0 = (lo[0] + hi[0] - side) / 2, cut
    else:
        cut = hi[1] - 0.22 * height
        side = max(width + 2 * margin, cut - lo[1] + margin)
        x0, y0 = (lo[0] + hi[0] - side) / 2, cut - side
    return window_matrix(x0, y0, side, side) @ rotation


def occlusion_mask(points, variant):
    uv_to_image = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), points.astype(np.float32)
    )
    corner = np.float32([
        [-0.12, -0.15], [0.11, -0.14], [0.22, 0.01],
        [0.20, 0.17], [0.08, 0.26], [-0.10, 0.22],
    ])
    if variant == "cover_corner":
        polygons = [corner]
    elif variant == "cover_two_corners":
        polygons = [corner, corner * [-1, 1] + [1, 0]]
    elif variant == "cover_edge":
        polygons = [np.float32([
            [0.31, 1.16], [0.30, 0.86], [0.37, 0.74],
            [0.51, 0.73], [0.62, 0.85], [0.62, 1.16],
        ])]
    elif variant == "cover_center":
        polygons = [np.float32([
            [0.18, 0.31], [0.29, 0.24], [0.66, 0.30],
            [0.72, 0.55], [0.60, 0.65], [0.24, 0.60],
        ])]
    else:
        polygons = [np.float32([
            [-0.08, 0.11], [0.55, 0.10], [0.72, 0.30],
            [0.68, 0.78], [0.53, 0.90], [-0.08, 0.91],
        ])]
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.rint(project(polygon, uv_to_image)).astype(np.int32)], 255)
    return mask


def add_occluder(image, mask, seed, material):
    rng = np.random.default_rng(seed)
    colors = [(177, 193, 209), (68, 76, 66), (200, 201, 195), (61, 74, 153)]
    color = np.array(colors[material % len(colors)], dtype=np.float32)
    noise = rng.normal(0, 3.5, (SIZE, SIZE, 1)).astype(np.float32)
    y, x = np.mgrid[:SIZE, :SIZE]
    grain = 3 * np.sin(x * 0.45) + 2 * np.cos(y * 0.32)
    texture = np.clip(color + noise + grain[..., None], 0, 255)
    shadow = cv2.GaussianBlur(np.roll(mask, (5, 4), axis=(0, 1)), (21, 21), 0)
    result = image.astype(np.float32) * (1 - 0.20 * shadow[..., None] / 255)
    alpha = cv2.GaussianBlur(mask, (3, 3), 0)[..., None] / 255
    return np.clip(result * (1 - alpha) + texture * alpha, 0, 255).astype(np.uint8)


def make_sample(image, points, variant, seed, material):
    height, width = image.shape[:2]
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    if variant == "original":
        matrix = window_matrix(0, 0, width, height)
    elif variant.startswith("large_"):
        occupancy = {"large_85": 0.85, "large_95": 0.95, "large_98": 0.985}[variant]
        matrix = large_matrix(points, occupancy)
    else:
        matrix = large_matrix(points, 0.92)
        if variant.startswith("clip_"):
            matrix = clipping_matrix(project(points, matrix), variant) @ matrix
    output = cv2.warpPerspective(
        image, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114),
    )
    transformed = project(points, matrix)
    inside = ((transformed >= 0) & (transformed < SIZE)).all(axis=1)
    visibility = np.where(inside, 2, 0)
    if variant.startswith("cover_"):
        mask = occlusion_mask(transformed, variant)
        output = add_occluder(output, mask, seed, material)
        xy = np.clip(np.rint(transformed).astype(int), 0, SIZE - 1)
        visibility[inside & (mask[xy[:, 1], xy[:, 0]] > 127)] = 1
    visible_area, clipped_polygon = intersection(transformed)
    lo = np.clip(clipped_polygon.min(axis=0), 0, SIZE)
    hi = np.clip(clipped_polygon.max(axis=0), 0, SIZE)
    card_mask = np.zeros_like(mask)
    cv2.fillPoly(card_mask, [np.rint(transformed).astype(np.int32)], 255)
    occluded = np.count_nonzero((card_mask > 0) & (mask > 127))
    return output, transformed, visibility, np.r_[lo, hi], {
        "matrix_source_to_output": matrix.tolist(),
        "retained_card_area_fraction": visible_area / cv2.contourArea(transformed),
        "occluded_card_area_fraction": occluded / max(np.count_nonzero(card_mask), 1),
        "bbox_longest_side_fraction": float(max(hi - lo) / SIZE),
    }, mask


def yolo_label(class_id, points, visibility, bbox):
    center = (bbox[:2] + bbox[2:]) / (2 * SIZE)
    extent = (bbox[2:] - bbox[:2]) / SIZE
    values = [str(class_id)] + [f"{value:.8f}" for value in np.r_[center, extent]]
    for point, visible in zip(points, visibility):
        xy = point / SIZE if visible else [0.0, 0.0]
        values.extend([f"{xy[0]:.8f}", f"{xy[1]:.8f}", str(int(visible))])
    return " ".join(values) + "\n"


def validate_sample(variant, points, visibility, bbox, label, metrics):
    values = np.array(list(map(float, label.split())))
    assert len(values) == 17 and np.isfinite(values).all()
    assert np.all((values[1:5] >= 0) & (values[1:5] <= 1))
    triples = values[5:].reshape(4, 3)
    assert np.all((triples[:, :2] >= 0) & (triples[:, :2] <= 1))
    assert np.all(triples[visibility == 0, :2] == 0)
    assert np.allclose(triples[visibility > 0, :2] * SIZE, points[visibility > 0], atol=1e-4)
    decoded_bbox = np.r_[values[1:3] - values[3:5] / 2,
                         values[1:3] + values[3:5] / 2] * SIZE
    assert np.allclose(decoded_bbox, bbox, atol=1e-4)
    assert np.all(bbox[2:] > bbox[:2])
    assert cv2.isContourConvex(points.astype(np.float32))
    if variant.startswith("large_"):
        assert np.all(visibility == 2)
        target = {"large_85": 0.85, "large_95": 0.95, "large_98": 0.985}[variant]
        assert abs(metrics["bbox_longest_side_fraction"] - target) < 1e-5
    if variant.startswith("clip_"):
        assert np.count_nonzero(visibility == 0) >= 1
        assert np.count_nonzero(visibility == 2) >= 2
        assert 0.6 <= metrics["retained_card_area_fraction"] < 1
    if variant in ("cover_corner", "cover_two_corners"):
        assert np.count_nonzero(visibility == 1) == (1 if variant == "cover_corner" else 2)
    if variant in ("cover_edge", "cover_center", "cover_heavy"):
        assert np.all(visibility == 2)


def annotate(image, points, visibility, bbox):
    result = image.copy()
    quad = np.rint(points).astype(np.int32)
    cv2.polylines(result, [quad], True, (255, 209, 78), 2, cv2.LINE_AA)
    cv2.rectangle(result, tuple(np.rint(bbox[:2]).astype(int)),
                  tuple(np.rint(bbox[2:]).astype(int)), (0, 199, 255), 2)
    for index, (point, visible) in enumerate(zip(quad, visibility)):
        if not visible:
            continue
        color = (59, 218, 95) if visible == 2 else (199, 75, 244)
        cv2.circle(result, tuple(point), 9, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(result, tuple(point), 7, color, -1, cv2.LINE_AA)
        x = int(np.clip(point[0] + 13, 8, SIZE - 100))
        y = int(np.clip(point[1] - 13, 22, SIZE - 8))
        text = f"K{index} v={visible}"
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .60,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .60,
                    color, 1, cv2.LINE_AA)
    return result


def font(size):
    path = Path("/System/Library/Fonts/Helvetica.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def contact_sheet(records, group, output):
    variants = [v for v in VARIANTS if v[1] == group or v[0] == "original"]
    cell, row_height, top = 350, 420, 100
    sheet = Image.new("RGB", (cell * 4 + 40, top + len(variants) * row_height + 25), "#f3f5f6")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 15), f"ID CARD AUGMENTATION / {group.upper()}", fill="#18212a", font=font(26))
    draw.text((20, 53), "Green: v=2  |  Magenta: v=1  |  Missing: v=0  |  Yellow: bbox", fill="#38454c", font=font(17))
    for row, (variant, _, title) in enumerate(variants):
        for col in range(4):
            record = next(r for r in records if r["source_index"] == col and r["variant"] == variant)
            x, y = 20 + col * cell, top + row * row_height
            tile = Image.open(output / record["annotated"]).resize((330, 330), Image.Resampling.LANCZOS)
            sheet.paste(tile, (x, y))
            draw.text((x, y + 337), f"S{col + 1} / {title}", fill="#18212a", font=font(15))
            draw.text((x, y + 360), " ".join(f"K{i}:{v}" for i, v in enumerate(record["visibility"])), fill="#44515a", font=font(15))
            metric = record["metrics"]
            line = f"retained {metric['retained_card_area_fraction']:.0%} / covered {metric['occluded_card_area_fraction']:.0%}"
            draw.text((x, y + 383), line, fill="#44515a", font=font(15))
    sheet.save(output / f"preview-{group}.jpg", quality=92)


def write_gallery(records, output):
    cards = []
    for record in records:
        title = html.escape(record["title"])
        metric = record["metrics"]
        visibility = " ".join(f"K{i}:{v}" for i, v in enumerate(record["visibility"]))
        cards.append(f'''<article data-group="{record['group']}">
<a href="{record['annotated']}" target="_blank"><img loading="lazy" src="{record['annotated']}" data-raw="{record['image']}" data-label="{record['annotated']}" alt="S{record['source_index']+1} {title}"></a>
<h2>S{record['source_index']+1} / {title}</h2><p>{visibility}</p>
<p>保留 {metric['retained_card_area_fraction']:.1%} · 遮挡 {metric['occluded_card_area_fraction']:.1%} · 最长边占比 {metric['bbox_longest_side_fraction']:.1%}</p>
<p><a href="{record['image']}" target="_blank">原始输出</a> · <a href="{record['label']}" target="_blank">YOLO 标签</a></p></article>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>证件数据增强 Demo</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f5f6;color:#18212a;font:14px system-ui,sans-serif;letter-spacing:0}header{padding:22px 24px;background:white;border-bottom:1px solid #d7dee1}h1{font-size:24px;margin:0 0 10px}header p{margin:6px 0;line-height:1.6}nav{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin-top:16px}select{padding:6px;font:inherit}main{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;padding:24px}article{min-width:0;background:white;border:1px solid #d7dee1;border-radius:4px;overflow:hidden}img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#727272}h2{font-size:14px;margin:12px}article p{font-size:12px;line-height:1.6;margin:8px 12px;overflow-wrap:anywhere}a{color:#146b77}article[hidden]{display:none}@media(max-width:1000px){main{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){main{grid-template-columns:minmax(0,1fr);padding:14px}}</style>
<header><h1>证件数据增强 Demo</h1><p>4 张训练源图 · 正反面与横竖方向 · 52 张增强变体 + 4 张原图对照</p><p>绿点 v=2：可见；紫点 v=1：被遮挡且坐标准确；v=0：画外、不监督坐标。黄框：bbox。遮挡物为程序合成纹理。</p><nav><label>场景 <select id="group"><option value="all">全部</option><option value="original">原图</option><option value="large">大占比</option><option value="clipping">截断</option><option value="occlusion">遮挡</option></select></label><label><input id="labels" type="checkbox" checked> 显示标注</label><span id="count"></span></nav></header><main>CARDS</main>
<script>const group=document.querySelector('#group'),labels=document.querySelector('#labels');function update(){let count=0;document.querySelectorAll('article').forEach(a=>{a.hidden=group.value!=='all'&&a.dataset.group!==group.value;if(!a.hidden)count++;const img=a.querySelector('img');img.src=labels.checked?img.dataset.label:img.dataset.raw;a.querySelector('a').href=img.src});document.querySelector('#count').textContent=count+' 张'}group.addEventListener('change',update);labels.addEventListener('change',update);update();</script></html>'''
    (output / "index.html").write_text(document.replace("CARDS", "\n".join(cards)), encoding="utf-8")


def overview(records, output):
    variants = ["original", "large_95", "large_98", "clip_right", "clip_corner",
                "cover_corner", "cover_center", "cover_heavy"]
    sheet = Image.new("RGB", (1440, 965), "#f3f5f6")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 15), "AUGMENTATION DEMO / TRANSFORMED GROUND TRUTH", fill="#18212a", font=font(25))
    draw.text((20, 55), "Green: visible  |  Magenta: occluded  |  Cropped-out corners: v=0  |  Yellow: bbox", fill="#38454c", font=font(18))
    for index, variant in enumerate(variants):
        record = next(r for r in records if r["source_index"] == 0 and r["variant"] == variant)
        x, y = 20 + index % 4 * 350, 100 + index // 4 * 420
        image = Image.open(output / record["annotated"]).resize((330, 330), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y))
        draw.text((x, y + 338), record["title"], fill="#18212a", font=font(16))
        draw.text((x, y + 363), " ".join(f"K{i}:{v}" for i, v in enumerate(record["visibility"])), fill="#44515a", font=font(16))
        metrics = record["metrics"]
        text = f"Retained {metrics['retained_card_area_fraction']:.0%} / covered {metrics['occluded_card_area_fraction']:.0%}"
        draw.text((x, y + 388), text, fill="#44515a", font=font(16))
    sheet.save(output / "overview.jpg", quality=93)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "runs/datasets/config_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/augmentation-demo-20261022")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output.resolve()
    for directory in ("images/train", "labels/train", "annotated", "masks"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    records = []
    for source_index, stem in enumerate(DEFAULT_SOURCES):
        source = next((args.dataset / "images/train").glob(stem + ".*"))
        lines = (args.dataset / "labels/train" / (stem + ".txt")).read_text().splitlines()
        if len(lines) != 1:
            raise ValueError("demo sources must contain one target")
        values = np.array(list(map(float, lines[0].split())))
        if len(values) != 17 or not np.all(values[7::3] == 2):
            raise ValueError("demo sources must have four visible labeled corners")
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"cannot decode {source}")
        height, width = image.shape[:2]
        points = values[5:].reshape(4, 3)[:, :2] * [width, height]
        for variant_index, (variant, group, title) in enumerate(VARIANTS):
            seed = args.seed + source_index * 100 + variant_index
            result, transformed, visibility, bbox, metrics, mask = make_sample(
                image, points, variant, seed, source_index + variant_index
            )
            label = yolo_label(int(values[0]), transformed, visibility, bbox)
            validate_sample(variant, transformed, visibility, bbox, label, metrics)
            reconstructed = project(transformed, np.linalg.inv(np.array(metrics["matrix_source_to_output"])))
            if not np.allclose(reconstructed, points, atol=0.002):
                raise ValueError("keypoints failed inverse-transform consistency check")
            name = f"s{source_index + 1}_{variant}"
            paths = {"image": f"images/train/{name}.jpg", "label": f"labels/train/{name}.txt",
                     "annotated": f"annotated/{name}.jpg", "mask": f"masks/{name}.png"}
            for key, data in (("image", result), ("annotated", annotate(result, transformed, visibility, bbox)), ("mask", mask)):
                if not cv2.imwrite(str(output / paths[key]), data):
                    raise OSError(f"could not write {paths[key]}")
            (output / paths["label"]).write_text(label, encoding="ascii")
            records.append({
                "source_index": source_index, "source": str(source.resolve()),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_keypoints": points.tolist(),
                "class_id": int(values[0]), "variant": variant, "group": group,
                "title": title, "seed": seed, "visibility": visibility.tolist(),
                "transformed_keypoints": transformed.tolist(), "bbox": bbox.tolist(),
                "metrics": metrics, **paths,
            })
    for group in ("large", "clipping", "occlusion"):
        contact_sheet(records, group, output)
    overview(records, output)
    write_gallery(records, output)
    report = {"seed": args.seed, "canvas_size": SIZE, "samples": len(records),
              "checks": "passed", "notes": [
                  "Demo only; not included in training manifests.",
                  "Occluders are synthetic textured polygons, not real hand cutouts.",
                  "Heavy occlusion is a stress case, not a default training ratio.",
                  "Geometry uses uniform scale, rotation and crop; no aspect stretching.",
              ], "records": records}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"output": str(output), "samples": len(records), "checks": "passed"}, indent=2))


if __name__ == "__main__":
    main()
