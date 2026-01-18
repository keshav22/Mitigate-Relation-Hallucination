import os
import json
import argparse
import torch
import re
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict

try:
    from groundingdino.util.inference import annotate
except Exception:
    annotate = None


def get_vg_path(image_id: str, vg_root: str):
    image_id = str(image_id).replace(".jpg", "")
    p1 = os.path.join(vg_root, "VG_100K", f"{image_id}.jpg")
    p2 = os.path.join(vg_root, "VG_100K_2", f"{image_id}.jpg")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None

def clean_prompt(prompt: str) -> str:
    prompt = prompt.split('?')[0]

    prompt = prompt.lower()

    prompt = re.sub(
        r'^(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\s+',
        '',
        prompt
    )

    prompt = re.sub(
        r'\b(in this photo|in this image|in the photo|in the image)\b',
        '',
        prompt
    )

    prompt = re.sub(r'\s+', ' ', prompt).strip()
    prompt = " . ".join(prompt.split()) + " ."   
    return prompt

def norm_cxcywh_to_xywh_pixel(box, W, H):
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H

    x1 = max(0.0, min(float(W - 1), x1))
    y1 = max(0.0, min(float(H - 1), y1))
    x2 = max(0.0, min(float(W - 1), x2))
    y2 = max(0.0, min(float(H - 1), y2))

    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    return {
        "x": int(round(x1)),
        "y": int(round(y1)),
        "w": int(round(x2 - x1)),
        "h": int(round(y2 - y1)),
    }


def run_dino(model, image_path, caption, box_threshold, text_threshold, device, topk):
    image_source, image = load_image(image_path)
    H, W = image_source.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    dets = []
    if boxes is None or len(boxes) == 0:
        return dets, W, H, image_source, boxes, logits, phrases

    boxes_list = boxes.detach().cpu().tolist()
    logits_list = logits.detach().cpu().tolist() if torch.is_tensor(logits) else list(logits)

    for b, s, ph in zip(boxes_list, logits_list, phrases):
        dets.append({
            **norm_cxcywh_to_xywh_pixel(b, W, H),
            "score": float(s),
            "matched_phrase": str(ph),
            "img_w": int(W),
            "img_h": int(H),
        })

    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets, W, H, image_source, boxes, logits, phrases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yesno_jsonl", required=True)
    ap.add_argument("--vg_root", required=True)
    ap.add_argument("--dino_config", required=True)
    ap.add_argument("--dino_weights", required=True)

    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=20)

    ap.add_argument("--unique_images", action="store_true", default=True,
                    help="Process each image_id only once (skip duplicates). Default: True")
    ap.add_argument("--max_samples", type=int, required=False,
                    help="Stop after processing N unique images (only counts unique image_ids).")

    ap.add_argument("--out_jsonl", default="dino_out.jsonl")
    ap.add_argument("--debug_dir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.dino_config, args.dino_weights).to(device).eval()

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    seen_images = set()
    processed_unique = 0
    skipped_no_image = 0
    skipped_duplicates = 0

    with open(args.out_jsonl, "w") as out_fp, open(args.yesno_jsonl, "r") as fp:
        for line in tqdm(fp, desc="GroundingDINO inference"):
            if args.max_samples is not None:
                if processed_unique >= args.max_samples:
                    break

            ex = json.loads(line)
            img_id = str(ex["image_id"]).replace(".jpg", "")
            q = clean_prompt(ex["query_prompt"])

            if args.unique_images and img_id in seen_images:
                skipped_duplicates += 1
                continue

            image_path = get_vg_path(img_id, args.vg_root)
            if image_path is None:
                skipped_no_image += 1
                # still mark as seen so we don't repeatedly fail on same image_id
                if args.unique_images:
                    seen_images.add(img_id)
                out_fp.write(json.dumps({
                    "image_id": img_id,
                    "query_prompt": q,
                    "error": "image_not_found",
                    "detections": [],
                }) + "\n")
                continue

            # mark seen BEFORE running so duplicates later are skipped
            if args.unique_images:
                seen_images.add(img_id)

            dets, W, H, image_source, boxes, logits, phrases = run_dino(
                model=model,
                image_path=image_path,
                caption=q,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=device,
                topk=args.topk,
            )

            out_fp.write(json.dumps({
                "image_id": img_id,
                "query_prompt": q,
                "org_query_prompt": ex["query_prompt"],
                "img_w": W,
                "img_h": H,
                "detections": dets,
            }) + "\n")
            out_fp.flush()

            if args.debug_dir and annotate is not None and boxes is not None and len(boxes) > 0:
                try:
                    import cv2
                    annotated = annotate(
                        image_source=image_source,
                        boxes=boxes,
                        logits=logits,
                        phrases=phrases,
                    )
                    cv2.imwrite(
                        os.path.join(args.debug_dir, f"{img_id}.jpg"),
                        annotated,
                    )
                except Exception as e:
                    print(f"[WARN] Debug image failed for {img_id}: {e}")

            processed_unique += 1

    print(f"Done. processed_unique={processed_unique}, device={device}")
    print(f"skipped_duplicates={skipped_duplicates}, skipped_no_image={skipped_no_image}")
    print(f"Wrote {args.out_jsonl}")


if __name__ == "__main__":
    main()