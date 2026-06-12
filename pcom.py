import torch, cv2, rasterio, os, json, numpy as np, gc
from lightglue import SuperPoint
from rasterio.warp import transform as transform_coords

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

ASSETS_DIR, GEOTIFF_PATH = "./offline_assets", "map.tif"
TILE_SIZE, OVERLAP = 1024, 256

def precompute():
    os.makedirs(os.path.join(ASSETS_DIR, "features"), exist_ok=True)
    # Using half precision for precompute
    extractor = SuperPoint(max_num_keypoints=2048).to(device).half().eval()
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).half().eval()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    ds = rasterio.open(GEOTIFF_PATH)
    meta, descs = [], []
    tile_idx = 0
    
    for y in range(0, ds.height - TILE_SIZE, TILE_SIZE - OVERLAP):
        for x in range(0, ds.width - TILE_SIZE, TILE_SIZE - OVERLAP):
            window = rasterio.windows.Window(x, y, TILE_SIZE, TILE_SIZE)
            img_rgb = ds.read([1, 2, 3], window=window).transpose(1, 2, 0)
            
            # Global Fingerprint
            img_p = cv2.resize(img_rgb, (224, 224))
            img_p_t = torch.from_numpy(img_p).permute(2,0,1).half().to(device).unsqueeze(0)/255.0
            with torch.inference_mode():
                descs.append(dino(img_p_t).cpu().numpy().flatten())
            
            # Precise Features (CLAHE + Half Precision)
            img_gray = clahe.apply(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
            img_t = torch.from_numpy(img_gray).half().to(device).unsqueeze(0).unsqueeze(0)/255.0
            with torch.inference_mode():
                feats = extractor.extract(img_t)
                # Save as half to reduce disk I/O time during main.py
                torch.save({k: v.cpu().half() if torch.is_floating_point(v) else v.cpu() 
                            for k, v in feats.items()}, 
                           os.path.join(ASSETS_DIR, "features", f"tile_{tile_idx}.pt"))

            raw_x, raw_y = ds.xy(y + TILE_SIZE//2, x + TILE_SIZE//2)
            lon, lat = transform_coords(ds.crs, 'EPSG:4326', [raw_x], [raw_y])
            meta.append({"id": tile_idx, "grid_pos": [x//(TILE_SIZE-OVERLAP), y//(TILE_SIZE-OVERLAP)],
                         "center_coords": [lon[0], lat[0]], "top_left_px": [x, y]})
            
            tile_idx += 1
            if tile_idx % 20 == 0:
                print(f"Processed {tile_idx} tiles...")
                gc.collect()

    np.save(os.path.join(ASSETS_DIR, "descriptors.npy"), np.array(descs).astype('float32'))
    with open(os.path.join(ASSETS_DIR, "metadata.json"), "w") as f: json.dump(meta, f)
    with open(os.path.join(ASSETS_DIR, "map_params.json"), "w") as f:
        json.dump({"transform": list(ds.transform), "crs": ds.crs.to_string()}, f)
    print("Precompute Done.")

if __name__ == "__main__":
    precompute()