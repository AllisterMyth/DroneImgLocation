import os, gc, sys, json, time, csv, torch, cv2, argparse, warnings
import numpy as np
from lightglue import SuperPoint, LightGlue
from lightglue.utils import rbd
import rasterio.transform
from rasterio.warp import transform as transform_coords 
import contextlib

# --- SUPPRESS NOISE AND WARNINGS ---
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning)   
os.environ['TORCH_HUB_LOG_LEVEL'] = 'ERROR'               

# Force PyTorch to use CPU fallback for Apple Silicon (MPS)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# --- DEVICE SELECTION ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

ASSETS_DIR, INPUT_FOLDER = "./offline_assets", "./Droneimg"
STATE_FILE = "tracker_state.json"

class TrackerLocalizer:
    def __init__(self):
        # --- CAMERA CALIBRATION (FROM USER DATA) ---
        self.orig_fx = 456.46871015134053
        self.orig_width = 1280
        
        with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            self.extractor = SuperPoint(max_num_keypoints=1024).to(device).half().eval() 
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).half().eval()
            self.matcher = LightGlue(features='superpoint').to(device).eval()
        
        self.descriptors = np.load(os.path.join(ASSETS_DIR, "descriptors.npy"))
        with open(os.path.join(ASSETS_DIR, "metadata.json"), "r") as f: self.metadata = json.load(f)
        
        with open(os.path.join(ASSETS_DIR, "map_params.json"), "r") as f:
            mp = json.load(f)
            self.map_transform = rasterio.transform.Affine(*mp['transform'])
            self.map_crs = mp['crs']
            
            # Calculate Map Ground Sample Distance (GSD) - meters per pixel
            # a is horizontal pixel resolution. If CRS is EPSG:4326, convert degrees to meters.
            self.map_gsd = abs(self.map_transform.a)
            if "4326" in str(self.map_crs):
                self.map_gsd *= 111319.49 # Approximation for meters per degree

        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                self.last_idx = state.get('last_idx')
                self.last_rot = state.get('last_rot', 0)
        else:
            self.last_idx = None
            self.last_rot = 0 

    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            last_idx_json = int(self.last_idx) if self.last_idx is not None else None
            last_rot_json = int(self.last_rot)
            json.dump({'last_idx': last_idx_json, 'last_rot': last_rot_json}, f)

    def get_local_indices(self):
        if self.last_idx is None: return []
        curr = self.metadata[self.last_idx]['grid_pos']
        return [i for i, m in enumerate(self.metadata) 
                if abs(m['grid_pos'][0]-curr[0]) <= 1 and abs(m['grid_pos'][1]-curr[1]) <= 1]

    def extract_at_scale(self, img_gray, scale=1.0):
        if scale != 1.0:
            img_gray = cv2.resize(img_gray, (0, 0), fx=scale, fy=scale)
        img_t = torch.from_numpy(img_gray).half().to(device).unsqueeze(0).unsqueeze(0)/255.0
        with torch.inference_mode():
            feats = self.extractor.extract(img_t)
        return feats, img_gray.shape

    def try_match(self, feats0, tile_indices):
        feats0_f = {k: v.float() if torch.is_floating_point(v) else v for k, v in feats0.items()}
        for idx in tile_indices:
            f1_raw = torch.load(os.path.join(ASSETS_DIR, "features", f"tile_{idx}.pt"), 
                                map_location=device, weights_only=True)
            f1 = {k: v.float() if torch.is_floating_point(v) else v for k, v in f1_raw.items()}
            with torch.inference_mode():
                matches01 = self.matcher({"image0": feats0_f, "image1": f1})
            m = matches01['matches0'][0]
            valid = m > -1
            if valid.sum() < 12: 
                del f1, f1_raw, matches01
                continue
            pts0 = rbd(feats0_f)['keypoints'][valid].cpu().numpy()
            pts1 = rbd(f1)['keypoints'][m[valid]].cpu().numpy()
            H, mask = cv2.findHomography(pts0, pts1, cv2.USAC_MAGSAC, 4.0)
            if H is not None and mask.sum() >= 12:
                inlier_count = int(mask.sum())
                del f1, f1_raw, matches01 
                return idx, H, inlier_count
            del f1, f1_raw, matches01
        return None, None, 0

    def finalize(self, idx, H, inliers, shape):
        """Calculates Lon, Lat, and Altitude using Homography scale."""
        # 1. LAT/LON CALCULATION
        cp = np.array([[[shape[1]/2, shape[0]/2]]], dtype=np.float32)
        px = cv2.perspectiveTransform(cp, H)[0][0]
        meta = self.metadata[idx]
        gx, gy = meta['top_left_px'][0] + px[0], meta['top_left_px'][1] + px[1]
        proj_x, proj_y = rasterio.transform.xy(self.map_transform, gy, gx)
        lon, lat = transform_coords(self.map_crs, 'EPSG:4326', [proj_x], [proj_y])
        
        # 2. ALTITUDE CALCULATION
        try:
            # Determinant of the 2x2 rotation/scaling part of Homography
            # This represents the area scale factor between image and map
            scale_factor = np.sqrt(abs(np.linalg.det(H[:2, :2])))
            
            # Adjust focal length for current image resize (e.g. 1024 or 512)
            focal_adjusted = self.orig_fx * (shape[1] / self.orig_width)
            
            # Altitude formula: (Focal_px * GSD_m) / Scale_Factor
            altitude = (focal_adjusted * self.map_gsd) / scale_factor
        except:
            altitude = 0.0

        self.last_idx = int(idx) 
        return [lon[0], lat[0], altitude], inliers, True

    def find_location(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: return [0,0,0], 0, False, "ERROR"
        
        h, w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, (1024, int(1024 * h/w)))
        
        rot_order = [self.last_rot, (self.last_rot + 1) % 4, (self.last_rot + 3) % 4, (self.last_rot + 2) % 4]

        # 1. LOCAL SEARCH
        local_ids = self.get_local_indices()
        if local_ids:
            for r in rot_order:
                img_rotated = np.rot90(img_resized, k=r)
                img_gray = self.clahe.apply(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY))
                
                # Try Local at 1.0 scale
                feats0, shape0 = self.extract_at_scale(img_gray, scale=1.0)
                idx, H, inliers = self.try_match(feats0, local_ids)
                if idx is not None:
                    self.last_rot = r
                    res, inliers, success = self.finalize(idx, H, inliers, shape0)
                    return res, inliers, success, f"LOCAL_1.0_R{r}"
                
                # Try Local at 0.5 scale fallback
                feats0_05, shape0_05 = self.extract_at_scale(img_gray, scale=0.5)
                idx, H, inliers = self.try_match(feats0_05, local_ids)
                if idx is not None:
                    self.last_rot = r
                    res, inliers, success = self.finalize(idx, H, inliers, shape0_05)
                    return res, inliers, success, f"LOCAL_0.5_R{r}"
                
                del feats0, feats0_05, img_rotated, img_gray

        # 2. GLOBAL SEARCH
        for r in rot_order:
            img_r = np.rot90(img_resized, k=r)
            img_gray_r = self.clahe.apply(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY))
            img_p = cv2.resize(img_r, (224, 224))
            img_p_t = torch.from_numpy(cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)).permute(2,0,1).half().to(device).unsqueeze(0)/255.0
            with torch.inference_mode():
                desc = self.dino(img_p_t).cpu().numpy().flatten()
            
            sims = np.dot(self.descriptors, desc)
            top_tiles = np.argsort(sims)[-10:][::-1]
            
            # Global 1.0 Scale
            feats0_r, shape_r = self.extract_at_scale(img_gray_r, scale=1.0)
            idx, H, inliers = self.try_match(feats0_r, top_tiles)
            if idx is not None:
                self.last_rot = r
                res, inliers, success = self.finalize(idx, H, inliers, shape_r)
                return res, inliers, success, f"GLOBAL_1.0_R{r}"
            
            # Global 0.5 Scale
            feats0_r_05, shape_r_05 = self.extract_at_scale(img_gray_r, scale=0.5)
            idx, H, inliers = self.try_match(feats0_r_05, top_tiles)
            if idx is not None:
                self.last_rot = r
                res, inliers, success = self.finalize(idx, H, inliers, shape_r_05)
                return res, inliers, success, f"GLOBAL_0.5_R{r}"
            
            del feats0_r, feats0_r_05, img_p_t, img_r, img_gray_r
            gc.collect()
            if device.type == 'mps': torch.mps.empty_cache()
            elif device.type == 'cuda': torch.cuda.empty_cache()

        self.last_idx = None 
        return [0,0,0], 0, False, "GLOBAL_FAILED"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    tracker = TrackerLocalizer()
    img_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    target_images = img_files[args.start : args.start + args.count]
    
    output_file = "localization_results.csv"
    write_header = not os.path.exists(output_file)

    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            # Added altitude_m column
            writer.writerow(['filename', 'longitude', 'latitude', 'altitude_m', 'inliers', 'time_taken_s', 'mode', 'status'])
        
        for img_name in target_images:
            start_t = time.time()
            img_path = os.path.join(INPUT_FOLDER, img_name)
            
            # results now returns [lon, lat, alt]
            results, inliers, success, mode = tracker.find_location(img_path)
            
            elapsed = time.time() - start_t
            status = "LOCKED" if success else "FAILED"
            
            writer.writerow([
                img_name, 
                results[0],           # Lon
                results[1],           # Lat
                round(results[2], 2), # Altitude
                inliers, 
                round(elapsed, 4), 
                mode, 
                status
            ])
            
            f.flush()
            os.fsync(f.fileno()) 
            
            print(f"{status:7} | {img_name:15} | Alt: {results[2]:5.1f}m | Inliers: {inliers:2} | {elapsed:.2f}s | {mode}")
            
            gc.collect()
            if device.type == 'mps': torch.mps.empty_cache()
            elif device.type == 'cuda': torch.cuda.empty_cache()
            
    tracker.save_state()

if __name__ == "__main__":
    main()