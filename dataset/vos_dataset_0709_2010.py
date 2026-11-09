import os
import json
import numpy as np
import cv2
import torch
import random
import logging

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from dataset.utils import sort_by_number, reseed

log = logging.getLogger(__name__)

class TenCamusDataset(Dataset):
    def __init__(self, filepath: str, mode: str = 'train', seq_length=10, max_num_obj=1, size=256, merge_probability=0.0, sparse_gt=False, hide_gt=False):
        super().__init__()
        self.filepath = filepath
        self.mode = mode
        self.seq_length = seq_length
        self.max_num_obj = max_num_obj
        self.size = size
        self.merge_probability = merge_probability
        self.sparse_gt = sparse_gt
        # UDA target mode: zero ALL label tensors so no target annotation can
        # ever leak into a loss. Images, selector and num_objects are kept so
        # the forward path stays well-formed. Default False = unchanged.
        self.hide_gt = hide_gt
        # video_id -> annotated (non-empty mask) frame indices; only used when sparse_gt
        self.annotated_frames = {}
        if sparse_gt:
            self.annotated_frames = self._load_annotated_cache()

        json_path = os.path.join(filepath, 'camus_public_datasplit_20250706.json')
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"{json_path} not found")

        with open(json_path, 'r') as f:
            all_data = json.load(f)
        if mode == 'train':
            self.patients = all_data['train_data']
        elif mode == 'val':
            self.patients = all_data['val_data']
        elif mode == 'test':
            self.patients = all_data['test_data']
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # collect all patient samples
        self.samples = []
        for pid in self.patients:
            img_dir = os.path.join(self.filepath, 'img', pid)
            mask_dir = os.path.join(self.filepath, 'gt_lv', pid)

            if (not os.path.isdir(img_dir)) or (not os.path.isdir(mask_dir)):
                log.warning(f"Skipping {pid}: missing image or mask dir.")
                continue

            img_list = sorted(os.listdir(img_dir), key=sort_by_number)
            mask_list = sorted(os.listdir(mask_dir), key=sort_by_number)

            if len(img_list) < self.seq_length:
                log.warning(f"Skipping {pid}: {len(img_list)} frames < required {self.seq_length}.")
                continue

            self.samples.append({
                'patient_id': pid,
                'img_dir': img_dir,
                'mask_dir': mask_dir,
                'img_list': img_list,
                'mask_list': mask_list,
            })

            if self.sparse_gt:
                ann = self.annotated_frames.get(pid, [])
                if len(ann) == 0:
                    log.warning(f"Skipping {pid}: no annotated frames (sparse_gt mode).")
                    self.samples.pop()
                    continue
                # need an annotated frame early enough to host a full window
                if not any(a + self.seq_length <= len(img_list) for a in ann):
                    log.warning(f"Skipping {pid}: annotations too late for a "
                                f"{self.seq_length}-frame window (sparse_gt mode).")
                    self.samples.pop()

        # sequence-level transform (shared across the whole clip)
        # demo: convert to PIL/Tensor then apply torchvision augmentations
        if self.mode == 'train':
            self.seq_transform_img = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            # at test time: resize only
            self.seq_transform_img = transforms.Compose([
                # optional: add Resize or leave empty
            ])

        # frame-level transform (applied per frame)
        # demo: ToTensor() + Resize; add ColorJitter/RandomAffine if needed
        self.frame_transform_img = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),  # => [0,1]
        ])

        # note: mask is a label (0/1), no /255 ToTensor() needed
        # but for pipeline consistency: PIL -> Resize -> long tensor
        self.frame_transform_mask = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.samples)

    def _load_annotated_cache(self) -> dict:
        """Sparse-GT datasets (e.g. EchoNet: 2 annotated frames per long video):
        cache video_id -> frame NUMBERS (from filename) with non-empty masks,
        so windows can be anchored on an annotated frame (required for
        ff_gt / num_objects). Numbers are mapped to img_list positions lazily
        because img/gt dirs may hold different file counts."""
        cache_path = os.path.join(self.filepath, '_annotated_frames.json')
        if os.path.isfile(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        log.warning(f"{cache_path} not found; scanning masks to build it (one-time)...")
        cache = {}
        for pid in self.patients:
            mask_dir = os.path.join(self.filepath, 'gt_lv', pid)
            if not os.path.isdir(mask_dir):
                continue
            ann = []
            for fname in sorted(os.listdir(mask_dir), key=sort_by_number):
                num = sort_by_number(fname)
                if num < 0:
                    continue
                m = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
                if m is not None and (m > 0).sum() > 0:
                    ann.append(num)
            cache[pid] = ann
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
        return cache

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        patient_id = sample['patient_id']
        img_dir = sample['img_dir']
        mask_dir = sample['mask_dir']
        img_list = sample['img_list']

        T = self.seq_length
        total_frames = len(img_list)

        if self.sparse_gt:
            # anchor the window on an annotated frame (position 0 carries ff_gt);
            # map annotated frame NUMBERS to img_list positions, keeping only
            # anchors that leave room for a full window
            ann_nums = set(self.annotated_frames.get(patient_id, []))
            ann = [i for i, name in enumerate(img_list)
                   if sort_by_number(name) in ann_nums and i + T <= total_frames]
            if ann:
                start_frame_idx = random.choice(ann)
            else:  # pathological: only late/missing annotations -> keep window valid
                start_frame_idx = max(0, total_frames - T)
        elif total_frames > T:
            start_frame_idx = random.randint(0, total_frames - T)
        else:
            start_frame_idx = 0
        frame_indices = range(start_frame_idx, start_frame_idx + T)

        frames = []
        masks = []

        for current_frame_idx in frame_indices:
            img_name = img_list[current_frame_idx]
            img_path = os.path.join(img_dir, img_name)
            
            # --- START of CHANGE: load mask for every frame ---
            # removed the previous "first/last frame only" mask load.
            # now we try to load a same-named mask file for every frame in the sequence.
            mask_img = np.zeros((256, 256), dtype=np.uint8)  # fallback: all-zero mask

            mask_path = os.path.join(mask_dir, img_name)
            if os.path.isfile(mask_path):
                loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if loaded_mask is not None:
                    mask_img = loaded_mask
                else:
                    log.warning(f"Failed to read mask from {mask_path}, using zeros.")
            # if mask file is missing, silently fall back to all-zero mask
            # --- END of CHANGE ---

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                log.warning(f"Failed to read image from {img_path}, using zeros.")
                image = np.zeros((256, 256), dtype=np.uint8)

            # convert to PIL for torchvision transforms
            pil_img = to_pil_image(image)
            pil_mask = to_pil_image(mask_img)

            frames.append(pil_img)
            masks.append(pil_mask)

        # ----------- sequence-level transform -----------
        # apply the same random transform across the whole sequence
        # strategy: sample one seed, then apply the same transform to every frame
        seq_seed = random.randint(0, 99999)
        transformed_frames = []
        transformed_masks = []

        for i in range(T):
            # image
            reseed(seq_seed)
            f_img = self.seq_transform_img(frames[i])
            transformed_frames.append(f_img)

            # mask must use the same random op as the image
            # torchvision transforms don't sync flip/rotation between image and mask,
            # so for production use Albumentations with shared random_seed
            # demo: apply the same transforms to mask too
            reseed(seq_seed)
            f_mask = self.seq_transform_img(masks[i])
            transformed_masks.append(f_mask)

        # ----------- frame-level transform -----------
        # per-frame: ToTensor(), Resize
        final_imgs = []
        final_masks = []
        for i in range(T):
            # image
            img_t = self.frame_transform_img(transformed_frames[i])  # => [C,H,W], float in [0,1]
            # mask
            mask_pil = self.frame_transform_mask(transformed_masks[i])
            mask_np = np.array(mask_pil, dtype=np.uint8)
            mask_np[mask_np != 1] = 0  # keep fg=1 only
            # mask_np[mask_np > 0] = 1  # alt: collapse all fg to 1
            mask_t = torch.from_numpy(mask_np).long().unsqueeze(0)  # => [1,H,W]

            final_imgs.append(img_t)
            final_masks.append(mask_t)

        # stack into [T, C, H, W] / [T, 1, H, W]
        final_imgs_t = torch.stack(final_imgs, dim=0)   # => [T, C, H, W]
        final_masks_t = torch.stack(final_masks, dim=0) # => [T, 1, H, W]

        # ----------- VOS-style output structure -----------
        # info holds metadata
        info = {
            'name': patient_id,
            'frames': [img_list[i] for i in range(T)]
        }

        # 1) collect non-zero labels from the first-frame mask => target_objects
        #    final_masks_t[0] has shape [1,H,W]
        first_frame_labels = final_masks_t[0].unique()
        first_frame_labels = [v.item() for v in first_frame_labels if v.item() != 0]
        # truncate to max_num_obj
        target_objects = first_frame_labels[:self.max_num_obj]

        # 2) build cls_gt: same shape as final_masks_t [T,1,H,W], init 0
        cls_gt = torch.zeros_like(final_masks_t)  # dtype long, shape [T,1,H,W]

        # 3) build first_frame_gt: [1, max_num_obj, H, W]
        first_frame_gt = torch.zeros(
            (1, self.max_num_obj, self.size, self.size), dtype=torch.long
        )

        # 4) iterate over each target ID and assign
        #    - cls_gt[this_mask] = i+1
        #    - first_frame_gt[0,i] = this_mask[0,0]
        for i, l in enumerate(target_objects):
            # this_mask => bool tensor [T,1,H,W]
            this_mask = (final_masks_t == l)

            # set pixels of this object to i+1 in cls_gt
            cls_gt[this_mask] = i + 1

            # first-frame channel i is the binary mask of this object
            # this_mask[0] => [1,H,W] => this_mask[0,0] => [H,W]
            first_frame_gt[0, i] = this_mask[0, 0].long()

        # 5) selector: [max_num_obj], 1 for present objects, 0 otherwise
        selector = torch.zeros(self.max_num_obj, dtype=torch.float32)
        selector[:len(target_objects)] = 1.0

        # 6) per-frame GT validity (1 = frame has a non-empty mask). Dense-GT
        #    datasets (CAMUS) are all-ones; sparse-GT ones (EchoNet) mark the
        #    annotated frames so the loss can skip empty-GT frames.
        gt_valid = (final_masks_t.sum(dim=(1, 2, 3)) > 0).float()

        # num_objects (record only, optional)
        info['num_objects'] = len(target_objects)

        data = {
            'rgb'      : final_imgs_t,    # [T,C,H,W], float in [0,1]
            'ff_gt'    : first_frame_gt,  # [T=1, num_objects, H, W]
            'cls_gt'   : cls_gt,          # [T,1,H,W], long
            'selector' : selector,        # [1]
            'gt_valid' : gt_valid,        # [T], float
            'info'     : info,
        }

        if self.hide_gt:
            data['ff_gt'] = torch.zeros_like(data['ff_gt'])
            data['cls_gt'] = torch.zeros_like(data['cls_gt'])
            data['gt_valid'] = torch.zeros_like(data['gt_valid'])

        return data


