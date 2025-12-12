# Copyright (c) OpenMMLab. All rights reserved.
"""
NuScenes-MQA Dataset for Language-guided BEVFormer
Integrates question-answer pairs with NuScenes driving data.
"""

import copy
import csv
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC

from .nuscenes_dataset import CustomNuScenesDataset


# Camera direction to BEV sector mapping
# BEV coordinate: x is forward, y is left
# Angles are measured from x-axis (forward), counter-clockwise
CAMERA_SECTORS = {
    'front': {'angle_range': (-30, 30), 'name': 'CAM_FRONT'},
    'front left': {'angle_range': (30, 70), 'name': 'CAM_FRONT_LEFT'},
    'front right': {'angle_range': (-70, -30), 'name': 'CAM_FRONT_RIGHT'},
    'back': {'angle_range': (150, 180), 'name': 'CAM_BACK'},  # Also covers -180 to -150
    'back left': {'angle_range': (110, 150), 'name': 'CAM_BACK_LEFT'},
    'back right': {'angle_range': (-150, -110), 'name': 'CAM_BACK_RIGHT'},
}

# Object class mapping for nuScenes
OBJECT_CLASSES = {
    'car': 0,
    'truck': 1,
    'construction_vehicle': 2,
    'bus': 3,
    'trailer': 4,
    'barrier': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'pedestrian': 8,
    'traffic_cone': 9,
    # Extended classes from MQA
    'adult pedestrian': 8,
    'child pedestrian': 8,
    'police officer': 8,
    'construction worker': 8,
    'stroller': 10,
    'wheelchair': 10,
    'portable personal mobility vehicle': 10,
    'animal': 11,
    'debris': 12,
    'pushable or pullable object': 12,
    'bicycle rack': 12,
    'rigid bus': 3,
    'bendy bus': 3,
    'bicycles': 7,
    'cars': 0,
    'trucks': 1,
    'motorcycles': 6,
    'pedestrians': 8,
}


def parse_markup(text: str) -> Dict[str, Any]:
    """Parse markup tags from MQA answer text.
    
    Tags:
        <cam>direction</cam>: Camera direction
        <obj>class</obj>: Object class
        <cnt>number</cnt>: Object count
        <loc>(x, y)</loc>: Location coordinates
        <dst>distance</dst>: Distance in meters
        <target>...</target>: Target object with attributes
    
    Returns:
        Dict with parsed fields
    """
    result = {
        'cameras': [],
        'objects': [],
        'counts': [],
        'locations': [],
        'distances': [],
        'targets': [],
    }
    
    # Parse <cam> tags
    cam_pattern = r'<cam>([^<]+)</cam>'
    result['cameras'] = re.findall(cam_pattern, text)
    
    # Parse <obj> tags
    obj_pattern = r'<obj>([^<]+)</obj>'
    result['objects'] = re.findall(obj_pattern, text)
    
    # Parse <cnt> tags
    cnt_pattern = r'<cnt>(\d+)</cnt>'
    result['counts'] = [int(c) for c in re.findall(cnt_pattern, text)]
    
    # Parse <loc> tags - format: (x, y)
    loc_pattern = r'<loc>\(([^,]+),\s*([^)]+)\)</loc>'
    locs = re.findall(loc_pattern, text)
    result['locations'] = [(float(x), float(y)) for x, y in locs]
    
    # Parse <dst> tags
    dst_pattern = r'<dst>([0-9.]+)</dst>'
    dsts = re.findall(dst_pattern, text)
    result['distances'] = [float(d) for d in dsts]
    
    # Parse <target> tags - contains cnt and obj
    target_pattern = r'<target>([^<]*(?:<[^>]+>[^<]*)*)</target>'
    targets = re.findall(target_pattern, text)
    for target in targets:
        cnt_match = re.search(r'<cnt>(\d+)</cnt>', target)
        obj_match = re.search(r'<obj>([^<]+)</obj>', target)
        if cnt_match and obj_match:
            result['targets'].append({
                'count': int(cnt_match.group(1)),
                'object': obj_match.group(1).lower().strip()
            })
    
    return result


def create_camera_prior_mask(
    camera_dir: str,
    bev_h: int,
    bev_w: int,
    pc_range: List[float],
) -> np.ndarray:
    """Create a prior mask for the specified camera direction.
    
    Args:
        camera_dir: Camera direction string (e.g., 'front left')
        bev_h, bev_w: BEV grid dimensions
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
    
    Returns:
        Binary mask of shape (bev_h, bev_w)
    """
    camera_dir = camera_dir.lower().strip()
    
    # Create coordinate grids
    x_range = np.linspace(pc_range[0], pc_range[3], bev_w)  # forward
    y_range = np.linspace(pc_range[1], pc_range[4], bev_h)  # left
    
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Calculate angle from ego vehicle (forward is 0 degrees)
    angles = np.arctan2(yy, xx) * 180 / np.pi
    
    mask = np.zeros((bev_h, bev_w), dtype=np.float32)
    
    if camera_dir in CAMERA_SECTORS:
        angle_range = CAMERA_SECTORS[camera_dir]['angle_range']
        
        if camera_dir == 'back':
            # Special case: back covers both sides of ±180
            mask[(angles >= 150) | (angles <= -150)] = 1.0
        else:
            mask[(angles >= angle_range[0]) & (angles <= angle_range[1])] = 1.0
    else:
        # Default: uniform mask
        mask[:] = 1.0 / (bev_h * bev_w)
    
    return mask


@DATASETS.register_module()
class NuScenesMQADataset(CustomNuScenesDataset):
    """NuScenes dataset with MQA (Markup Question-Answering) annotations.
    
    This dataset extends CustomNuScenesDataset to include:
    - Question text with markup tags
    - Answer text with structured labels
    - Prior masks based on camera directions
    - Multi-task labels (count, object class, distance, location)
    """
    
    QUESTION_TYPES = [
        'object_presence_confirmation',      # Count specific object in camera
        'important_object_count_and_direction',  # All objects in camera direction
        'object_location_coordinates',       # (x, y) of nearest object
        'relative_distance_to_vehicles',     # Distance to nearest object
    ]
    
    def __init__(
        self,
        mqa_ann_file: str,
        max_count: int = 20,
        num_object_classes: int = 13,
        use_camera_prior: bool = True,
        *args,
        **kwargs
    ):
        """Initialize NuScenes-MQA Dataset.
        
        Args:
            mqa_ann_file: Path to MQA CSV annotation file
            max_count: Maximum count for classification (0 to max_count)
            num_object_classes: Number of object classes
            use_camera_prior: Whether to generate camera-based prior masks
        """
        # Initialize valid_mqa_samples before super().__init__()
        # because parent class may call __len__ during initialization
        self.valid_mqa_samples = []
        self.mqa_annotations = []
        self.token_to_idx = {}
        
        self.mqa_ann_file = mqa_ann_file
        self.max_count = max_count
        self.num_object_classes = num_object_classes
        self.use_camera_prior = use_camera_prior
        
        super().__init__(*args, **kwargs)
        
        # Load MQA annotations
        self.mqa_annotations = self._load_mqa_annotations()
        
        # Build sample_token to data_info index mapping
        self.token_to_idx = {
            info['token']: idx for idx, info in enumerate(self.data_infos)
        }
        
        # Filter MQA samples that exist in nuScenes data
        self.valid_mqa_samples = self._filter_valid_samples()

        # mmdet's (Distributed)GroupSampler relies on `dataset.flag`.
        # Some 3D datasets don't set it, or it may be empty after init.
        # For MQA we don't need grouping, so use a single group for all samples.
        if not hasattr(self, 'flag') or getattr(self.flag, 'size', 0) != len(self):
            self.flag = np.zeros(len(self), dtype=np.uint8)
        
        print(f"Loaded {len(self.valid_mqa_samples)} valid MQA samples "
              f"from {len(self.mqa_annotations)} total annotations")
    
    def _load_mqa_annotations(self) -> List[Dict]:
        """Load MQA annotations from CSV file."""
        annotations = []
        with open(self.mqa_ann_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations.append({
                    'sample_token': row['sample_token'],
                    'question': row['question'],
                    'answer': row['answer'],
                    'question_type': row['question_type'],
                })
        return annotations
    
    def _filter_valid_samples(self) -> List[int]:
        """Filter MQA samples that have corresponding nuScenes data."""
        valid_indices = []
        for idx, ann in enumerate(self.mqa_annotations):
            if ann['sample_token'] in self.token_to_idx:
                valid_indices.append(idx)
        return valid_indices
    
    def __len__(self) -> int:
        """Return number of valid MQA samples."""
        return len(self.valid_mqa_samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item by index.
        
        Note: idx is an index into valid_mqa_samples, not data_infos.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        
        while True:
            data = self.prepare_train_data(idx)
            if data is not None:
                return data
            # If data is None, try next sample
            idx = (idx + 1) % len(self)
    
    def _parse_mqa_labels(self, mqa_ann: Dict) -> Dict:
        """Parse MQA annotation to extract structured labels.
        
        Returns:
            Dict containing:
                - question_type_id: int
                - camera_dirs: List[str]
                - target_objects: List[Dict] with 'class_id' and 'count'
                - location: Optional[Tuple[float, float]]
                - distance: Optional[float]
                - prior_mask: Optional[np.ndarray]
        """
        question = mqa_ann['question']
        answer = mqa_ann['answer'].split(':')[0]  # Take first answer variant
        question_type = mqa_ann['question_type']
        
        # Parse markup from question and answer
        q_parsed = parse_markup(question)
        a_parsed = parse_markup(answer)
        
        # Get camera directions
        camera_dirs = q_parsed['cameras'] if q_parsed['cameras'] else a_parsed['cameras']
        
        # Parse target objects
        target_objects = []
        for target in a_parsed['targets']:
            obj_name = target['object']
            class_id = OBJECT_CLASSES.get(obj_name, -1)
            target_objects.append({
                'class_id': class_id,
                'count': min(target['count'], self.max_count),
                'name': obj_name,
            })
        
        # If no targets parsed, try to get from obj/cnt tags
        if not target_objects and a_parsed['objects'] and a_parsed['counts']:
            for obj, cnt in zip(a_parsed['objects'], a_parsed['counts']):
                obj_name = obj.lower().strip()
                class_id = OBJECT_CLASSES.get(obj_name, -1)
                target_objects.append({
                    'class_id': class_id,
                    'count': min(cnt, self.max_count),
                    'name': obj_name,
                })
        
        # Get location and distance
        location = a_parsed['locations'][0] if a_parsed['locations'] else None
        distance = a_parsed['distances'][0] if a_parsed['distances'] else None
        
        # Question type ID
        question_type_id = self.QUESTION_TYPES.index(question_type) \
            if question_type in self.QUESTION_TYPES else -1
        
        return {
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'question_type_id': question_type_id,
            'camera_dirs': camera_dirs,
            'target_objects': target_objects,
            'location': location,
            'distance': distance,
        }
    
    def get_data_info(self, index: int) -> Optional[Dict]:
        """Get data info for MQA sample."""
        # Get MQA annotation
        mqa_idx = self.valid_mqa_samples[index]
        mqa_ann = self.mqa_annotations[mqa_idx]
        
        # Get corresponding nuScenes sample index
        sample_token = mqa_ann['sample_token']
        nus_idx = self.token_to_idx[sample_token]
        
        # Get base data info from parent class
        input_dict = super().get_data_info(nus_idx)
        if input_dict is None:
            return None
        
        # Parse MQA labels
        mqa_labels = self._parse_mqa_labels(mqa_ann)
        
        # Add MQA specific fields
        input_dict['question'] = mqa_labels['question']
        input_dict['answer'] = mqa_labels['answer']
        input_dict['question_type'] = mqa_labels['question_type']
        input_dict['question_type_id'] = mqa_labels['question_type_id']
        input_dict['camera_dirs'] = mqa_labels['camera_dirs']
        input_dict['target_objects'] = mqa_labels['target_objects']
        input_dict['location'] = mqa_labels['location']
        input_dict['distance'] = mqa_labels['distance']
        
        return input_dict
    
    def prepare_train_data(self, index: int) -> Optional[Dict]:
        """Prepare training data for MQA.
        
        For MQA, we use single frame (no queue) as the question is about
        the current frame only.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        if example is None:
            return None
        
        # Add MQA labels to example
        example['question'] = DC(input_dict['question'], cpu_only=True)
        example['question_type_id'] = DC(
            torch.tensor(input_dict['question_type_id']), cpu_only=False, stack=True, pad_dims=None)
        
        # Process target objects into tensors
        # Create count tensor: [num_classes] where each entry is the count
        count_tensor = torch.zeros(self.num_object_classes, dtype=torch.long)
        for target in input_dict['target_objects']:
            if target['class_id'] >= 0 and target['class_id'] < self.num_object_classes:
                count_tensor[target['class_id']] = target['count']
        example['target_counts'] = DC(count_tensor, cpu_only=False, stack=True, pad_dims=None)
        
        # Total count for simple counting task
        total_count = sum(t['count'] for t in input_dict['target_objects'])
        example['total_count'] = DC(
            torch.tensor(min(total_count, self.max_count), dtype=torch.long),
            cpu_only=False, stack=True, pad_dims=None)
        
        # Primary object class (first target object)
        if input_dict['target_objects']:
            primary_class = input_dict['target_objects'][0]['class_id']
            primary_count = input_dict['target_objects'][0]['count']
        else:
            primary_class = -1
            primary_count = 0
        example['primary_object_class'] = DC(
            torch.tensor(primary_class, dtype=torch.long), cpu_only=False, stack=True, pad_dims=None)
        example['primary_object_count'] = DC(
            torch.tensor(min(primary_count, self.max_count), dtype=torch.long),
            cpu_only=False, stack=True, pad_dims=None)
        
        # Location and distance (for regression tasks)
        if input_dict['location'] is not None:
            location = torch.tensor(input_dict['location'], dtype=torch.float32)
        else:
            location = torch.tensor([0.0, 0.0], dtype=torch.float32)
        example['target_location'] = DC(location, cpu_only=False, stack=True, pad_dims=None)
        example['has_location'] = DC(
            torch.tensor(input_dict['location'] is not None, dtype=torch.bool),
            cpu_only=False, stack=True, pad_dims=None)
        
        if input_dict['distance'] is not None:
            distance = torch.tensor([input_dict['distance']], dtype=torch.float32)
        else:
            distance = torch.tensor([0.0], dtype=torch.float32)
        example['target_distance'] = DC(distance, cpu_only=False, stack=True, pad_dims=None)
        example['has_distance'] = DC(
            torch.tensor(input_dict['distance'] is not None, dtype=torch.bool),
            cpu_only=False, stack=True, pad_dims=None)
        
        # Create prior mask based on camera direction
        if self.use_camera_prior and input_dict['camera_dirs']:
            camera_dir = input_dict['camera_dirs'][0]
            prior_mask = create_camera_prior_mask(
                camera_dir,
                self.bev_size[0],
                self.bev_size[1],
                self.pc_range if hasattr(self, 'pc_range') else 
                    [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            )
            example['camera_prior_mask'] = DC(
                torch.from_numpy(prior_mask), cpu_only=False, stack=True, pad_dims=None)
        else:
            example['camera_prior_mask'] = DC(
                torch.ones(self.bev_size, dtype=torch.float32) / 
                (self.bev_size[0] * self.bev_size[1]),
                cpu_only=False, stack=True, pad_dims=None)
        
        # Camera direction ID for text encoding
        camera_dir_id = -1
        if input_dict['camera_dirs']:
            cam_dir = input_dict['camera_dirs'][0].lower().strip()
            cam_names = list(CAMERA_SECTORS.keys())
            if cam_dir in cam_names:
                camera_dir_id = cam_names.index(cam_dir)
        example['camera_dir_id'] = DC(
            torch.tensor(camera_dir_id, dtype=torch.long), cpu_only=False, stack=True, pad_dims=None)
        
        return example
    
    def prepare_test_data(self, index: int) -> Optional[Dict]:
        """Prepare test data - similar to train but without GT."""
        return self.prepare_train_data(index)
    
    def evaluate(
        self,
        results: List[Dict],
        metric: str = 'accuracy',
        logger=None,
        **kwargs
    ) -> Dict:
        """Evaluate MQA predictions.
        
        Args:
            results: List of prediction dicts with keys:
                - count_pred: Predicted counts
                - class_pred: Predicted object classes
                - location_pred: Predicted locations
                - distance_pred: Predicted distances
        
        Returns:
            Dict of evaluation metrics
        """
        metrics = {}
        
        # Collect ground truth
        count_correct = 0
        count_total = 0
        class_correct = 0
        class_total = 0
        location_errors = []
        distance_errors = []
        
        for idx, result in enumerate(results):
            mqa_idx = self.valid_mqa_samples[idx]
            mqa_ann = self.mqa_annotations[mqa_idx]
            mqa_labels = self._parse_mqa_labels(mqa_ann)
            
            # Count accuracy
            if 'count_pred' in result and mqa_labels['target_objects']:
                gt_count = sum(t['count'] for t in mqa_labels['target_objects'])
                pred_count = result['count_pred']
                if gt_count == pred_count:
                    count_correct += 1
                count_total += 1
            
            # Class accuracy
            if 'class_pred' in result and mqa_labels['target_objects']:
                gt_class = mqa_labels['target_objects'][0]['class_id']
                pred_class = result['class_pred']
                if gt_class == pred_class:
                    class_correct += 1
                class_total += 1
            
            # Location error
            if 'location_pred' in result and mqa_labels['location'] is not None:
                gt_loc = np.array(mqa_labels['location'])
                pred_loc = np.array(result['location_pred'])
                location_errors.append(np.linalg.norm(gt_loc - pred_loc))
            
            # Distance error
            if 'distance_pred' in result and mqa_labels['distance'] is not None:
                gt_dist = mqa_labels['distance']
                pred_dist = result['distance_pred']
                distance_errors.append(abs(gt_dist - pred_dist))
        
        # Compute final metrics
        if count_total > 0:
            metrics['count_accuracy'] = count_correct / count_total
        if class_total > 0:
            metrics['class_accuracy'] = class_correct / class_total
        if location_errors:
            metrics['location_mae'] = np.mean(location_errors)
        if distance_errors:
            metrics['distance_mae'] = np.mean(distance_errors)
        
        return metrics
