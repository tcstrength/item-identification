from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class RectangleLabel(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    rotation: Optional[float] = None
    rectanglelabels: Optional[List[str]] = None

class AnnotationResult(BaseModel):
    original_width: Optional[int] = None
    original_height: Optional[int] = None
    image_rotation: Optional[int] = None
    value: Optional[RectangleLabel] = None
    id: Optional[str] = None
    from_name: Optional[str] = None
    to_name: Optional[str] = None
    type: Optional[str] = None
    origin: Optional[str] = None

class Annotation(BaseModel):
    id: Optional[int] = None
    result: Optional[List[AnnotationResult]] = None
    created_username: Optional[str] = None
    created_ago: Optional[str] = None
    completed_by: Optional[int] = None
    was_cancelled: Optional[bool] = None
    ground_truth: Optional[bool] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    draft_created_at: Optional[datetime] = None
    lead_time: Optional[float] = None
    import_id: Optional[int] = None
    last_action: Optional[Any] = None
    bulk_created: Optional[bool] = None
    task: Optional[int] = None
    project: Optional[int] = None
    updated_by: Optional[int] = None
    parent_prediction: Optional[Any] = None
    parent_annotation: Optional[Any] = None
    last_created_by: Optional[Any] = None

class LabelStudioTask(BaseModel):
    id: Optional[int] = None
    data: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_labeled: Optional[bool] = None
    overlap: Optional[int] = None
    inner_id: Optional[int] = None
    total_annotations: Optional[int] = None
    cancelled_annotations: Optional[int] = None
    total_predictions: Optional[int] = None
    comment_count: Optional[int] = None
    unresolved_comment_count: Optional[int] = None
    last_comment_updated_at: Optional[datetime] = None
    project: Optional[int] = None
    updated_by: Optional[int] = None
    file_upload: Optional[int] = None
    comment_authors: Optional[List[Any]] = None
    annotations: Optional[List[Annotation]] = None
    predictions: Optional[List[Any]] = None
