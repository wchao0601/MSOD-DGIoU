import torch
import math

def DGIoU(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Diagonal-Geometry Intersection over Union (DGIoU) between boxes.
    
    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. 
                              If False, input boxes are in (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: DGIoU values with shape (n,) depending on the specified flags.
    """
    
    # ==================== Box Format Conversion ====================
    if xywh:  # Convert from (x, y, w, h) to (x1, y1, x2, y2)
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_half, h1_half, w2_half, h2_half = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_half, x1 + w1_half, y1 - h1_half, y1 + h1_half
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_half, x2 + w2_half, y2 - h2_half, y2 + h2_half
    else:  # Already in (x1, y1, x2, y2) format
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(min=eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(min=eps)
    
    # ==================== Dynamic Sampling Parameters ====================
    num_samples = 10  # Number of sampling points along box boundaries
    alpha = 0.95      # Radius scaling factor
    intersection_sum = 0.0  # Sum of intersection distances
    union_sum = 0.0         # Sum of union distances
    base_iou = 0.0          # Base IoU value
    
    # Calculate box dimensions
    dx_pred, dy_pred = (b1_x2 - b1_x1), (b1_y2 - b1_y1)
    dx_gt, dy_gt = (b2_x2 - b2_x1), (b2_y2 - b2_y1)
    
    # ==================== Dynamic Distance Calculation ====================
    # Calculate distances between top-left and bottom-right corners
    dist_top = torch.sqrt((b1_x1 - b2_x1 + eps).pow(2) + 
                         (b1_y1 - b2_y1 + eps).pow(2) + 1)
    dist_bottom = torch.sqrt((b1_x2 - b2_x2 + eps).pow(2) + 
                            (b1_y2 - b2_y2 + eps).pow(2) + 1)
    
    # Calculate radius for dynamic sampling
    radius = torch.max(dist_bottom, dist_top) + 1
    total_distance = dist_top + dist_bottom
    
    # Calculate step sizes for top and bottom segments
    dx_pred_top = dx_pred * dist_top / total_distance
    dy_pred_top = dy_pred * dist_top / total_distance
    dx_gt_top = dx_gt * dist_top / total_distance
    dy_gt_top = dy_gt * dist_top / total_distance
    
    dx_pred_bottom = dx_pred * dist_bottom / total_distance
    dy_pred_bottom = dy_pred * dist_bottom / total_distance
    dx_gt_bottom = dx_gt * dist_bottom / total_distance
    dy_gt_bottom = dy_gt * dist_bottom / total_distance
    
    # Adjust radius with scaling factor
    adjusted_radius = radius / alpha
    mid_point = num_samples // 2
    
    # ==================== Dynamic Sampling Loop ====================
    for i in range(num_samples + 1):
        if i <= mid_point:
            # Sample along top segment
            line_x1 = b1_x1 + dx_pred_top * i / mid_point
            line_y1 = b1_y1 + dy_pred_top * i / mid_point
            line_x2 = b2_x1 + dx_gt_top * i / mid_point
            line_y2 = b2_y1 + dy_gt_top * i / mid_point
        else:
            # Sample along bottom segment
            line_x1 = b1_x1 + dx_pred_top + dx_pred_bottom * (i - mid_point) / mid_point
            line_y1 = b1_y1 + dy_pred_top + dy_pred_bottom * (i - mid_point) / mid_point
            line_x2 = b2_x1 + dx_gt_top + dx_gt_bottom * (i - mid_point) / mid_point
            line_y2 = b2_y1 + dy_gt_top + dy_gt_bottom * (i - mid_point) / mid_point
        
        # Calculate distance between sampled points
        point_distance = torch.sqrt((line_x1 - line_x2 + eps).pow(2) + 
                                  (line_y1 - line_y2 + eps).pow(2) + eps)
        
        # Update intersection and union sums
        union_sum += (adjusted_radius * 2) + point_distance
        intersection_sum += (adjusted_radius * 2) - point_distance
    
    # ==================== Calculate DGIoU ====================
    dg_iou = base_iou + intersection_sum / (union_sum + eps)
    
    # ==================== Enhanced IoU Calculations ====================
    if CIoU or DIoU or GIoU:
        # Calculate smallest enclosing box (convex hull)
        convex_width = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        convex_height = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        
        if CIoU or DIoU:
            # Calculate convex diagonal squared
            convex_diag_sq = convex_width.pow(2) + convex_height.pow(2) + eps
            
            # Calculate center distance squared
            center_dist_sq = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + 
                            (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
            
            if CIoU:
                # Calculate aspect ratio consistency term
                aspect_ratio_term = (4 / math.pi**2) * (
                    (w2 / h2).atan() - (w1 / h1).atan()
                ).pow(2)
                
                # Calculate alpha parameter for CIoU
                with torch.no_grad():
                    alpha = aspect_ratio_term / (aspect_ratio_term - dg_iou + (1 + eps))
                
                # Return Complete DGIoU
                return dg_iou - (center_dist_sq / convex_diag_sq + 
                                    aspect_ratio_term * alpha)
            
            # Return Distance DGIoU
            return dg_iou - center_dist_sq / convex_diag_sq
        
        # Calculate Generalized DGIoU
        convex_area = convex_width * convex_height + eps
        
        # Calculate intersection area
        inter_width = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0)
        inter_height = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(min=0)
        intersection_area = inter_width * inter_height
        
        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area + eps
        
        # Return Generalized DGIoU
        return dg_iou - (convex_area - union_area) / convex_area
    
    # Return basic DGIoU
    return dg_iou

def main():
    box1 = torch.tensor([[0,0,80,100]])
    box2 = torch.tensor([[115,15,180,85]])
    dgiou = DGIoU(box1, box2, xywh=False, CIoU=False)
    print('DGIoU:',dgiou,'\n')

if __name__=='__main__' :
    main()