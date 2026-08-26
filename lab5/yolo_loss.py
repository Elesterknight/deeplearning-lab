import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BoxLoss(nn.Module):
    def __init__(self, loss_type='giou'):
        super(BoxLoss, self).__init__()
        self.type = loss_type
        # MSELoss should be defined once here
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred_boxes, target_boxes, anchors, obj_mask):
        """
        pred_boxes: [bsz, grid, grid, anchors, 4] (raw predictions tx, ty, tw, th)
        target_boxes: [bsz, grid, grid, anchors, 4] (encoded targets tx, ty, tw, th)
        anchors: list of (w, h) for the anchors at this scale
        obj_mask: [bsz, grid, grid, anchors] boolean mask for objects present
        """
        bsz, grid, _, num_anchors, _ = pred_boxes.size()
        device = pred_boxes.device
        dtype = pred_boxes.dtype

        if obj_mask.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # We only compute loss for the boxes responsible for detecting an object
        pred_boxes_obj = pred_boxes[obj_mask]
        target_boxes_obj = target_boxes[obj_mask]

        if self.type == 'giou':
            # Create grid offsets
            grid_range = torch.arange(grid, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(grid_range, grid_range, indexing='ij')
            
            # ############## BUG 修正處 ##############
            # Reshape grid offsets to [1, grid, grid, 1, 2] to allow broadcasting
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).view(1, grid, grid, 1, 2)
            
            # Create anchor tensors, shape [1, 1, 1, num_anchors, 2]
            anchors_tensor = torch.tensor(anchors, device=device, dtype=dtype).view(1, 1, 1, num_anchors, 2)
            
            # Decode predictions to be relative to the entire image
            # Broadcasting will handle the addition correctly
            pred_xy_decoded = (torch.sigmoid(pred_boxes[..., 0:2]) + grid_xy) / grid
            pred_wh_decoded = torch.exp(pred_boxes[..., 2:4]) * anchors_tensor
            pred_boxes_decoded = torch.cat([pred_xy_decoded, pred_wh_decoded], dim=-1)

            # Decode targets as well
            target_xy_decoded = (torch.sigmoid(target_boxes[..., 0:2]) + grid_xy) / grid
            target_wh_decoded = torch.exp(target_boxes[..., 2:4]) * anchors_tensor
            target_boxes_decoded = torch.cat([target_xy_decoded, target_wh_decoded], dim=-1)
            
            # Select only the decoded boxes where an object is present
            pred_decoded_obj = pred_boxes_decoded[obj_mask]
            target_decoded_obj = target_boxes_decoded[obj_mask]
            
            # ###########################################
            
            # Convert (center_x, center_y, w, h) to (x1, y1, x2, y2)
            p_x1 = pred_decoded_obj[:, 0] - pred_decoded_obj[:, 2] / 2
            p_y1 = pred_decoded_obj[:, 1] - pred_decoded_obj[:, 3] / 2
            p_x2 = pred_decoded_obj[:, 0] + pred_decoded_obj[:, 2] / 2
            p_y2 = pred_decoded_obj[:, 1] + pred_decoded_obj[:, 3] / 2
            
            t_x1 = target_decoded_obj[:, 0] - target_decoded_obj[:, 2] / 2
            t_y1 = target_decoded_obj[:, 1] - target_decoded_obj[:, 3] / 2
            t_x2 = target_decoded_obj[:, 0] + target_decoded_obj[:, 2] / 2
            t_y2 = target_decoded_obj[:, 1] + target_decoded_obj[:, 3] / 2

            # Intersection
            inter_x1 = torch.max(p_x1, t_x1)
            inter_y1 = torch.max(p_y1, t_y1)
            inter_x2 = torch.min(p_x2, t_x2)
            inter_y2 = torch.min(p_y2, t_y2)
            inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

            # Union
            p_area = pred_decoded_obj[:, 2] * pred_decoded_obj[:, 3]
            t_area = target_decoded_obj[:, 2] * target_decoded_obj[:, 3]
            union_area = p_area + t_area - inter_area

            # Smallest enclosing box
            c_x1 = torch.min(p_x1, t_x1)
            c_y1 = torch.min(p_y1, t_y1)
            c_x2 = torch.max(p_x2, t_x2)
            c_y2 = torch.max(p_y2, t_y2)
            c_area = torch.clamp(c_x2 - c_x1, min=0) * torch.clamp(c_y2 - c_y1, min=0)

            eps = 1e-7
            iou = inter_area / (union_area + eps)
            giou = iou - (c_area - union_area) / (c_area + eps)
            giou_loss = 1.0 - giou
            
            return giou_loss.sum()

        elif self.type == 'mse':
            # For MSE, we compute loss on the raw (encoded) coordinates, which is more stable
            loss_xy = self.mse_loss(pred_boxes_obj[..., 0:2], target_boxes_obj[..., 0:2])
            loss_wh = self.mse_loss(pred_boxes_obj[..., 2:4], target_boxes_obj[..., 2:4])
            box_loss = torch.sum(loss_xy) + torch.sum(loss_wh)
            return box_loss
        else:
            raise NotImplementedError(f"Box loss type '{self.type}' not implemented.")

class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        lambda_coord=2.0,
        lambda_obj=1.0,
        lambda_noobj=0.2,
        lambda_class=1.0,
        anchors=None,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none')
        self.box_loss = BoxLoss(loss_type='giou')
        self.anchors = anchors

    def forward(self, predictions, targets):
        device = predictions[0].device

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss_pos = torch.tensor(0.0, device=device)
        total_obj_loss_neg = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        total_num_pos = 0
        total_num_neg = 0

        for pred, gt, anchors in zip(predictions, targets, self.anchors):
            bsz, grid, _, num_anchors, num_classes_plus_5 = gt.shape
            num_classes = num_classes_plus_5 - 5
            
            pred = pred.view(bsz, grid, grid, num_anchors, -1)


            pred_box = pred[..., :4]          # (tx, ty, tw, th)
            pred_obj = pred[..., 4]          # objectness
            pred_cls = pred[..., 5:]          # class scores
            
            gt_box = gt[..., :4]
            gt_obj_mask = gt[..., 4].bool()  
            gt_cls = gt[..., 5:]
            

            num_pos = torch.sum(gt_obj_mask)
            num_neg = bsz * grid * grid * num_anchors - num_pos
            total_num_pos += num_pos
            total_num_neg += num_neg
            
            if num_pos == 0:
                
                noobj_loss = self.bce_loss(pred_obj, torch.zeros_like(pred_obj))
                total_obj_loss_neg += torch.sum(noobj_loss)
                continue

   
            box_loss_val = self.box_loss(pred_box, gt_box, anchors, gt_obj_mask)
            total_box_loss += box_loss_val
            
     
            obj_loss_pos = self.bce_loss(pred_obj[gt_obj_mask], torch.ones_like(pred_obj[gt_obj_mask]))
            total_obj_loss_pos += torch.sum(obj_loss_pos)

            gt_noobj_mask = ~gt_obj_mask
            obj_loss_neg = self.bce_loss(pred_obj[gt_noobj_mask], torch.zeros_like(pred_obj[gt_noobj_mask]))
            total_obj_loss_neg += torch.sum(obj_loss_neg)

            cls_loss = self.bce_loss(pred_cls[gt_obj_mask], gt_cls[gt_obj_mask])
            total_cls_loss += torch.sum(cls_loss)
            


        pos_denom = max(total_num_pos, 1)
        neg_denom = max(total_num_neg, 1)

        total_box_loss /= pos_denom
        total_obj_loss = total_obj_loss_pos / pos_denom
        total_noobj_loss = total_obj_loss_neg / neg_denom
        total_cls_loss /= pos_denom

        total_loss = (
            self.lambda_coord * total_box_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_noobj * total_noobj_loss +
            self.lambda_class * total_cls_loss
        )
        
        loss_dict = {
            'total': total_loss,
            'box': total_box_loss,
            'obj': total_obj_loss,
            'noobj': total_noobj_loss,
            'cls': total_cls_loss,
        }
        
        return loss_dict