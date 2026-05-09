import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.tools.rotation import mat_to_quat
def reduction_batch_based(image_loss, M):
    
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return torch.sum(image_loss) * 0.0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based, frame_id_mask=None):
    # mask for distinguish different frames
    valid_id_mask_x = torch.ones_like(mask[:, :, 1:])
    valid_id_mask_y = torch.ones_like(mask[:, 1:, :])
    if frame_id_mask is not None:
        valid_id_mask_x = ((frame_id_mask[:, :, 1:] - frame_id_mask[:, :, :-1]) == 0).to(mask.dtype)
        valid_id_mask_y = ((frame_id_mask[:, 1:, :] - frame_id_mask[:, :-1, :]) == 0).to(mask.dtype)
    
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(torch.mul(mask[:, :, 1:], mask[:, :, :-1]), valid_id_mask_x)
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(torch.mul(mask[:, 1:, :], mask[:, :-1, :]), valid_id_mask_y)
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)

def normalize_prediction_robust(target, mask, ms=None):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    if ms is None:
        m = torch.zeros_like(ssum)
        s = torch.ones_like(ssum)

        m[valid] = torch.median((mask[valid] * target[valid]).view(valid.sum(), -1), dim=1).values
    else:
        m, s = ms

    target = target - m.view(-1, 1, 1)

    if ms is None:
        sq = torch.sum(mask * target.abs(), (1, 2))
        s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1)), (m.detach(), s.detach())


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid]
                  * b_1[valid]) / (det[valid] + 1e-6)
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid]
                  * b_1[valid]) / (det[valid] + 1e-6)

    return x_0, x_1

class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.__data_loss = TrimmedMAELoss(reduction=reduction, trim=trim)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None
        self.__prediction_median_scale = None
        self.__target_median_scale = None

    def forward(self, prediction, target, mask, pred_ms=None, tar_ms=None, num_frame_h=1, no_norm=False):
        if no_norm:
            self.__prediction_ssi, self.__prediction_median_scale = prediction, (0, 1)
            target_, self.__target_median_scale = target, (0, 1)
        else:
            self.__prediction_ssi, self.__prediction_median_scale = normalize_prediction_robust(prediction, mask, ms=pred_ms)
            target_, self.__target_median_scale = normalize_prediction_robust(target, mask, ms=tar_ms)

        ssi = self.__data_loss(self.__prediction_ssi, target_, mask)
        if self.__alpha > 0:
            gm =  self.__regularization_loss(
                self.__prediction_ssi, target_, mask, num_frame_h=num_frame_h
            )
        else:
            gm=0
        total=ssi+self.__alpha*gm
        return total,ssi,gm

    def get_median_scale(self):
        return self.__prediction_median_scale, self.__target_median_scale

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class TrimmedMAELoss(nn.Module):
    def __init__(self, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.trim = trim

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask, weight_mask=None):
        if torch.sum(mask) == 0:
            return torch.sum(prediction) * 0.0
        M = torch.sum(mask, (1, 2))
        res = prediction - target
        if weight_mask is not None:
            res = res * weight_mask
        res = res[mask.bool()].abs()
        trimmed, _ = torch.sort(res.view(-1), descending=False)
        keep_num = int(len(res) * (1.0 - self.trim))
        if keep_num <= 0:
            return torch.sum(prediction) * 0.0
        trimmed = trimmed[: keep_num]

        return self.__reduction(trimmed, M)

    
class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask, num_frame_h=1):
        total = 0

        frame_id_mask = None
        if num_frame_h > 1:
            frame_h = mask.shape[1] // num_frame_h
            frame_id_mask = torch.zeros_like(mask)
            for i in range(num_frame_h):
                frame_id_mask[:, i*frame_h:(i+1)*frame_h, :] = i+1

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
                frame_id_mask=frame_id_mask[:, ::step, ::step] if num_frame_h > 1 else None,
            )

        return total


class TemporalGradientMatchingLoss(nn.Module):
    def __init__(self, trim=0.2, temp_grad_scales=4, temp_grad_decay=0.5, reduction="batch-based", diff_depth_th=0.05):
        super().__init__()

        self.data_loss = TrimmedMAELoss(trim=trim, reduction=reduction)
        self.temp_grad_scales = temp_grad_scales
        self.temp_grad_decay = temp_grad_decay
        self.diff_depth_th = diff_depth_th

    def forward(self, prediction, target, mask):
        '''
            prediction: Shape(B, T, H, W)
            target: Shape(B, T, H, W)
            mask: Shape(B, T, H, W)
        '''
        total = 0
        cnt = 0
        mask=mask.bool()
        min_target = torch.where(mask.bool(), target, torch.inf).min(-1).values.min(-1).values
        max_target = torch.where(mask.bool(), target, -torch.inf).max(-1).values.max(-1).values
        target_th = (max_target - min_target) * self.diff_depth_th

        for scale in range(self.temp_grad_scales):
            temp_stride = pow(2, scale)
            if temp_stride < prediction.shape[1]:
                pred_temp_grad = torch.diff(prediction[:,::temp_stride,...], dim=1)
                target_temp_grad = torch.diff(target[:,::temp_stride,...], dim=1)
                temp_mask = mask[:,::temp_stride,...][:,1:,...] & mask[:,::temp_stride,...][:,:-1,...]
                
                valid_mask_from_target_th = target_temp_grad.abs() < target_th.unsqueeze(-1).unsqueeze(-1)[:,::temp_stride,...][:,1:,...]
                temp_mask = temp_mask & valid_mask_from_target_th

                total += self.data_loss(prediction=pred_temp_grad.flatten(0, 1), target=target_temp_grad.flatten(0, 1), mask=temp_mask.flatten(0, 1)) * pow(self.temp_grad_decay, scale)
                cnt += 1

        return total / cnt

class compute_camera_loss(nn.Module):
    def __init__(self, loss_type="l1", gamma=0.6, pose_encoding_type="absT_quaR_FoV", weight_trans=1.0, weight_rot=1.0, weight_focal=0.5):
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma
        self.pose_encoding_type = pose_encoding_type
        self.weight_trans = weight_trans
        self.weight_rot = weight_rot
        self.weight_focal = weight_focal

    def check_and_fix_inf_nan(self, input_tensor, loss_name="default", hard_max=100):
        """
        Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
        
        Args:
            input_tensor (torch.Tensor): The loss tensor to check and fix.
            loss_name (str): Name of the loss (for diagnostic prints).
            hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                    [-hard_max, hard_max] will be clamped. If None, 
                                    no clamping is performed. Defaults to 100.
        """
        if input_tensor is None:
            return input_tensor
        
        # Check for inf/nan values
        has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()

        # Apply hard clamping if specified
        if hard_max is not None:
            input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

        return input_tensor

    def camera_loss_single(self,pred_pose_enc, gt_pose_enc, loss_type):
        """
        Computes translation, rotation, and focal loss for a batch of pose encodings.
        
        Args:
            pred_pose_enc: (N, D) predicted pose encoding
            gt_pose_enc: (N, D) ground truth pose encoding
            loss_type: "l1" (abs error) or "l2" (euclidean error)
        Returns:
            loss_T: translation loss (mean)
            loss_R: rotation loss (mean)
            loss_FL: focal length/intrinsics loss (mean)
        
        NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
            So here we use l1 loss.
        """
        if loss_type == "l1":
            # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
            loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
            loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
            loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
        elif loss_type == "l2":
            # L2 norm for each component
            loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
            loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
            loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Check/fix numerical issues (nan/inf) for each loss component
        loss_T = self.check_and_fix_inf_nan(loss_T, "loss_T")
        loss_R = self.check_and_fix_inf_nan(loss_R, "loss_R")
        loss_FL = self.check_and_fix_inf_nan(loss_FL, "loss_FL")

        # Clamp outlier translation loss to prevent instability, then average
        loss_T = loss_T.clamp(max=100).mean()
        loss_R = loss_R.mean()
        loss_FL = loss_FL.mean()

        return loss_T, loss_R, loss_FL
    
    def extri_intri_to_pose_encoding(
        self,
        extrinsics,
        intrinsics,
        image_size_hw=None, 
        pose_encoding_type=None
    ):
        """Convert camera extrinsics and intrinsics to a compact pose encoding.

        This function transforms camera parameters into a unified pose encoding format,
        which can be used for various downstream tasks like pose prediction or representation.

        Args:
            extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
                where B is batch size and S is sequence length.
                In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
                The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
            intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
                Defined in pixels, with format:
                [[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]]
                where fx, fy are focal lengths and (cx, cy) is the principal point
            image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
                Required for computing field of view values. For example: (256, 512).
            pose_encoding_type (str): Type of pose encoding to use. Currently only
                supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).

        Returns:
            torch.Tensor: Encoded camera pose parameters with shape BxSx9.
                For "absT_quaR_FoV" type, the 9 dimensions are:
                - [:3] = absolute translation vector T (3D)
                - [3:7] = rotation as quaternion quat (4D)
                - [7:] = field of view (2D)
        """

        # extrinsics: BxSx3x4
        # intrinsics: BxSx3x3

        if pose_encoding_type == "absT_quaR_FoV":
            R = extrinsics[..., :3, :3]  # BxSx3x3
            T = extrinsics[..., :3, 3]  # BxSx3

            quat = mat_to_quat(R)
            # Note the order of h and w here
            H, W = image_size_hw
            fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
            fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
            pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
        else:
            raise NotImplementedError

        return pose_encoding
    
    def forward(self, pose_encoding_pred, intrinsic_gt, extrinsic_gt, images, point_masks):
        # List of predicted pose encodings per stage
        pred_pose_encodings = pose_encoding_pred
        # Number of prediction stages
        n_stages = len(pred_pose_encodings)
        # Get ground truth camera extrinsics and intrinsics
        B,T,_,_=extrinsic_gt.shape
        trans = extrinsic_gt[..., :3, 3]
        scale = trans.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0] + 1e-6
        extrinsic_gt[..., :3, 3] /= scale
        gt_extrinsics = extrinsic_gt
        gt_intrinsics = intrinsic_gt.unsqueeze(1).repeat(1, T, 1, 1)
        image_hw = images.shape[-2:]
        valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
        # Encode ground truth pose to match predicted encoding format
        gt_pose_encoding = self.extri_intri_to_pose_encoding(
            gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=self.pose_encoding_type
        )

        # Initialize loss accumulators for translation, rotation, focal length
        total_loss_T = total_loss_R = total_loss_FL = 0

        # Compute loss for each prediction stage with temporal weighting
        for stage_idx in range(n_stages):
            # Later stages get higher weight (gamma^0 = 1.0 for final stage)
            stage_weight = self.gamma ** (n_stages - stage_idx - 1)
            pred_pose_stage = pred_pose_encodings[stage_idx]

            # Only consider valid frames for loss computation
            if valid_frame_mask.sum() == 0:
                # If no valid frames, set losses to zero to avoid gradient issues
                loss_T_stage = (pred_pose_stage * 0).mean()
                loss_R_stage = (pred_pose_stage * 0).mean()
                loss_FL_stage = (pred_pose_stage * 0).mean()
            else:
                # Only consider valid frames for loss computation
                loss_T_stage, loss_R_stage, loss_FL_stage = self.camera_loss_single(
                    pred_pose_stage[valid_frame_mask].clone(),
                    gt_pose_encoding[valid_frame_mask].clone(),
                    loss_type=self.loss_type
                )
            # Accumulate weighted losses across stages
            total_loss_T += loss_T_stage * stage_weight
            total_loss_R += loss_R_stage * stage_weight
            total_loss_FL += loss_FL_stage * stage_weight

        # Average over all stages
        avg_loss_T = total_loss_T / n_stages
        avg_loss_R = total_loss_R / n_stages
        avg_loss_FL = total_loss_FL / n_stages

        # Compute total weighted camera loss
        total_camera_loss = (
            avg_loss_T * self.weight_trans +
            avg_loss_R * self.weight_rot 
        )

        # Return loss dictionary with individual components
        return {
            "loss_camera": total_camera_loss,
            "loss_T": avg_loss_T,
            "loss_R": avg_loss_R,
        }
    
class Cameraloss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.compute_camera_loss=compute_camera_loss()

    def closed_form_inverse_se3(self,se3, R=None, T=None):
        """
        Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

        If `R` and `T` are provided, they must correspond to the rotation and translation
        components of `se3`. Otherwise, they will be extracted from `se3`.

        Args:
            se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
            R (optional): Nx3x3 array or tensor of rotation matrices.
            T (optional): Nx3x1 array or tensor of translation vectors.

        Returns:
            Inverted SE3 matrices with the same type and device as `se3`.

        Shapes:
            se3: (N, 4, 4)
            R: (N, 3, 3)
            T: (N, 3, 1)
        """
        # Check if se3 is a numpy array or a torch tensor
        is_numpy = isinstance(se3, np.ndarray)

        # Validate shapes
        if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
            raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

        # Extract R and T if not provided
        if R is None:
            R = se3[:, :3, :3]  # (N,3,3)
        if T is None:
            T = se3[:, :3, 3:]  # (N,3,1)

        # Transpose R
        if is_numpy:
            # Compute the transpose of the rotation for NumPy
            R_transposed = np.transpose(R, (0, 2, 1))
            # -R^T t for NumPy
            top_right = -np.matmul(R_transposed, T)
            inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
        else:
            R_transposed = R.transpose(1, 2)  # (N,3,3)
            top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
            inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
            inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

        inverted_matrix[:, :3, :3] = R_transposed
        inverted_matrix[:, :3, 3:] = top_right

        return inverted_matrix
    
    def forward(self, prediction, target, mask, extrinsic_gt, intrinsic_gt, extrinsic_pred, pose_enc_list):
        target = target.clone()
        B,T,H,W=prediction.shape
        prediction = torch.clamp(prediction, min=5e-3, max=1500)
        all_world_coords =[]
        all_world_coords_posepred =[]
        for i in range(B):
            intrinsic=intrinsic_gt[i]
            fu, fv = intrinsic[0, 0], intrinsic[1, 1]
            cu, cv = intrinsic[0, 2], intrinsic[1, 2]
            batch_coords = []
            batch_coords_posepred=[]
            for j in range(T):
                v, u = torch.meshgrid(
                    torch.arange(H, device=target.device),
                    torch.arange(W, device=target.device),
                    indexing='ij'   
                )
                x_cam = (u - cu) * target[i][j] / fu
                y_cam = (v - cv) * target[i][j] / fv
                z_cam = target[i][j]
                x_cam_posepred = (u - cu) * target[i][j] / fu
                y_cam_posepred = (v - cv) * target[i][j] / fv
                z_cam_posepred = target[i][j]
                cam_coords_points = torch.stack((x_cam, y_cam, z_cam), dim=-1).to(torch.float32)
                cam_coords_points_posepred = torch.stack((x_cam_posepred, y_cam_posepred, z_cam_posepred), dim=-1).to(torch.float32)
                cam_to_world_extrinsic = self.closed_form_inverse_se3(extrinsic_gt[i][j][None])[0]
                cam_to_world_extrinsic_pred = self.closed_form_inverse_se3(extrinsic_pred[i][j][None])[0]
                R_cam_to_world = cam_to_world_extrinsic[:3, :3]
                t_cam_to_world = cam_to_world_extrinsic[:3, 3]
                R_cam_to_world_pred = cam_to_world_extrinsic_pred[:3, :3]
                t_cam_to_world_pred = cam_to_world_extrinsic_pred[:3, 3]
                world_coords_points = (
                    torch.matmul(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
                )
                world_coords_points_posepred = (
                    torch.matmul(cam_coords_points_posepred, R_cam_to_world_pred.T) + t_cam_to_world_pred
                )
                batch_coords.append(world_coords_points.unsqueeze(0))
                batch_coords_posepred.append(world_coords_points_posepred.unsqueeze(0))
            batch_coords = torch.cat(batch_coords, dim=0)
            batch_coords_posepred = torch.cat(batch_coords_posepred, dim=0)
            all_world_coords.append(batch_coords.unsqueeze(0))
            all_world_coords_posepred.append(batch_coords_posepred.unsqueeze(0))
        final_world_coords = torch.cat(all_world_coords, dim=0)
        del all_world_coords,world_coords_points
        torch.cuda.empty_cache()
        R_0=extrinsic_gt[:, 0, :3, :3]
        T_0=extrinsic_gt[:, 0, :3, 3]
        new_world_points = (final_world_coords @ R_0.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + T_0.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * mask).sum(dim=[1,2,3])
        valid_count = mask.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
        T0_inv = self.closed_form_inverse_se3(extrinsic_gt[:, 0]).unsqueeze(1).expand(-1, T, -1, -1)
        extrinsic_gt = torch.matmul(extrinsic_gt, T0_inv)
        extrinsic_gt[:, :, :3, 3] = extrinsic_gt[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        camera_loss_dict =self.compute_camera_loss(pose_enc_list,intrinsic_gt,extrinsic_gt,target,mask)
        loss_camera=camera_loss_dict["loss_camera"]
        loss_T=camera_loss_dict['loss_T']
        loss_R=camera_loss_dict['loss_R']
        return loss_camera,loss_T,loss_R

class VideoDepthLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.2, scales=4, trim=0, stable_scale=10, reduction="batch-based", pose_flag=True):
        super().__init__()
        self.beta = beta
        self.spatial_loss = TrimmedProcrustesLoss(alpha=alpha, scales=scales, trim=trim, reduction=reduction)
        self.stable_loss = TemporalGradientMatchingLoss(trim=trim, reduction=reduction, temp_grad_decay=0.5, temp_grad_scales=1)
        self.camera_loss = Cameraloss()
        self.stable_scale = stable_scale
        self.data_loss = TrimmedMAELoss(trim=trim, reduction=reduction)
        self.pose_flag = pose_flag
           
    def forward(self, prediction, target, mask,intrinsic_gt,extrinsic_gt,pose_enc_list,extrinsic_pred):
        loss_dict = {}
        target = target.clone()
        target_inverse = torch.zeros_like(target)
        valid_mask=target>0
        target_inverse[valid_mask] = 1 /target[valid_mask]
        B,T,H,W=prediction.shape
        prediction = torch.clamp(prediction, min=5e-3, max=1500)
        extrinsic_gt=torch.stack(extrinsic_gt,dim=1)
        for i in range(B):
            pred=prediction[i]
            depth_gt=target[i]
            valid_mask=(torch.logical_and((depth_gt>1e-3), (depth_gt<400))).float()
            with torch.no_grad():
                gt_disp_masked = 1. / (depth_gt[valid_mask.bool()].reshape(-1, 1) + 1e-8)
                depth_pred = torch.clamp(pred, min=1e-3)
                pred_disp_masked = depth_pred[valid_mask.bool()].reshape(-1, 1)
                A = torch.cat([pred_disp_masked, torch.ones_like(pred_disp_masked)], dim=-1)
                X = torch.linalg.lstsq(A, gt_disp_masked).solution  # PyTorch 2.0+
                scale, shift = X[0].item(), X[1].item()
            aligned_pred = torch.clamp(scale * depth_pred + shift, min=1e-3)
            pred_depth = torch.where(aligned_pred > 0, 1.0 / aligned_pred, 0)
            pred_depth = torch.clamp(pred_depth, min=1e-3,max=400)
        K=torch.linalg.inv(intrinsic_gt)
        #compute loss
        total = 0
        #ssi, gm, tgm
        loss_dict['spatial_loss'],loss_dict['ssi'],loss_dict['gm'] = self.spatial_loss(prediction=prediction.flatten(0, 1), target=target_inverse.flatten(0, 1), mask=mask.flatten(0, 1).float())
        total += loss_dict['spatial_loss']
        scale, shift = compute_scale_and_shift(prediction.flatten(1,2), target_inverse.flatten(1,2), mask.flatten(1,2))
        prediction = scale.view(-1, 1, 1, 1) * prediction + shift.view(-1, 1, 1, 1)
        loss_dict['stable_loss'] = self.stable_loss(prediction=prediction, target=target_inverse, mask=mask) * self.stable_scale
        total += loss_dict['stable_loss']
        #camera_loss
        if self.pose_flag:
            loss_dict['pose_loss'],loss_dict['trans'],loss_dict['quat']=self.camera_loss(prediction, target, mask, extrinsic_gt, intrinsic_gt, extrinsic_pred, pose_enc_list)
            total += loss_dict['pose_loss'] * self.beta
        loss_dict['total_loss'] = total
        return loss_dict

