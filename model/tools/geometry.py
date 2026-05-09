from functools import partial
from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import einops as ein

def normalize_pose_translations(pose_translations, return_norm_factor=False):
    """
    Normalize the pose translations by the average norm of the non-zero pose translations.

    Args:
        pose_translations (torch.Tensor): Pose translations tensor of size [B, V, 3]. B is the batch size, V is the number of views.
    Returns:
        normalized_pose_translations (torch.Tensor): Normalized pose translations tensor of size [B, V, 3].
        norm_factor (torch.Tensor): Norm factor tensor of size B.
    """
    assert pose_translations.ndim == 3 and pose_translations.shape[2] == 3
    # Compute distance of all pose translations to origin
    pose_translations_dis = pose_translations.norm(dim=-1)  # [B, V]
    non_zero_pose_translations_dis = pose_translations_dis > 0  # [B, V]

    # Calculate the average norm of the translations across all views (considering only views with non-zero translations)
    sum_of_all_views_pose_translations = pose_translations_dis.sum(dim=1)  # [B]
    count_of_all_views_with_non_zero_pose_translations = (
        non_zero_pose_translations_dis.sum(dim=1)
    )  # [B]
    norm_factor = sum_of_all_views_pose_translations / (
        count_of_all_views_with_non_zero_pose_translations + 1e-8
    )  # [B]

    # Normalize the pose translations by the norm factor
    norm_factor = norm_factor.clip(min=1e-8)
    normalized_pose_translations = pose_translations / norm_factor.unsqueeze(
        -1
    ).unsqueeze(-1)

    # Create the output tuple
    output = (
        (normalized_pose_translations, norm_factor)
        if return_norm_factor
        else normalized_pose_translations
    )

    return output

def quaternion_inverse(quat):
    """
    Compute the inverse of a quaternion.

    Args:
        - quat: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - inv_quat: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
    """
    # Unsqueeze batch dimension if not present
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Compute the inverse
    quat_conj = quat.clone()
    quat_conj[:, :3] = -quat_conj[:, :3]
    quat_norm = torch.sum(quat * quat, dim=1, keepdim=True)
    inv_quat = quat_conj / quat_norm

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        inv_quat = inv_quat.squeeze(0)

    return inv_quat

def quaternion_to_rotation_matrix(quat):
    """
    Convert a quaternion into a 3x3 rotation matrix.

    Args:
        - quat: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - rot_matrix: 3x3 or Bx3x3 torch tensor
    """
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Ensure the quaternion is normalized
    quat = quat / quat.norm(dim=1, keepdim=True)
    x, y, z, w = quat.unbind(dim=1)

    # Compute the rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Construct the rotation matrix
    rot_matrix = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=1,
    ).view(-1, 3, 3)

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        rot_matrix = rot_matrix.squeeze(0)

    return rot_matrix

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.

    Args:
        - q1: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - q2: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - qm: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
    """
    # Unsqueeze batch dimension if not present
    if q1.dim() == 1:
        q1 = q1.unsqueeze(0)
        q2 = q2.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Unbind the quaternions
    x1, y1, z1, w1 = q1.unbind(dim=1)
    x2, y2, z2, w2 = q2.unbind(dim=1)

    # Compute the product
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    # Stack the components
    qm = torch.stack([x, y, z, w], dim=1)

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        qm = qm.squeeze(0)

    return qm

def transform_pose_using_quats_and_trans_2_to_1(quats1, trans1, quats2, trans2):
    """
    Transform quats and translation of pose2 from absolute frame (pose2 to world) to relative frame (pose2 to pose1).

    Args:
        - quats1: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - trans1: 3 or Bx3 torch tensor
        - quats2: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - trans2: 3 or Bx3 torch tensor

    Returns:
        - quats: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - trans: 3 or Bx3 torch tensor
    """
    # Unsqueeze batch dimension if not present
    if quats1.dim() == 1:
        quats1 = quats1.unsqueeze(0)
        trans1 = trans1.unsqueeze(0)
        quats2 = quats2.unsqueeze(0)
        trans2 = trans2.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Compute the inverse of view1's pose
    inv_quats1 = quaternion_inverse(quats1)
    R1_inv = quaternion_to_rotation_matrix(inv_quats1)
    t1_inv = -1 * ein.einsum(R1_inv, trans1, "b i j, b j -> b i")

    # Transform view2's pose to view1's frame
    quats = quaternion_multiply(inv_quats1, quats2)
    trans = ein.einsum(R1_inv, trans2, "b i j, b j -> b i") + t1_inv

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        quats = quats.squeeze(0)
        trans = trans.squeeze(0)

    return quats, trans

class GlobalRepresentationEncoder(nn.Module):
    "UniCeption Global Representation Encoder"

    def __init__(
        self,
        name: str,
        in_chans: int = 3,
        enc_embed_dim: int = 1024,
        intermediate_dims: List[int] = [128, 256, 512],
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-6),
        pretrained_checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Global Representation Encoder for projecting a global representation to a desired latent dimension.

        Args:
            name (str): Name of the Encoder.
            in_chans (int): Number of input channels.
            enc_embed_dim (int): Embedding dimension of the encoder.
            intermediate_dims (List[int]): List of intermediate dimensions of the encoder.
            act_layer (Type[nn.Module]): Activation layer to use in the encoder.
            norm_layer (Union[Type[nn.Module], Callable[..., nn.Module]]): Final normalization layer to use in the encoder.
            pretrained_checkpoint_path (Optional[str]): Path to pretrained checkpoint. (default: None)
        """
        super().__init__(*args, **kwargs)

        # Initialize the attributes
        self.name = name
        self.in_chans = in_chans
        self.enc_embed_dim = enc_embed_dim
        self.intermediate_dims = intermediate_dims
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        # Init the activation layer
        self.act_layer = act_layer()

        # Initialize the encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_chans, self.intermediate_dims[0]),
            self.act_layer,
        )
        for intermediate_idx in range(1, len(self.intermediate_dims)):
            self.encoder = nn.Sequential(
                self.encoder,
                nn.Linear(self.intermediate_dims[intermediate_idx - 1], self.intermediate_dims[intermediate_idx]),
                self.act_layer,
            )
        self.encoder = nn.Sequential(
            self.encoder,
            nn.Linear(self.intermediate_dims[-1], self.enc_embed_dim),
        )

        # Init weights of the final norm layer
        self.norm_layer = norm_layer(enc_embed_dim) if norm_layer else nn.Identity()
        if isinstance(self.norm_layer, nn.LayerNorm):
            nn.init.constant_(self.norm_layer.bias, 0)
            nn.init.constant_(self.norm_layer.weight, 1.0)

        # Load pretrained weights if provided
        if self.pretrained_checkpoint_path is not None:
            print(
                f"Loading pretrained Global Representation Encoder checkpoint from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, encoder_input):
        # Get the input data and verify the shape of the input
        input_data = encoder_input.data
        assert input_data.ndim == 2, "Input data must have shape (B, C)"
        assert input_data.shape[1] == self.in_chans, f"Input data must have {self.in_chans} channels"

        # Encode the global representation
        features = self.encoder(input_data)

        # Normalize the output
        features = self.norm_layer(features)

        return features


