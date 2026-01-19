import cv2
import math
import numpy as np
import os
import torch
import torch.nn.functional as F
import io
import torchvision.transforms.functional as TF
import re

from PIL import Image
from typing import Optional
from torchvision.utils import make_grid

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in Image.registered_extensions())


def scandir_images(dir, max_dataset_size=float("inf"), followlinks=True):
    """Get all image files from a directory and return a sorted list of fullpath.
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=followlinks)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return sorted(images[:min(max_dataset_size, len(images))])


def imread2pil(img_source, rgb=False):
    """Read image to tensor.

    Args:
        img_source (str, bytes, or PIL.Image): image filepath string, image contents as a bytearray or a PIL Image instance
        rgb: convert input to RGB if true
    """
    if type(img_source) == bytes:
        img = Image.open(io.BytesIO(img_source))
    elif type(img_source) == str:
        assert is_image_file(img_source), f'{img_source} is not a valid image file.'
        img = Image.open(img_source)
    elif isinstance(img_source, Image.Image):
        img = img_source
    else:
        raise Exception("Unsupported source type")
    if rgb:
        img = img.convert('RGB')
    return img


def imread2tensor(img_source, rgb=False):
    """Read image to tensor.

    Args:
        img_source (str, bytes, or PIL.Image): image filepath string, image contents as a bytearray or a PIL Image instance
        rgb: convert input to RGB if true
    """
    if type(img_source) == bytes:
        img = Image.open(io.BytesIO(img_source))
    elif type(img_source) == str:
        assert is_image_file(img_source), f'{img_source} is not a valid image file.'
        img = Image.open(img_source)
    elif isinstance(img_source, Image.Image):
        img = img_source
    else:
        raise Exception("Unsupported source type")
    if rgb:
        img = img.convert('RGB')
    img_tensor = TF.to_tensor(img)
    return img_tensor


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]

def yuvread2tensor(
    path: str,
    width: int,
    height: int,
    fmt: str = 'yuv420p',
    bitdepth: int = 8,
    frame_idx: int = 0,
    normalize: bool = False
) -> torch.Tensor:
    """
    Read one frame of raw YUV stream, return torch.Tensor, shape=(C, H, W):
      - fmt: 'yuv400p','yuv420p','yuv422p','yuv444p'
      - bitdepth: 8,10,12,14,16
      - normalize: Whether to normalize to [0,1]. False returns original pixel values (float32).

    Channel order same as imread2tensor: Y,U,V. YUV400 returns (1,H,W), others return (3,H,W).
    """
    # Bytes per pixel
    bps = 1 if bitdepth <= 8 else 2

    # Calculate pixel count for each plane
    y_sz = width * height
    if fmt == 'yuv400p':
        u_sz = v_sz = 0
    elif fmt == 'yuv420p':
        u_sz = v_sz = (width // 2) * (height // 2)
    elif fmt == 'yuv422p':
        u_sz = v_sz = (width // 2) * height
    elif fmt == 'yuv444p':
        u_sz = v_sz = width * height
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    frame_bytes = (y_sz + u_sz + v_sz) * bps

    # Locate and read
    with open(path, 'rb') as f:
        f.seek(frame_idx * frame_bytes, os.SEEK_SET)
        raw = f.read(frame_bytes)
        if len(raw) < frame_bytes:
            raise EOFError(f"Cannot read frame {frame_idx} from {path}, file too short.")

    # Split Y/U/V raw bytes
    off = 0
    y_raw = raw[off:off + y_sz * bps]; off += y_sz * bps
    u_raw = raw[off:off + u_sz * bps]; off += u_sz * bps
    v_raw = raw[off:off + v_sz * bps]

    # Convert to numpy
    dtype = np.uint8 if bps == 1 else np.uint16
    Y = np.frombuffer(y_raw, dtype=dtype).reshape((height, width))
    if fmt == 'yuv400p':
        arr = Y[np.newaxis, :, :].astype(np.float32)
    else:
        # U/V plane reshape
        if fmt == 'yuv420p':
            h2, w2 = height // 2, width // 2
        elif fmt == 'yuv422p':
            h2, w2 = height, width // 2
        else:  # 'yuv444p'
            h2, w2 = height, width

        U = np.frombuffer(u_raw, dtype=dtype).reshape((h2, w2))
        V = np.frombuffer(v_raw, dtype=dtype).reshape((h2, w2))

        # Upsample to full resolution
        if fmt == 'yuv420p':
            U = U.repeat(2, axis=0).repeat(2, axis=1)
            V = V.repeat(2, axis=0).repeat(2, axis=1)
        elif fmt == 'yuv422p':
            U = U.repeat(1, axis=0).repeat(2, axis=1)
            V = V.repeat(1, axis=0).repeat(2, axis=1)
        # yuv444p doesn't need upsampling

        # Concatenate channels and convert to float32
        arr = np.stack([Y, U, V], axis=2).transpose(2, 0, 1).astype(np.float32)

    # Whether to normalize
    if normalize:
        max_val = float((1 << bitdepth) - 1)
        arr = arr / max_val

    # Return torch.Tensor
    return torch.from_numpy(arr)

def parse_y4m_header(path: str):
    """
    Parse .y4m header, return width, height, fmt, bitdepth
    """
    with open(path, 'rb') as f:
        header = f.readline().decode('ascii', errors='ignore').strip()
    if not header.startswith('YUV4MPEG2'):
        raise ValueError(f"{path} is not a valid .y4m format")
    tokens = header.split()[1:]
    width = height = None
    fmt = None
    bitdepth = 8
    for tok in tokens:
        if tok.startswith('W'):
            width = int(tok[1:])
        elif tok.startswith('H'):
            height = int(tok[1:])
        elif tok.startswith('C'):
            c = tok[1:]
            if c.startswith('420'):
                fmt = 'yuv420p'
            elif c.startswith('422'):
                fmt = 'yuv422p'
            elif c.startswith('444'):
                fmt = 'yuv444p'
            elif c in ('mono', '400'):
                fmt = 'yuv400p'
            else:
                if '420' in c:
                    fmt = 'yuv420p'
                elif '422' in c:
                    fmt = 'yuv422p'
                elif '444' in c:
                    fmt = 'yuv444p'
                else:
                    raise ValueError(f"Unrecognized chroma format: C{c}")
            # Capture e.g. "420p10" or ending digits as bitdepth
            m = re.search(r'p(\d+)$', c)
            if m:
                bitdepth = int(m.group(1))
    if None in (width, height, fmt):
        raise ValueError("Could not get complete W/H/C information from .y4m header")
    return width, height, fmt, bitdepth

def y4mread2tensor(
    path: str,
    frame_idx: int = 0,
    normalize: bool = False
) -> torch.Tensor:
    """
    Read specified frame from .y4m file, return torch.Tensor, shape=(C, H, W):
      - Auto-parse width, height, fmt, bitdepth from file header
      - frame_idx: Frame index to read (0-based)
      - normalize: Whether to normalize pixel values to [0,1] (otherwise return original values float32)

    Channel order same as imread2tensor: Y,U,V. For yuv400p, returns (1,H,W), others (3,H,W).
    """
    # --- 1. Parse file header ---
    width, height, fmt, bitdepth = parse_y4m_header(path)

    # --- Read data ---
    with open(path, 'rb') as f:
        # --- 2. Calculate bytes per frame ---
        bps = 1 if bitdepth <= 8 else 2
        y_sz = width * height
        if fmt == 'yuv400p':
            u_sz = v_sz = 0
        elif fmt == 'yuv420p':
            u_sz = v_sz = (width // 2) * (height // 2)
        elif fmt == 'yuv422p':
            u_sz = v_sz = (width // 2) * height
        else:  # 'yuv444p'
            u_sz = v_sz = width * height

        frame_bytes = (y_sz + u_sz + v_sz) * bps

        # --- 3. Locate to frame_idx frame ---
        # Y4M frame has a "FRAME" marker before it
        for _ in range(frame_idx):
            line = f.readline()
            # If not FRAME, keep reading until found
            while line and not line.startswith(b'FRAME'):
                line = f.readline()
            # Skip the raw data after it
            f.seek(frame_bytes, os.SEEK_CUR)

        # Read FRAME marker of target frame
        line = f.readline()
        while line and not line.startswith(b'FRAME'):
            line = f.readline()
        # Read raw bytes
        raw = f.read(frame_bytes)
        if len(raw) < frame_bytes:
            raise EOFError(f"Cannot read frame {frame_idx} from {path} (file too short)")

    # --- 4. Split Y/U/V and upsample U/V ---
    off = 0
    y_raw = raw[off : off + y_sz * bps];  off += y_sz * bps
    u_raw = raw[off : off + u_sz * bps];  off += u_sz * bps
    v_raw = raw[off : off + v_sz * bps]

    dtype = np.uint8 if bps == 1 else np.uint16
    Y = np.frombuffer(y_raw, dtype=dtype).reshape((height, width))

    if fmt == 'yuv400p':
        arr = Y[np.newaxis, :, :].astype(np.float32)
    else:
        # U/V reshape
        if fmt == 'yuv420p':
            h2, w2 = height // 2, width // 2
        elif fmt == 'yuv422p':
            h2, w2 = height, width // 2
        else:  # yuv444p
            h2, w2 = height, width

        U = np.frombuffer(u_raw, dtype=dtype).reshape((h2, w2))
        V = np.frombuffer(v_raw, dtype=dtype).reshape((h2, w2))

        # Upsample to full resolution
        if fmt == 'yuv420p':
            U = U.repeat(2, axis=0).repeat(2, axis=1)
            V = V.repeat(2, axis=0).repeat(2, axis=1)
        elif fmt == 'yuv422p':
            U = U.repeat(1, axis=0).repeat(2, axis=1)
            V = V.repeat(1, axis=0).repeat(2, axis=1)
        # yuv444p doesn't need upsampling

        # Concatenate channels and convert to float32
        arr = np.stack([Y, U, V], axis=2).transpose(2, 0, 1).astype(np.float32)

    # --- 5. Optional normalization ---
    if normalize:
        max_val = float((1 << bitdepth) - 1)
        arr = arr / max_val

    # --- 6. Convert to torch.Tensor and return ---
    return torch.from_numpy(arr)


def tensor2yuv(
    tensor: torch.Tensor,
    output_path: str,
    fmt: str = 'yuv420p',
    bitdepth: int = 8,
    normalize: bool = False
) -> None:
    """
    Write a Tensor read from yuvread2tensor back to original YUV file.

    Args:
        tensor (torch.Tensor): shape = (C, H, W), C=1(yuv400p) or 3(others); dtype=float32,
                               if normalize=True, values in [0,1]; otherwise original pixel values.
        output_path (str): Target YUV file path, directory will be created automatically.
        fmt (str): 'yuv400p','yuv420p','yuv422p','yuv444p'.
        bitdepth (int): Bits per pixel, 8,10,12,14,16.
        normalize (bool): Whether input Tensor is normalized to [0,1]. If True, multiply by max_val to restore original range.

    Process:
      1. Derive H,W from tensor.shape; calculate bps, max_val;
      2. If normalize, multiply by max_val; otherwise treat as original pixel values;
      3. clamp + round then convert to uint8/uint16;
      4. Downsample U,V according to fmt (420→(2,2) block take top-left; 422→sample along even width indices; 444/400 unchanged);
      5. Concatenate bytes in Y,U,V order and write file.
    """
    # 1) Check
    if tensor.dim() != 3:
        raise ValueError(f'tensor must with shape (C,H,W), got {tensor.shape}')
    C, H, W = tensor.shape
    if fmt == 'yuv400p' and C != 1:
        raise ValueError('yuv400p requires C=1')
    if fmt != 'yuv400p' and C != 3:
        raise ValueError(f'{fmt} requires C=3')

    # 2) Prepare parameters
    bps = 1 if bitdepth <= 8 else 2
    max_val = (1 << bitdepth) - 1

    # 3) Restore integer pixels from tensor
    arr = tensor.clone().cpu()
    if normalize:
        arr = arr * float(max_val)
    arr = arr.clamp(0, max_val).round().to(torch.int64)

    # 4) Split channels and convert to numpy
    Y = arr[0].numpy().astype(np.uint16 if bps==2 else np.uint8)
    if fmt == 'yuv400p':
        U = V = None
    else:
        U_full = arr[1].numpy().astype(np.uint16 if bps==2 else np.uint8)
        V_full = arr[2].numpy().astype(np.uint16 if bps==2 else np.uint8)
        # Downsample
        if fmt == 'yuv420p':
            U = U_full[::2, ::2]
            V = V_full[::2, ::2]
        elif fmt == 'yuv422p':
            U = U_full[:, ::2]
            V = V_full[:, ::2]
        elif fmt == 'yuv444p':
            U = U_full
            V = V_full
        else:
            raise ValueError(f'Unsupported format: {fmt}')

    # 5) Generate bytes
    y_bytes = Y.tobytes()
    if fmt == 'yuv400p':
        uv_bytes = b''
    else:
        uv_bytes = U.tobytes() + V.tobytes()

    # 6) Write file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(y_bytes)
        f.write(uv_bytes)

def tensor2y4m(
    tensor: torch.Tensor,
    output_path: str,
    fmt: str = 'yuv420p',
    bitdepth: int = 8,
    normalize: bool = False,
    fps_num: int = 25,
    fps_den: int = 1,
    interlace: str = 'p',
    aspect_num: int = 0,
    aspect_den: int = 0
) -> None:
    """
    Write a Tensor read from y4mread2tensor()/yuvread2tensor() to a .y4m file (single frame).

    Args:
        tensor (torch.Tensor): shape = (C, H, W), C=1(yuv400p) or 3(others); dtype=float32.
                               If normalize=True, values in [0,1]; otherwise original pixel values.
        output_path (str): Target .y4m file path, directory will be created automatically.
        fmt (str): 'yuv400p','yuv420p','yuv422p','yuv444p'.
        bitdepth (int): Bits per pixel, 8,10,12,14,16.
        normalize (bool): Whether input Tensor is normalized to [0,1]. If True, multiply by max_val to restore original range.
        fps_num, fps_den (int): Frame rate numerator/denominator, default 25/1.
        interlace (str): Scanning method, 'p'|'t'|'b', default 'p' (progressive).
        aspect_num, aspect_den (int): Pixel aspect ratio, default 0:0.
    """
    # 1) Validate and extract dimensions
    if tensor.dim() != 3 or tensor.dtype != torch.float32:
        raise ValueError(f'tensor must be float32 with shape (C,H,W), got {tensor.shape},{tensor.dtype}')
    C, H, W = tensor.shape
    if fmt == 'yuv400p' and C != 1:
        raise ValueError('yuv400p requires C=1')
    if fmt != 'yuv400p' and C != 3:
        raise ValueError(f'{fmt} requires C=3')

    # 2) Calculate bps and max value
    bps = 1 if bitdepth <= 8 else 2
    max_val = (1 << bitdepth) - 1

    # 3) Restore integer pixels from tensor
    arr = tensor.clone().cpu()
    if normalize:
        arr = arr * float(max_val)
    arr = arr.clamp(0, max_val).round().to(torch.int64)

    # 4) Split channels and downsample U/V
    Y = arr[0].numpy().astype(np.uint16 if bps==2 else np.uint8)
    if fmt == 'yuv400p':
        U = V = None
    else:
        U_full = arr[1].numpy().astype(np.uint16 if bps==2 else np.uint8)
        V_full = arr[2].numpy().astype(np.uint16 if bps==2 else np.uint8)
        if fmt == 'yuv420p':
            U = U_full[::2, ::2]
            V = V_full[::2, ::2]
        elif fmt == 'yuv422p':
            U = U_full[:, ::2]
            V = V_full[:, ::2]
        elif fmt == 'yuv444p':
            U = U_full
            V = V_full
        else:
            raise ValueError(f'Unsupported format: {fmt}')

    # 5) Generate raw byte stream
    y_bytes = Y.tobytes()
    uv_bytes = b''
    if fmt != 'yuv400p':
        uv_bytes = U.tobytes() + V.tobytes()

    # 6) Construct Y4M header
    # C field
    if fmt == 'yuv400p':
        chroma = 'mono'
    else:
        base = fmt.replace('yuv', '')  # '420p','422p','444p'
        # Remove trailing 'p'
        if base.endswith('p'):
            base = base[:-1]
        # Has bitdepth >8
        if bitdepth > 8:
            chroma = f'{base}p{bitdepth}'
        else:
            chroma = base
    header = (
        f"YUV4MPEG2 W{W} H{H} F{fps_num}:{fps_den} I{interlace} "
        f"A{aspect_num}:{aspect_den} C{chroma}jpeg XYSCSS={chroma}JPEG\n"
    )

    # 7) Write file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        # 7.1 Write header
        f.write(header.encode('ascii'))
        # 7.2 Write single frame marker
        f.write(b"FRAME\n")
        # 7.3 Write Y/U/V data
        f.write(y_bytes)
        f.write(uv_bytes)

def inject_jnd(
        original : torch.Tensor,
        jnd_map  : torch.Tensor,
        channels : str = 'YUV',
        max_val  : Optional[float] = None,
        seed     : Optional[int]   = None,
        target   : Optional[int] = 0,
        block_sz : Optional[int] = 8
) -> torch.Tensor:
    """
    Inject / smooth pixels by JND.

    target = 0 : add random ±T_JND noise
    target = 1 : block-mean smoothing
    target = 2 : JND-bounded Gaussian smoothing
    target = 3 : JND-guided adaptive Wiener filtering
    target = 4 : JND-bounded Guided filtering (edge-preserving)
    """

    # 1) reproducible seed
    if seed is not None:
        torch.manual_seed(seed)

    # 2) shape check
    if original.shape != jnd_map.shape:
        raise ValueError(f"original {original.shape} vs jnd_map {jnd_map.shape}")

    C, H, W = original.shape
    ch = channels.upper()
    if ch not in ('Y', 'UV', 'YUV'):
        raise ValueError("channels must be 'Y', 'UV' or 'YUV'")

    # 3) build per-channel mask (broadcastable (C,1,1))
    if C == 3:
        if ch == 'Y':
            mask = torch.tensor([1, 0, 0], device=original.device,
                                dtype=original.dtype).view(3, 1, 1)
        elif ch == 'UV':
            mask = torch.tensor([0, 1, 1], device=original.device,
                                dtype=original.dtype).view(3, 1, 1)
        else:
            mask = torch.ones((3, 1, 1),  device=original.device,
                              dtype=original.dtype)
    else:  # fallback: non-3 channels → inject all
        mask = torch.ones((C, 1, 1), device=original.device,
                          dtype=original.dtype)

    # ------------------------------------------------------------------
    # target = 0  ——  Original noise injection scheme
    # ------------------------------------------------------------------
    if target == 0:
        r      = torch.rand_like(jnd_map) * 2.0 - 1.0   # uniform [-1,1]
        noise  = jnd_map * r * mask
        result = original + noise

    # ------------------------------------------------------------------
    # target = 1  ——  Block mean smoothing
    # ------------------------------------------------------------------
    elif target == 1:
        # Calculate block mean F̅_B; do same operation for selected channels
        # (1) unfold -> (C, blocks, patch) then average
        patches = original.unfold(1, block_sz, block_sz) \
                           .unfold(2, block_sz, block_sz)   # (C, nH, nW, b, b)

        block_mean = patches.mean(dim=(-1, -2))          # (C, nH, nW)
        F_B = block_mean.repeat_interleave(block_sz, dim=1) \
                         .repeat_interleave(block_sz, dim=2)  # → (C, H, W)

        # Only use block mean on selected channels, other channels unchanged
        F_B = F_B * mask + original * (1 - mask)

        # Smooth pixel by pixel according to equation (16)
        delta = original - F_B
        T     = jnd_map

        cond_lt = (delta < -T) & (mask.bool())
        cond_gt = (delta >  T) & (mask.bool())
        cond_mid= (delta.abs() <= T) & (mask.bool())

        result = original.clone()
        result[cond_lt]  = original[cond_lt]  + T[cond_lt]
        result[cond_mid] = F_B[cond_mid]                      # Set block mean
        result[cond_gt]  = original[cond_gt]  - T[cond_gt]

    # ------------------------------------------------------------------
    # target = 2  ——  JND-Bounded Gaussian Smoothing
    # ------------------------------------------------------------------
    elif target == 2:
        # Gaussian blur parameters
        sigma = 1.5
        k_sz = 5
        
        # Apply Gaussian Blur
        blurred = TF.gaussian_blur(original, kernel_size=[k_sz, k_sz], sigma=[sigma])
        
        # Determine the "noise" (difference)
        diff = original - blurred
        
        # Clamp the difference to be within JND range
        # This ensures we only remove high-frequency details that are below JND
        diff_clamped = diff.clamp(-jnd_map, jnd_map)
        
        # Apply only to selected channels
        diff_clamped = diff_clamped * mask
        
        result = original - diff_clamped

    # ------------------------------------------------------------------
    # target = 3  ——  Adaptive Wiener Filtering
    # ------------------------------------------------------------------
    elif target == 3:
        # Local mean and variance using Average Pooling
        k = 5
        pad = k // 2
        
        def box_filter(x, k, p):
            x_in = x.unsqueeze(0)
            x_out = F.avg_pool2d(x_in, kernel_size=k, stride=1, padding=p)
            return x_out.squeeze(0)

        mu = box_filter(original, k, pad)
        mu2 = box_filter(original**2, k, pad)
        var = (mu2 - mu**2).clamp(min=0.0)
        
        # Assume local noise variance limit is JND^2
        noise_var = jnd_map**2
        
        # Wiener filter formula: result = mu + (var - noise_var)/var * (original - mu)
        weight = (var - noise_var).clamp(min=0.0) / (var + 1e-5)
        
        smoothed = mu + weight * (original - mu)
        
        # Strictly bound the change by JND
        diff = smoothed - original
        diff = diff.clamp(-jnd_map, jnd_map)
        
        result = original + diff * mask

    # ------------------------------------------------------------------
    # target = 4  ——  Guided Filtering (Edge Preserving)
    # ------------------------------------------------------------------
    elif target == 4:
        # Radius for guided filter
        r = block_sz // 2 if block_sz else 2
        eps = 1e-2
        
        def box_filter_g(x, r):
            k = 2*r + 1
            x_in = x.unsqueeze(0)
            # Use reflection padding
            x_pad = F.pad(x_in, (r, r, r, r), mode='reflect')
            x_out = F.avg_pool2d(x_pad, kernel_size=k, stride=1, padding=0)
            return x_out.squeeze(0)
            
        I = original
        p = original
        
        mean_I = box_filter_g(I, r)
        mean_p = box_filter_g(p, r)
        mean_Ip = box_filter_g(I*p, r)
        mean_II = box_filter_g(I*I, r)
        
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = box_filter_g(a, r)
        mean_b = box_filter_g(b, r)
        
        q = mean_a * I + mean_b
        
        # JND Bounding
        diff = q - original
        diff = diff.clamp(-jnd_map, jnd_map)
        
        result = original + diff * mask

    else:
        raise ValueError("target must be 0, 1, 2, 3 or 4")

    # 4) optional clamp
    if max_val is not None:
        result = result.clamp(0.0, max_val)

    return result

def block_transform(
    x: torch.Tensor,
    fn: callable,
    block_size: int
) -> torch.Tensor:
    """
    Generic block processing: for (C,H,W) input, cut it into non-overlapping (BH,BW) block_sizexblock_size,
    Call fn(blocks) for transformation (fn input/output are both (num_blocks, block_size, block_size)),
    then reassemble back to (C,H,W).
    """
    C, H, W = x.shape
    assert H % block_size == 0 and W % block_size == 0, \
        f"H={H},W={W} must be divisible by block_size={block_size}"

    BH, BW = H // block_size, W // block_size

    # 1) Reshape (C,H,W) to (C, BH, block_size, BW, block_size)
    x5 = x.view(C, BH, block_size, BW, block_size)
    # 2) Permute to (C, BH, BW, block_size, block_size)
    x5 = x5.permute(0, 1, 3, 2, 4)
    # 3) Merge first 3 dimensions: get (C*BH*BW, block_size, block_size)
    blocks = x5.reshape(-1, block_size, block_size)
    # 4) Call fn block by block
    out_blocks = fn(blocks)
    # 5) Restore back to (C, BH, BW, block_size, block_size)
    out5 = out_blocks.reshape(C, BH, BW, block_size, block_size)
    # 6) invert permute -> (C, BH, block_size, BW, block_size)
    out5 = out5.permute(0, 1, 3, 2, 4)
    # 7) reshape back to (C,H,W)
    return out5.reshape(C, H, W)

def block_idtransform(
    coeff: torch.Tensor,
    fn_inv: callable,
    block_size: int
) -> torch.Tensor:
    # IDCT / KLT inverse transform corresponds to block_transform
    return block_transform(coeff, fn_inv, block_size)

def get_block_saliency_mask(
    Y: torch.Tensor,
    block_size: int = 8,
    saliency_model: str = "spectral",
) -> np.ndarray:
    """
    Calculate block-level saliency and return pixel-level binary mask.

    Args:
      Y              : Grayscale image, shape = (H, W), dtype uint8 or convertible to uint8
      block_size     : Block size (pixels)
      saliency_model : 'spectral' or 'fine'

    Returns:
      mask_pixel     : bool array, shape = (H, W), True=pixels in salient blocks
    """
    # —— 1) Prepare uint8 grayscale numpy array —— #
    if hasattr(Y, "cpu"):  # torch.Tensor
        Y_np = Y.detach().cpu().clamp(0, 255).round().to(torch.uint8).numpy()  # type: ignore
    else:
        Y_np = Y.astype(np.uint8)

    H, W = Y_np.shape

    # —— 2) Generate saliency map and binarize —— #
    if saliency_model.lower() == "spectral":
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    elif saliency_model.lower() == "fine":
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
    else:
        raise ValueError(f"Unsupported saliency_model: {saliency_model}")

    # computeSaliency needs BGR
    _, sal_map = sal.computeSaliency(cv2.cvtColor(Y_np, cv2.COLOR_GRAY2BGR))
    # Normalize to [0,255]
    sal_map_u8 = (sal_map * 255).astype(np.uint8)
    # Otsu binarization
    _, sal_bin = cv2.threshold(
        sal_map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    sal_bin = sal_bin.astype(bool)  # True indicates salient pixels

    # —— 3) Count by blocks: generate block-level boolean grid —— #
    H_blk = (H + block_size - 1) // block_size
    W_blk = (W + block_size - 1) // block_size
    block_mask = np.zeros((H_blk, W_blk), dtype=bool)

    for by in range(H_blk):
        for bx in range(W_blk):
            y0, y1 = by * block_size, min((by + 1) * block_size, H)
            x0, x1 = bx * block_size, min((bx + 1) * block_size, W)
            if sal_bin[y0:y1, x0:x1].any():
                block_mask[by, bx] = True

    # —— 4) Tile block-level mask back to pixel-level —— #
    mask_pixel = np.repeat(
        np.repeat(block_mask, block_size, axis=0),
        block_size, axis=1
    )
    # Crop to original size
    mask_pixel = mask_pixel[:H, :W]

    return mask_pixel