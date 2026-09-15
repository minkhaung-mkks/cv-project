#!/usr/bin/env python3
"""
3D Gaussian Splatting, from scratch, in pure numpy.

Every Gaussian is a blob in space with a position, an anisotropic covariance
(scale + rotation), an opacity and a color. Rendering = project each Gaussian to
a 2D Gaussian in the image (EWA splatting), sort front to back, alpha-blend.
Because that whole chain is differentiable, the blobs can be fit to photos with
gradient descent - this file implements the forward pass AND the analytic
backward pass (no autograd library available, so the derivatives are hand
derived; test_gradients.py checks them against finite differences).

Parameter storage (all raw, activations applied inside):
    mu    (N,3)  position, world space
    logs  (N,3)  log of the 3 axis scales
    quat  (N,4)  rotation, (w,x,y,z), normalized on use
    opa   (N,)   opacity logit      -> sigmoid
    rgb   (N,3)  color logits (BGR) -> sigmoid
"""

import numpy as np

BLUR = 0.3        # low-pass added to the 2D covariance, keeps splats >= 1 pixel
MAX_ALPHA = 0.99
MIN_ALPHA = 1.0 / 255.0
T_STOP = 1e-4     # a tile is done once this little light gets through
CHUNK = 96        # gaussians blended per chunk, so saturated tiles can bail out
DTYPE = np.float32  # rasterizer working precision (tests raise it to float64)


# ---------------------------------------------------------------- activations
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def quat_to_rot(q):
    """(N,4) -> (N,3,3), normalizing q first."""
    n = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = n[:, 0], n[:, 1], n[:, 2], n[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_rot_backward(q, gR):
    """Push (N,3,3) gradient on R back to the raw (unnormalized) quaternion."""
    nrm = np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    n = q / nrm
    w, x, y, z = n[:, 0], n[:, 1], n[:, 2], n[:, 3]
    O = np.zeros_like(w)

    def mat(rows):
        return 2 * np.stack([np.stack(r, -1) for r in rows], -2)

    dw = mat([[O, -z, y], [z, O, -x], [-y, x, O]])
    dx = mat([[O, y, z], [y, -2 * x, -w], [z, w, -2 * x]])
    dy = mat([[-2 * y, x, w], [x, O, z], [-w, z, -2 * y]])
    dz = mat([[-2 * z, -w, x], [w, -2 * z, y], [x, y, O]])
    gn = np.stack([(gR * d).sum((1, 2)) for d in (dw, dx, dy, dz)], -1)
    # through the normalization q -> q/|q|
    return (gn - (gn * n).sum(1, keepdims=True) * n) / nrm


# ---------------------------------------------------------------- projection
def project(mu, logs, quat, cam, cache=True):
    """World-space Gaussians -> screen-space 2D Gaussians (EWA splatting).

    cam = (f, cx, cy, R, C) with R the 3x3 world->camera rotation.
    Returns uv (N,2), depth (N,), conic (N,3) as (a,b,c) of the inverse 2D
    covariance, radius (N,), valid (N,) and a cache for the backward pass.
    """
    f, cx, cy, R, C = cam
    s = np.exp(logs)
    Rq = quat_to_rot(quat)

    t = (mu - C) @ R.T                                   # camera space
    z = t[:, 2]
    valid = z > 1e-2
    zs = np.where(valid, z, 1.0)

    uv = np.stack([f * t[:, 0] / zs + cx, f * t[:, 1] / zs + cy], 1)

    M = Rq * s[:, None, :]                               # Rq @ diag(s)
    Sw = M @ np.transpose(M, (0, 2, 1))                  # world covariance

    J = np.zeros((len(mu), 2, 3))
    J[:, 0, 0] = f / zs
    J[:, 1, 1] = f / zs
    J[:, 0, 2] = -f * t[:, 0] / zs ** 2
    J[:, 1, 2] = -f * t[:, 1] / zs ** 2

    Tm = J @ R                                           # 2x3, world -> screen
    S2 = Tm @ Sw @ np.transpose(Tm, (0, 2, 1))
    A = S2[:, 0, 0] + BLUR
    B = S2[:, 0, 1]
    Cc = S2[:, 1, 1] + BLUR

    det = A * Cc - B * B
    valid &= det > 1e-12
    dets = np.where(valid, det, 1.0)
    conic = np.stack([Cc / dets, -B / dets, A / dets], 1)

    # 3 sigma radius from the larger eigenvalue of the 2D covariance
    mid = 0.5 * (A + Cc)
    disc = np.sqrt(np.maximum(mid * mid - dets, 1e-9))
    radius = 3.0 * np.sqrt(np.maximum(mid + disc, 1e-9))

    ctx = None
    if cache:
        ctx = dict(t=t, zs=zs, J=J, Tm=Tm, Sw=Sw, M=M, Rq=Rq, s=s,
                   A=A, B=B, Cc=Cc, det=dets, valid=valid, cam=cam)
    return uv, z, conic, radius, valid, ctx


def project_backward(g_uv, g_conic, ctx):
    """Screen-space gradients -> gradients on (mu, logs, quat)."""
    f, cx, cy, R, C = ctx["cam"]
    t, zs, J, Tm, Sw, M, Rq, s = (ctx[k] for k in ("t", "zs", "J", "Tm", "Sw", "M", "Rq", "s"))
    A, B, Cc, det, valid = (ctx[k] for k in ("A", "B", "Cc", "det", "valid"))
    N = len(t)

    ga, gb, gc = g_conic[:, 0], g_conic[:, 1], g_conic[:, 2]
    d2 = det * det
    # conic = (Cc/det, -B/det, A/det) differentiated wrt the covariance entries
    gA = ga * (-Cc * Cc / d2) + gb * (B * Cc / d2) + gc * (-B * B / d2)
    gB = ga * (2 * B * Cc / d2) + gb * (-1 / det - 2 * B * B / d2) + gc * (2 * A * B / d2)
    gC = ga * (-B * B / d2) + gb * (A * B / d2) + gc * (-A * A / d2)

    G2 = np.zeros((N, 2, 2))
    G2[:, 0, 0] = gA
    G2[:, 0, 1] = G2[:, 1, 0] = 0.5 * gB
    G2[:, 1, 1] = gC

    TmT = np.transpose(Tm, (0, 2, 1))
    g_Sw = TmT @ G2 @ Tm                                 # S2 = Tm Sw Tm^T
    g_Tm = 2 * (G2 @ Tm @ Sw)
    g_J = g_Tm @ R.T                                     # Tm = J R

    # gradient on the camera-space position, from J and from the projection
    gt = np.zeros((N, 3))
    inv_z2 = 1.0 / zs ** 2
    gt[:, 0] += g_J[:, 0, 2] * (-f * inv_z2)
    gt[:, 1] += g_J[:, 1, 2] * (-f * inv_z2)
    gt[:, 2] += (g_J[:, 0, 0] + g_J[:, 1, 1]) * (-f * inv_z2)
    gt[:, 2] += (g_J[:, 0, 2] * t[:, 0] + g_J[:, 1, 2] * t[:, 1]) * (2 * f / zs ** 3)

    gu, gv = g_uv[:, 0], g_uv[:, 1]
    gt[:, 0] += gu * f / zs
    gt[:, 1] += gv * f / zs
    gt[:, 2] += -(gu * t[:, 0] + gv * t[:, 1]) * f * inv_z2

    g_mu = gt @ R                                        # t = R (mu - C)

    g_M = 2 * (0.5 * (g_Sw + np.transpose(g_Sw, (0, 2, 1))) @ M)   # Sw = M M^T
    g_s = (g_M * Rq).sum(1)                              # M = Rq diag(s)
    g_logs = g_s * s
    g_quat = quat_rot_backward(ctx["q_raw"], g_M * s[:, None, :])

    m = valid[:, None]
    return g_mu * m, g_logs * m, g_quat * m


# ---------------------------------------------------------------- rasterizer
def _tile_lists(uv, radius, depth, valid, H, W, tile):
    """Assign every Gaussian to the tiles its 3-sigma box touches,
    then order the (tile, gaussian) pairs by tile and by depth."""
    ntx, nty = (W + tile - 1) // tile, (H + tile - 1) // tile
    r = radius
    tx0 = np.clip(np.floor((uv[:, 0] - r) / tile), 0, ntx - 1).astype(np.int64)
    tx1 = np.clip(np.floor((uv[:, 0] + r) / tile), 0, ntx - 1).astype(np.int64)
    ty0 = np.clip(np.floor((uv[:, 1] - r) / tile), 0, nty - 1).astype(np.int64)
    ty1 = np.clip(np.floor((uv[:, 1] + r) / tile), 0, nty - 1).astype(np.int64)

    on = valid & (uv[:, 0] + r >= 0) & (uv[:, 0] - r < W) & \
        (uv[:, 1] + r >= 0) & (uv[:, 1] - r < H) & (r > 0)
    wid, hei = (tx1 - tx0 + 1), (ty1 - ty0 + 1)
    counts = np.where(on, wid * hei, 0)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(ntx * nty + 1, np.int64), np.zeros(0, np.int64), ntx, nty

    gid = np.repeat(np.arange(len(uv)), counts)
    start = np.cumsum(counts) - counts
    k = np.arange(total) - np.repeat(start, counts)
    wrep = np.repeat(wid, counts)
    tile_id = (np.repeat(ty0, counts) + k // wrep) * ntx + (np.repeat(tx0, counts) + k % wrep)

    order = np.lexsort((depth[gid], tile_id))
    gid, tile_id = gid[order], tile_id[order]
    bounds = np.searchsorted(tile_id, np.arange(ntx * nty + 1))
    return bounds, gid, ntx, nty


def rasterize(uv, conic, radius, depth, opacity, color, H, W,
              tile=16, bg=0.0, grad_out=None, color_only=False):
    """Front-to-back alpha blending of the screen-space Gaussians.

    grad_out=None  -> returns (image, alpha_map)
    grad_out=(H,W,C) array, or a callable (tile_pixels, y0, y1, x0, x1) ->
    (P,C), giving the gradient of the loss wrt the image. The callable form
    lets training do forward and backward in a single pass, since a per-pixel
    loss only needs the pixels of the tile currently being blended.
    Returns (image, alpha_map, grads) with grads a dict of screen-space
    gradients (uv, conic, opacity, color).
    """
    # Tiles with no projected Gaussians are skipped below, but must still show
    # the chosen background rather than black rectangles.
    img = np.full((H, W, color.shape[1]), bg, DTYPE)
    acc_a = np.zeros((H, W), DTYPE)
    bounds, gid, ntx, nty = _tile_lists(uv, radius, depth, np.ones(len(uv), bool), H, W, tile)

    if grad_out is not None:
        g_uv = np.zeros_like(uv)
        g_conic = np.zeros_like(conic)
        g_opa = np.zeros(len(uv))
        g_col = np.zeros_like(color)

    for ty in range(nty):
        for tx in range(ntx):
            ti = ty * ntx + tx
            lo, hi = bounds[ti], bounds[ti + 1]
            if hi <= lo:
                continue
            tidx = gid[lo:hi]

            y0, y1 = ty * tile, min((ty + 1) * tile, H)
            x0, x1 = tx * tile, min((tx + 1) * tile, W)
            yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
            px = xx.ravel().astype(DTYPE)
            py = yy.ravel().astype(DTYPE)
            P, ch = len(px), color.shape[1]

            tile_img = np.zeros((P, ch), DTYPE)
            T_run = np.ones(P, DTYPE)
            keep = []            # chunks that actually contributed, for the backward pass

            # blend front to back in chunks; stop as soon as the tile is opaque
            for cs in range(0, len(tidx), CHUNK):
                idx = tidx[cs:cs + CHUNK]
                dx = px[None, :] - uv[idx, 0][:, None].astype(DTYPE)
                dy = py[None, :] - uv[idx, 1][:, None].astype(DTYPE)
                a = conic[idx, 0][:, None].astype(DTYPE)
                b = conic[idx, 1][:, None].astype(DTYPE)
                c = conic[idx, 2][:, None].astype(DTYPE)
                q = a * dx * dx + 2 * b * dx * dy + c * dy * dy
                G = np.exp(-0.5 * np.clip(q, 0, 60))

                op = opacity[idx][:, None].astype(DTYPE)
                raw = op * G
                alpha = np.minimum(MAX_ALPHA, raw)
                live = alpha > MIN_ALPHA
                alpha = np.where(live, alpha, DTYPE(0.0))

                Tcum = np.cumprod(1.0 - alpha, axis=0)
                Tex = np.empty_like(Tcum)
                Tex[0] = T_run
                Tex[1:] = T_run * Tcum[:-1]
                w = alpha * Tex

                cols = color[idx].astype(DTYPE)
                tile_img += w.T @ cols                  # (P,ch), BLAS
                T_run = T_run * Tcum[-1]

                if grad_out is not None:
                    keep.append((idx, w) if color_only else (idx, dx, dy, a, b, c, G, alpha, Tex, live, raw, cols, w))
                if T_run.max() < T_STOP:
                    break

            tile_img += bg * T_run[:, None]          # background shows through
            img[y0:y1, x0:x1] = tile_img.reshape(y1 - y0, x1 - x0, ch)
            acc_a[y0:y1, x0:x1] = (1.0 - T_run).reshape(y1 - y0, x1 - x0)

            if grad_out is None:
                continue

            if callable(grad_out):
                gout = np.asarray(grad_out(tile_img, y0, y1, x0, x1), DTYPE)
            else:
                gout = grad_out[y0:y1, x0:x1].reshape(-1, ch).astype(DTYPE)

            if color_only:
                for idx, w in keep:
                    np.add.at(g_col, idx, w @ gout)
                continue

            # Walk the chunks back to front. Only the dot product of the
            # blended-behind color with the pixel gradient is ever needed, so
            # the suffix sums stay 2D (K,P) instead of (K,P,channels).
            gsum = gout.sum(1)
            tail = bg * gsum * T_run                     # (P,) light from the background
            for idx, dx, dy, a, b, c, G, alpha, Tex, live, raw, cols, w in reversed(keep):
                cg = cols @ gout.T                       # (K,P), BLAS
                np.add.at(g_col, idx, w @ gout)

                e = w * cg                               # what this gaussian contributes
                rev = np.cumsum(e[::-1], axis=0)[::-1]
                after = np.empty_like(e)
                after[:-1] = rev[1:]
                after[-1] = 0.0
                after += tail
                tail = tail + rev[0]

                denom = np.maximum(1.0 - alpha, DTYPE(1e-6))
                g_alpha = Tex * cg - after / denom
                g_alpha = np.where(live & (raw < MAX_ALPHA), g_alpha, DTYPE(0.0))

                np.add.at(g_opa, idx, (g_alpha * G).sum(1))
                g_q = -0.5 * G * g_alpha * opacity[idx][:, None].astype(DTYPE)

                np.add.at(g_conic, idx, np.stack([(g_q * dx * dx).sum(1),
                                                  (g_q * 2 * dx * dy).sum(1),
                                                  (g_q * dy * dy).sum(1)], 1))
                g_dx = g_q * 2 * (a * dx + b * dy)
                g_dy = g_q * 2 * (b * dx + c * dy)
                np.add.at(g_uv, idx, np.stack([-g_dx.sum(1), -g_dy.sum(1)], 1))

    if grad_out is None:
        return img, acc_a
    return img, acc_a, dict(uv=g_uv, conic=g_conic, opacity=g_opa, color=g_col)


# ---------------------------------------------------------------- full model
def render(params, cam, H, W, tile=16, bg=0.0, grad_out=None, alpha_channel=False, appearance_only=False):
    """Render the whole Gaussian set through one camera.

    alpha_channel adds a 4th output channel that is the accumulated coverage,
    so silhouette masks can supervise the geometry directly.
    With grad_out, also returns gradients wrt the raw parameters."""
    mu, logs, quat = params["mu"], params["logs"], params["quat"]
    opa_raw, rgb_raw = params["opa"], params["rgb"]
    opacity = sigmoid(opa_raw)
    base_color = sigmoid(rgb_raw)
    color = base_color
    direction = basis = distance = color_live = None
    if "sh" in params:
        # Real degree-one SH, in the conventional 3DGS coefficient order.
        # Directions point from camera to Gaussian; coefficients store BGR.
        delta = mu - cam[4]
        distance = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
        direction = delta / distance
        k = 0.4886025119029199
        basis = k * np.stack([-direction[:, 1], direction[:, 2], -direction[:, 0]], 1)
        raw_color = base_color + np.einsum("nk,nkc->nc", basis, params["sh"])
        color_live = (raw_color > 0) & (raw_color < 1)
        color = np.clip(raw_color, 0, 1)
    if alpha_channel:
        color = np.concatenate([color, np.ones((len(color), 1))], 1)

    uv, depth, conic, radius, valid, ctx = project(mu, logs, quat, cam)
    dep = np.where(valid, depth, 1e9)
    rad = np.where(valid, radius, 0.0)

    if grad_out is None:
        img, alpha = rasterize(uv, conic, rad, dep, opacity, color, H, W, tile, bg)
        return img, alpha

    img, alpha, g = rasterize(uv, conic, rad, dep, opacity, color, H, W, tile, bg, grad_out, color_only=appearance_only)
    ctx["q_raw"] = quat
    if appearance_only:
        g_mu, g_logs, g_quat = np.zeros_like(mu), np.zeros_like(logs), np.zeros_like(quat)
    else:
        g_mu, g_logs, g_quat = project_backward(g["uv"], g["conic"], ctx)
    g_color = g["color"][:, :3]
    if color_live is not None:
        g_color = g_color * color_live
    grads = dict(mu=g_mu, logs=g_logs, quat=g_quat,
                 opa=g["opacity"] * opacity * (1 - opacity),
                 rgb=g_color * base_color * (1 - base_color))
    if basis is not None:
        grads["sh"] = basis[:, :, None] * g_color[:, None, :]
        g_basis = np.einsum("nc,nkc->nk", g_color, params["sh"])
        g_direction = k * np.stack([-g_basis[:, 2], -g_basis[:, 0], g_basis[:, 1]], 1)
        if not appearance_only:
            grads["mu"] += (g_direction - direction * (g_direction * direction).sum(1, keepdims=True)) / distance
    return img, alpha, grads, g["uv"]


# ---------------------------------------------------------------- optimizer
class Adam:
    """Adam with per-parameter learning rates, able to grow/shrink with the
    Gaussian set (densification changes how many parameters exist)."""

    def __init__(self, params, lrs, b1=0.9, b2=0.999, eps=1e-8):
        self.lrs = dict(lrs)
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr_scale=1.0):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for k, p in params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            step = (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)
            p -= self.lrs[k] * lr_scale * step

    def reset(self, key):
        """Drop the momentum for one parameter (used after an opacity reset)."""
        self.m[key][:] = 0.0
        self.v[key][:] = 0.0

    def keep(self, mask):
        for k in self.m:
            self.m[k] = self.m[k][mask]
            self.v[k] = self.v[k][mask]

    def grow(self, n_new):
        for k in self.m:
            z = np.zeros((n_new,) + self.m[k].shape[1:])
            self.m[k] = np.concatenate([self.m[k], z], 0)
            self.v[k] = np.concatenate([self.v[k], z], 0)
