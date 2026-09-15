#!/usr/bin/env python3
"""
Fit 3D Gaussians to a turntable photo set (3D Gaussian Splatting, numpy only).

    python train_splat.py demo_images --init model.npz --iters 1500
    python splat_viewer.py splats.npz

Camera poses are not estimated - like carve.py, the photos are assumed to be
evenly spaced around the object, so the poses come for free. Photo segmentation
and framing normalization are reused from carve.py, and the carved point cloud
makes an excellent initialization for the Gaussians.

Losses: L1 on color + L1 on the rendered coverage vs the silhouette mask.
Adaptive density control: clone / split Gaussians with a large screen-space
gradient, prune transparent or oversized ones.
"""

import argparse
import time

import cv2
import numpy as np

import carve
import poses as poses_mod
import splat_core as sc


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- data
def prepare(args):
    if args.poses:
        imgs, msks, raw_cams = poses_mod.load(args.poses, args.images, args.res, args.radius)
        if all((m > 250).all() for m in msks):
            print("dataset has no alpha channel, segmenting instead...")
            msks = carve.segment(imgs, [f"v{i}" for i in range(len(imgs))],
                                 args.mask, args.max_hole)
    else:
        images, paths = carve.load_images(args.images, args.max_size)
        print("segmenting...")
        masks = carve.segment(images, paths, args.mask, args.max_hole)
        imgs, msks = carve.normalize(images, masks, args.alpha, args.res, False)
        raw_cams = carve.make_cameras(len(imgs), args.res, args.alpha, args.elev,
                                      args.radius, args.clockwise)

    targets, alphas = [], []
    for img, m in zip(imgs, msks):
        a = (m > 0).astype(np.float64)
        targets.append(img.astype(np.float64) / 255.0 * a[..., None])   # object over black
        alphas.append(a)
    # cameras arrive as (K, R, C); the splat renderer wants (f, cx, cy, R, C)
    cams = [(K[0, 0], K[0, 2], K[1, 2], R, C) for K, R, C in raw_cams]
    return targets, alphas, cams, imgs, msks, raw_cams


def envelope_cloud(imgs, msks, raw_cams, keep=0.9, grid=96, bound=1.0):
    """Carve a filled-in envelope of the object to seed the Gaussians.

    The holes are deliberately closed here: on a see-through object the honest
    visual hull collapses to nothing, but its envelope is still a fine place to
    put the initial blobs. The real holes come back through the silhouette loss
    during training.
    """
    env = [carve.clean_mask(m.copy(), max_hole=1.0) for m in msks]
    lo, hi = carve.fit_bounds(env, raw_cams, keep, bound)
    occ, lins = carve.carve(env, raw_cams, grid, keep, lo, hi, quiet=True)
    pts, nrm = carve.surface_points(occ, lins)
    col, hit = carve.colorize(pts, nrm, imgs, raw_cams, occ, lins, False)
    voxel = float(np.mean((hi - lo) / grid))
    print(f"  envelope: {int(hit.sum())} seed points, voxel {voxel:.4f}")
    return pts[hit], col[hit].astype(np.float64) / 255.0, voxel


# ---------------------------------------------------------------- init
def init_params(args, seed_cloud=None):
    rng = np.random.default_rng(0)
    if seed_cloud is not None or args.init:
        if seed_cloud is not None:
            pts, cols, voxel = seed_cloud
            src = "carved envelope"
        else:
            d = np.load(args.init)
            pts = d["points"].astype(np.float64)
            cols = d["colors"].astype(np.float64) / 255.0
            voxel = float(d["voxel"]) if "voxel" in d else 1.0 / 128
            src = args.init
        n_all = len(pts)
        if n_all > args.num:
            sel = rng.choice(n_all, args.num, replace=False)
            pts, cols = pts[sel], cols[sel]
        # thinning the cloud spreads the points out, so start the blobs bigger
        spread = np.sqrt(max(1.0, n_all / max(len(pts), 1)))
        scale0 = voxel * spread * 0.8
        print(f"init from {src}: {len(pts)} gaussians, scale {scale0:.4f}")
    else:
        k = args.num
        v = rng.normal(0, 1, (k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        pts = v * rng.uniform(0, 0.35, (k, 1)) ** (1 / 3)
        cols = np.full((k, 3), 0.5)
        scale0 = 0.04
        print(f"random init: {k} gaussians")

    n = len(pts)
    return dict(
        mu=pts.copy(),
        logs=np.full((n, 3), np.log(scale0)),
        quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        opa=np.full(n, logit(0.5)),
        rgb=logit(cols),
    )


def eval_psnr(params, cams, targets, H, W, tile):
    """Mean PSNR over every training view."""
    mse = 0.0
    for cam, tgt in zip(cams, targets):
        img, _ = sc.render(params, cam, H, W, tile=tile, bg=0.0)
        mse += ((img[..., :3] - tgt) ** 2).mean()
    mse /= len(cams)
    return 10 * np.log10(1.0 / max(mse, 1e-12))


# ---------------------------------------------------------------- densify
def densify(params, opt, stats, args, it):
    grad = stats["grad"] / np.maximum(stats["count"], 1)
    scale = np.exp(params["logs"]).max(1)
    opacity = sc.sigmoid(params["opa"])

    # 1. prune: invisible or bloated blobs
    keep = (opacity > args.prune_opacity) & (scale < args.max_scale)
    if keep.sum() < 16:
        keep[:] = True
    if (~keep).any():
        for k in params:
            params[k] = params[k][keep]
        opt.keep(keep)
        grad, scale = grad[keep], scale[keep]

    # 2. grow: blobs the loss keeps pushing around are under-describing the object
    room = args.max_num - len(params["mu"])
    n_new = 0
    if room > 0 and len(grad) > 8:
        thr = np.quantile(grad, 1.0 - args.grow_frac)
        pick = np.nonzero((grad >= thr) & (grad > 0))[0][:room]
        if len(pick):
            rng = np.random.default_rng(it)
            small = scale[pick] < args.split_scale
            new = {k: params[k][pick].copy() for k in params}
            # small ones are cloned and nudged, big ones are split in two
            jitter = rng.normal(0, 1, (len(pick), 3)) * np.exp(new["logs"])
            new["mu"] += np.where(small[:, None], jitter * 0.3, jitter)
            new["logs"][~small] -= np.log(1.6)
            params["logs"][pick[~small]] -= np.log(1.6)
            for k in params:
                params[k] = np.concatenate([params[k], new[k]], 0)
            n_new = len(pick)
            opt.grow(n_new)

    stats["grad"] = np.zeros(len(params["mu"]))
    stats["count"] = np.zeros(len(params["mu"]))
    return int(keep.size - keep.sum()), n_new


# ---------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser(description="3D Gaussian Splatting on a turntable photo set")
    ap.add_argument("images")
    ap.add_argument("--init", default="auto",
                    help="'auto' carves a filled envelope internally and seeds the "
                         "gaussians with it (default); or a model.npz from carve.py; "
                         "or 'none' for a random cloud")
    ap.add_argument("--init-grid", type=int, default=96, help="envelope carve resolution")
    ap.add_argument("--init-keep", type=float, default=0.9,
                    help="view agreement for the envelope carve")
    ap.add_argument("--out", default="splats.npz")
    ap.add_argument("--resume", help="warm-start Gaussian parameters from an existing NPZ; resets Adam")
    ap.add_argument("--view-dependent", action="store_true", help="fit degree-one spherical-harmonic color")
    ap.add_argument("--lr-sh", type=float, default=0.02)
    ap.add_argument("--sh-reg", type=float, default=0.001, help="L2 regularization of directional color")
    ap.add_argument("--appearance-only", action="store_true", help="freeze geometry and opacity; fit colors only")
    ap.add_argument("--surface-prior", help="optional NPZ point cloud limiting drift away from measured geometry")
    ap.add_argument("--surface-radius", type=float, default=0.03,
                    help="maximum distance from the measured surface when --surface-prior is used")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--res", type=int, default=128, help="training resolution")
    ap.add_argument("--num", type=int, default=4000, help="initial gaussian count")
    ap.add_argument("--max-num", type=int, default=30000)
    ap.add_argument("--tile", type=int, default=16)
    # scene / camera (must match how the photos were taken, same as carve.py)
    ap.add_argument("--elev", type=float, default=15.0)
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--clockwise", action="store_true")
    ap.add_argument("--mask", default="auto", choices=["auto", "grabcut", "bgcolor", "otsu", "alpha"])
    ap.add_argument("--max-size", type=int, default=800)
    ap.add_argument("--max-hole", type=float, default=0.002,
                    help="fill silhouette holes smaller than this fraction of the object "
                         "area; the default only swallows segmentation speckle")
    ap.add_argument("--poses", metavar="TRANSFORMS.JSON",
                    help="real camera poses (NeRF/Blender transforms.json)")
    # optimization
    ap.add_argument("--lr-mu", type=float, default=8e-3)
    ap.add_argument("--lr-logs", type=float, default=4e-2)
    ap.add_argument("--lr-quat", type=float, default=1.5e-2)
    ap.add_argument("--lr-opa", type=float, default=3e-1)
    ap.add_argument("--lr-rgb", type=float, default=1.5e-1)
    ap.add_argument("--w-alpha", type=float, default=0.5, help="weight of the silhouette loss")
    ap.add_argument("--l2-mix", type=float, default=0.5,
                    help="blend of L2 into the L1 loss (0 = pure L1)")
    # density control
    ap.add_argument("--densify-every", type=int, default=150)
    ap.add_argument("--densify-until", type=float, default=0.7, help="fraction of training")
    ap.add_argument("--densify-from", type=int, default=200)
    ap.add_argument("--grow-frac", type=float, default=0.06)
    ap.add_argument("--split-scale", type=float, default=0.02)
    ap.add_argument("--max-scale", type=float, default=0.25)
    ap.add_argument("--prune-opacity", type=float, default=0.01)
    ap.add_argument("--opacity-reset", type=int, default=0, metavar="N",
                    help="every N iterations, push all opacities back down (3DGS trick, "
                         "clears haze; try 500 on see-through objects)")
    ap.add_argument("--reset-opacity", type=float, default=0.08)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--preview", default="preview_splat.png")
    args = ap.parse_args()
    surface_tree = None
    if args.surface_prior:
        from scipy.spatial import cKDTree
        if not np.isfinite(args.surface_radius) or args.surface_radius <= 0:
            ap.error("--surface-radius must be finite and positive")
        with np.load(args.surface_prior) as prior:
            surface_points = prior["points"].astype(np.float64)
        if not len(surface_points) or not np.isfinite(surface_points).all():
            ap.error("surface prior must contain finite, nonempty points")
        surface_tree = cKDTree(surface_points)
    if args.init in ("none", "random", ""):
        args.init = None

    targets, alphas, cams, imgs, msks, raw_cams = prepare(args)
    H = W = args.res
    seed = None
    if args.init == "auto" and not args.resume:
        print("carving an envelope to initialize the gaussians...")
        seed = envelope_cloud(imgs, msks, raw_cams, keep=args.init_keep, grid=args.init_grid)
    if args.resume:
        with np.load(args.resume) as checkpoint:
            params = {k: checkpoint[k].copy() for k in ("mu", "logs", "quat", "opa", "rgb")}
            if "sh" in checkpoint:
                params["sh"] = checkpoint["sh"].copy()
        print(f"warm-start from {args.resume}: {len(params['mu'])} gaussians (fresh Adam)")
    else:
        params = init_params(args, seed)
    if args.view_dependent and "sh" not in params:
        params["sh"] = np.zeros((len(params["mu"]), 3, 3), np.float64)
    opt = sc.Adam(params, dict(mu=args.lr_mu, logs=args.lr_logs, quat=args.lr_quat,
                               opa=args.lr_opa, rgb=args.lr_rgb, sh=args.lr_sh))
    stats = dict(grad=np.zeros(len(params["mu"])), count=np.zeros(len(params["mu"])))

    rng = np.random.default_rng(1)
    npix = H * W
    t0 = time.time()
    print(f"training {args.iters} iters at {W}x{H} on {len(cams)} views")

    for it in range(1, args.iters + 1):
        # position lr decays like in the 3DGS paper: coarse layout first, then
        # detail; everything else follows a gentler cosine decay
        opt.lrs["mu"] = args.lr_mu * (0.01 ** (it / args.iters))
        lr_scale = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / args.iters))

        v = int(rng.integers(len(cams)))
        tgt, amask = targets[v], alphas[v]

        def tile_grad(tile_px, y0, y1, x0, x1, tgt=tgt, amask=amask):
            """L1 on color + L1 on coverage, evaluated on one tile."""
            t3 = tgt[y0:y1, x0:x1].reshape(-1, 3)
            ta = amask[y0:y1, x0:x1].reshape(-1)
            m = args.l2_mix
            e3 = tile_px[:, :3] - t3
            ea = tile_px[:, 3] - ta
            g = np.empty_like(tile_px)
            g[:, :3] = ((1 - m) * np.sign(e3) + m * 2 * e3) / (npix * 3)
            g[:, 3] = args.w_alpha * ((1 - m) * np.sign(ea) + m * 2 * ea) / npix
            return g

        img, _, grads, g_uv = sc.render(params, cams[v], H, W, tile=args.tile, bg=0.0,
                                        grad_out=tile_grad, alpha_channel=True, appearance_only=args.appearance_only)
        if "sh" in params:
            grads["sh"] += args.sh_reg * params["sh"] / len(params["mu"])
        if args.appearance_only:
            for key in ("mu", "logs", "quat", "opa"):
                grads[key].fill(0)
        opt.step(params, grads, lr_scale=lr_scale)
        if "sh" in params:
            params["sh"] = np.clip(params["sh"], -0.5, 0.5)
        if surface_tree is not None and not args.appearance_only:
            # Moving ribbons/specular highlights must not pull geometry into space.
            distances, nearest = surface_tree.query(params["mu"])
            excess = distances > args.surface_radius
            anchors = surface_points[nearest[excess]]
            params["mu"][excess] = anchors + (params["mu"][excess] - anchors) * (
                args.surface_radius / distances[excess, None])
            params["logs"] = np.minimum(params["logs"], np.log(args.max_scale))

        d_rgb = img[..., :3] - tgt
        d_a = img[..., 3] - amask
        l1, la = np.abs(d_rgb).mean(), np.abs(d_a).mean()

        gn = np.linalg.norm(g_uv, axis=1)
        stats["grad"] += gn
        stats["count"] += (gn > 0)

        # 3DGS opacity reset: knock everything back to nearly transparent and
        # let the loss re-earn it. Semi-transparent haze sitting in the holes of
        # a see-through object cannot pay for itself and disappears.
        if not args.appearance_only and args.opacity_reset and it % args.opacity_reset == 0 and \
                it <= args.densify_until * args.iters:
            params["opa"] = np.minimum(params["opa"], logit(args.reset_opacity))
            opt.reset("opa")
            print(f"  [{it:5d}] opacity reset to {args.reset_opacity}")

        if (not args.appearance_only and it >= args.densify_from and it % args.densify_every == 0
                and it <= args.densify_until * args.iters):
            pruned, added = densify(params, opt, stats, args, it)
            print(f"  [{it:5d}] densify: -{pruned} +{added} -> {len(params['mu'])} gaussians")

        if it % 50 == 0:
            print(f"  [{it:5d}/{args.iters}] training L1 {l1:.4f}  elapsed {time.time()-t0:.1f}s", flush=True)

        if it % args.eval_every == 0 or it == 1:
            psnr = eval_psnr(params, cams, targets, H, W, args.tile)
            print(f"  [{it:5d}] L1 {l1:.4f}  mask {la:.4f}  PSNR(all views) {psnr:5.2f} dB  "
                  f"N {len(params['mu']):6d}  {time.time()-t0:5.1f}s")
            np.savez_compressed(args.out + ".checkpoint.npz", **params)
            if args.preview:
                tiles = []
                for j in (0, len(cams) // 3, 2 * len(cams) // 3):
                    preview, _ = sc.render(params, cams[j], H, W, tile=args.tile, bg=0.0)
                    tiles.extend([np.clip(preview * 255, 0, 255).astype(np.uint8), imgs[j]])
                cv2.imwrite(args.preview, np.hstack(tiles))

    # remember how the training cameras were set up, so the viewer frames the
    # model the same way (with --poses the intrinsics come from the dataset,
    # not from the turntable formula)
    centers = np.array([c[4] for c in cams])
    focals = np.array([c[0] for c in cams])
    center = np.median(params["mu"], 0)
    cam_radius = float(np.linalg.norm(centers - center, axis=1).mean())
    # median elevation over all cameras, so one odd top-down view does not
    # decide where the viewer starts
    elev = float(np.clip(np.median(np.degrees(np.arcsin(
        np.clip((centers - center)[:, 1] / max(cam_radius, 1e-9), -1, 1)))), -60, 60))
    np.savez_compressed(args.out, **params,
                        meta=np.array([elev, cam_radius, float(focals.mean()) / args.res,
                                       args.res]),
                        center=center)
    print(f"\nsaved {args.out}  ({len(params['mu'])} gaussians)")

    if args.preview:
        tiles = []
        for i in [0, len(cams) // 4, len(cams) // 2]:
            im, _ = sc.render(params, cams[i], H, W, tile=args.tile, bg=0.0)
            tiles.append(np.clip(im * 255, 0, 255).astype(np.uint8))
            tiles.append(np.clip(targets[i] * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(args.preview, np.hstack(tiles))
        print(f"wrote {args.preview} (render, photo, render, photo, ...)")
    print(f"spin it:  python viewer_splat.py {args.out}")


if __name__ == "__main__":
    main()
