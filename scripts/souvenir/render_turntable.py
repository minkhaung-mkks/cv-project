#!/usr/bin/env python3
"""
Presentation renderer: a splats.npz -> a turntable video worth projecting.

viewer_splat.py --spin already renders a 360, but it renders it the way the
model was trained: small in frame, on near-black, at the elevation the
cameras happened to sit at. This wraps the same rasterizer with the knobs a
demo needs - zoom, exposure, background, a slow elevation drift so the shape
reads as solid, and an optional caption.

Nothing here changes the model. Every pixel still comes out of
splat_core.render; --gain and --gamma are display tone mapping applied after
rendering, exactly like turning up a projector.

    python render_turntable.py real_v3_splats.npz demo.mp4
    python render_turntable.py real_v3_splats.npz demo.mp4 --gain 2.2 --zoom 1.5
    python render_turntable.py real_v3_splats.npz stills/ --stills 8
"""

import argparse
import os

import cv2
import numpy as np

import splat_core as sc
from viewer_splat import load, look_at


def render_one(params, meta, center, yaw, pitch, res, zoom, bg, gain, gamma):
    _, radius0, f_norm, _ = meta
    R, C = look_at(yaw, pitch, radius0, center)
    cam = (f_norm * res * zoom, res / 2, res / 2, R, C)
    img, _ = sc.render(params, cam, res, res, tile=16, bg=bg)
    img = np.clip(img * gain, 0, 1) ** (1.0 / max(gamma, 1e-6))
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def caption(img, text, sub=""):
    if not text:
        return img
    h = img.shape[0]
    cv2.putText(img, text, (18, h - (44 if sub else 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (18, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="splats.npz from train_splat.py")
    ap.add_argument("out", help="output .mp4, or a directory if --stills is given")
    ap.add_argument("--size", type=int, default=800, help="output frame size")
    ap.add_argument("--res", type=int, default=400,
                    help="render resolution (upsampled to --size)")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--zoom", type=float, default=1.35,
                    help="fill more of the frame than the training cameras did")
    ap.add_argument("--bg", type=float, default=0.10)
    ap.add_argument("--gain", type=float, default=1.9,
                    help="display exposure - the ornament trains dark")
    ap.add_argument("--gamma", type=float, default=1.25, help="display gamma")
    ap.add_argument("--elev", type=float, default=None,
                    help="fixed elevation in degrees (default: the training one)")
    ap.add_argument("--wobble", type=float, default=8.0,
                    help="degrees of slow elevation drift over the loop, 0 for none")
    ap.add_argument("--label", default="", help="caption burned into the corner")
    ap.add_argument("--sublabel", default="")
    ap.add_argument("--yaw-start", type=float, default=0.0,
                    help="azimuth the orbit starts at, degrees")
    ap.add_argument("--yaw-sweep", type=float, default=360.0,
                    help="how far the orbit travels. Use less than 360 when "
                         "the views only cover an arc - a model fitted to a "
                         "90 degree arc has nothing behind it, and spinning "
                         "it all the way round shows you the nothing. Below "
                         "360 the orbit rocks out and back so the clip loops.")
    ap.add_argument("--stills", type=int, default=0,
                    help="write this many evenly spaced PNGs instead of a video")
    a = ap.parse_args()

    params, meta, center = load(a.model)
    if center is None:
        center = np.median(params["mu"], 0)
    base = np.radians(a.elev if a.elev is not None else meta[0])
    print(f"{len(params['mu'])} gaussians  elev {np.degrees(base):.0f} deg  zoom {a.zoom}")

    def frame_at(i, n):
        if a.yaw_sweep >= 360.0:
            yaw = np.radians(a.yaw_start) + 2 * np.pi * i / n
        else:
            # out and back, so a partial orbit still loops seamlessly
            tri = 1 - abs(2.0 * i / n - 1.0)
            yaw = np.radians(a.yaw_start + a.yaw_sweep * tri)
        pitch = base + np.radians(a.wobble) * np.sin(2 * np.pi * i / n)
        img = render_one(params, meta, center, yaw, pitch,
                         a.res, a.zoom, a.bg, a.gain, a.gamma)
        img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_CUBIC)
        return caption(img, a.label, a.sublabel)

    if a.stills:
        os.makedirs(a.out, exist_ok=True)
        for k in range(a.stills):
            i = int(k * a.frames / a.stills)
            cv2.imwrite(os.path.join(a.out, f"turn_{k:02d}.png"), frame_at(i, a.frames))
            print(f"\r  still {k+1}/{a.stills}", end="", flush=True)
        print(f"\nwrote {a.stills} stills to {a.out}/")
        return

    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if a.out.endswith(".mp4") else "MJPG"))
    vw = cv2.VideoWriter(a.out, fourcc, a.fps, (a.size, a.size))
    for i in range(a.frames):
        vw.write(frame_at(i, a.frames))
        print(f"\r  frame {i+1}/{a.frames}", end="", flush=True)
    vw.release()
    print(f"\nwrote {a.out}  ({a.frames} frames, {a.frames/a.fps:.1f}s)")


if __name__ == "__main__":
    main()
