"""Lyric PV renderer: 'Now there's only ONE thing I can do / Fight until the end like I PROMISED to'
Pure numpy / OpenCV / PIL frame renderer, piped to ffmpeg."""
import numpy as np, cv2, math, os, sys, subprocess
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)
W, H, FPS = 1920, 1080, 60
DUR = 11.85
NF = int(round(DUR * FPS))
HERE = os.path.dirname(os.path.abspath(__file__))
FN = HERE + '/fonts/IM_Fell_English_SC.ttf'
FE = HERE + '/fonts/Abril_Fatface.ttf'
AUDIO = HERE + '/work/audio.wav'
VLEAD = 0.010          # visuals lead the audio by a hair so hits feel tight

BONE = np.array([0.91, 0.86, 0.77], np.float32)
DRED = np.array([0.42, 0.02, 0.03], np.float32)

KICKS = [1.654, 2.281, 2.914, 3.547, 4.180, 4.801, 5.439, 6.072, 6.699, 7.332, 7.964, 8.603, 9.236, 9.863, 10.495]
OFFS = [1.974, 2.601, 3.239, 3.866, 4.499, 5.143, 5.759, 6.391, 7.024, 7.663, 8.290, 8.922, 9.561, 10.188, 10.815]
MUSIC_END = 11.05


# ---------------------------------------------------------------- helpers
def cl(x, a=0., b=1.): return min(max(x, a), b)
def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
def e_out3(u): u = cl(u); return 1 - (1 - u) ** 3
def e_out5(u): u = cl(u); return 1 - (1 - u) ** 5
def e_outexpo(u): u = cl(u); return 1. if u >= 1 else 1 - 2 ** (-10 * u)
def e_inout(u): u = cl(u); return 4 * u ** 3 if u < .5 else 1 - (-2 * u + 2) ** 3 / 2
def e_in2(u): u = cl(u); return u * u
def lin(u): return cl(u)
def dec(t, t0, tau): return 0. if t < t0 else math.exp(-(t - t0) / tau)

def T(x, y): return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)
def Sc(s, s2=None): return np.array([[s, 0, 0], [0, s if s2 is None else s2, 0], [0, 0, 1]], np.float64)
def Rot(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


class Noise1:
    def __init__(s, seed, n=5):
        r = np.random.default_rng(seed)
        s.f = r.uniform(0.7, 1.4, n) * np.array([1, 1.9, 3.7, 6.3, 11.1])[:n]
        s.p = r.uniform(0, 2 * np.pi, n); s.a = 1 / np.arange(1, n + 1) ** 0.8
    def __call__(s, t, freq=1.):
        return float(np.sum(s.a * np.sin(2 * np.pi * s.f * freq * t + s.p)) / np.sum(s.a) * 1.5)


def fbm(h, w, cell, octaves, rng):
    out = np.zeros((h, w), np.float32); amp = 1.; tot = 0.
    for _ in range(octaves):
        gh, gw = int(h / cell) + 3, int(w / cell) + 3
        g = rng.random((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)[:h, :w]
        out += amp * up; tot += amp; amp *= 0.55; cell = max(cell / 2, 1.)
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


_fc = {}
def font(path, px):
    k = (path, int(px))
    if k not in _fc: _fc[k] = ImageFont.truetype(path, int(px))
    return _fc[k]


def measure(text, fpath, size, tracking=0.):
    S = 4.; f = font(fpath, round(size * S))
    asc, _ = f.getmetrics(); x = 0.; ink = [1e9, 1e9, -1e9, -1e9]
    for ch in text:
        bb = f.getbbox(ch)
        if bb[2] > bb[0]:
            ink = [min(ink[0], x + bb[0]), min(ink[1], bb[1] - asc), max(ink[2], x + bb[2]), max(ink[3], bb[3] - asc)]
        x += f.getlength(ch) + tracking * size * S
    adv = x - tracking * size * S
    return adv / S, [v / S for v in ink]


# ---------------------------------------------------------------- blood shading
LIGHT = np.array([-0.45, -0.65, 0.62]); LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0, 0, 1.]); HALF /= np.linalg.norm(HALF)

def blood_shade(A, px_scale, wet=1.0, dark=1.0):
    """A: alpha (float32 HxW). px_scale ~ pixels per 'unit' (controls bevel width). returns premult rgba"""
    sig = max(1.0, 2.2 * px_scale)
    h = cv2.GaussianBlur(A, (0, 0), sig)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    k = 3.2 * sig
    nx, ny = -gx * k, -gy * k
    inv = 1. / np.sqrt(nx * nx + ny * ny + 1.)
    nx *= inv; ny *= inv; nz = inv
    diff = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    sp = np.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0, 1)
    spec = sp ** 60 * 1.1 * wet + sp ** 12 * 0.10 * wet
    th = np.clip(h * 1.15, 0, 1) ** 0.6
    out = np.empty(A.shape + (4,), np.float32)
    base_r = (0.95 - 0.33 * th) * dark; base_g = (0.040 - 0.025 * th) * dark; base_b = (0.050 - 0.030 * th) * dark
    lit = 0.38 + 0.85 * diff
    out[..., 0] = (base_r * lit + spec) * A
    out[..., 1] = (base_g * lit + spec * 0.80) * A
    out[..., 2] = (base_b * lit + spec * 0.78) * A
    out[..., 3] = A
    return out


# ---------------------------------------------------------------- drips
class Drip:
    pass

def make_drips(A, n, px, rng, t0, spread=0.7, maxlen=1.0):
    """find bottom edges of strokes in alpha A and spawn n drips"""
    M = A > 0.5
    edge = M[:-1, :] & ~M[1:, :]
    ys, xs = np.nonzero(edge)
    if len(ys) == 0: return []
    ybot = ys.max()
    # need some stroke thickness above the edge
    th = max(2, int(px * 0.035))
    ok = np.array([M[max(0, y - th):y + 1, x].all() for y, x in zip(ys, xs)])
    ys, xs = ys[ok], xs[ok]
    w = np.exp(-(ybot - ys) / (0.10 * px)) + 0.04
    w /= w.sum()
    drips = []; used = []
    tries = 0
    while len(drips) < n and tries < 4000:
        tries += 1
        i = rng.choice(len(ys), p=w)
        x, y = int(xs[i]), int(ys[i])
        if any(abs(x - u) < px * 0.07 for u in used): continue
        used.append(x)
        d = Drip()
        d.x, d.y = x, y
        d.w = px * rng.uniform(0.016, 0.040) * (1.25 if rng.random() < 0.25 else 1.0)
        d.L = px * (0.10 + maxlen * 1.25 * rng.beta(1.1, 2.6))
        d.ts = t0 + rng.uniform(0, spread)
        d.dur = 0.5 + d.L / px * rng.uniform(1.0, 2.2)
        d.ph = rng.uniform(0, 1)
        d.fall = rng.random() < 0.45
        drips.append(d)
    return drips


def draw_drips(shape, drips, t, ss=1):
    """returns alpha (float32) of drips at time t, plus stain points"""
    Hh, Ww = shape
    img = np.zeros((Hh, Ww), np.uint8)
    stains = []
    sh = 3; F = 1 << sh
    for d in drips:
        if t <= d.ts: continue
        u = (t - d.ts) / d.dur
        prog = e_out3(min(u, 1.)) ** 0.9
        # stutter: blood creeps, stalls, creeps
        prog = max(0., prog - 0.035 * max(0., math.sin(2 * math.pi * (u * 3.0 + d.ph))) * (1 - min(u, 1)))
        L = d.L * prog; w = d.w
        stains.append((d.x, d.y, w, min(1., (t - d.ts) / 0.35)))
        x0, y0 = d.x, d.y
        if L < w * 0.9:
            ax = max(1, int(w * 0.95 * F)); ay = max(1, int(max(L, 1.) * F))
            cv2.ellipse(img, (int(x0 * F), int(y0 * F)), (ax, ay), 0, 0, 180, 255, -1, cv2.LINE_AA, sh)
            continue
        tip = y0 + L
        bulb = 0.62 * w * (1.0 + 0.25 * min(u, 1))
        taper = 0.42 * w
        pts_l = [(x0 - 1.25 * w, y0 - 0.35 * w), (x0 - 0.85 * w, y0 + 0.10 * w), (x0 - 0.6 * w, y0 + 0.45 * w),
                 (x0 - 0.5 * w, y0 + 0.9 * w), (x0 - taper, tip - bulb * 1.2)]
        pts_r = [(2 * x0 - px_, py_) for px_, py_ in reversed(pts_l)]
        poly = np.array([[int(px_ * F), int(py_ * F)] for px_, py_ in pts_l + pts_r], np.int32)
        cv2.fillPoly(img, [poly], 255, cv2.LINE_AA, sh)
        cv2.ellipse(img, (int(x0 * F), int((tip - bulb * 0.9) * F)), (int(bulb * F), int(bulb * 1.12 * F)), 0, 0, 360, 255, -1, cv2.LINE_AA, sh)
        if d.fall and u > 1.0:
            tf = (t - d.ts - d.dur)
            yy = tip + 0.5 * 2600 * (w / 6) * tf * tf + 30 * tf
            r = bulb * 0.8
            if yy < Hh + r * 3:
                cv2.ellipse(img, (int(x0 * F), int(yy * F)), (int(r * F), int(r * 1.5 * F)), 0, 0, 360, 255, -1, cv2.LINE_AA, sh)
    return img.astype(np.float32) * (1 / 255.), stains


# ---------------------------------------------------------------- text sprite
class Word:
    def __init__(s, text, fpath, size, origin, t_in, t_out=99., ent='slam', ent_dur=None, ex='cut', ex_dur=0.45,
                 style='bone', tracking=0., rs=2.0, drips=0, drip_t=None, drip_spread=0.7, drip_len=1.0, seed=0,
                 z=10, impact=1.0, echo=True, slam_from=2.3, misreg=True, distress=1.0, split_angle=0.):
        s.text, s.size, s.t_in, s.t_out = text, size, t_in, t_out
        s.ent, s.ex, s.ex_dur, s.style, s.z, s.impact, s.echo = ent, ex, ex_dur, style, z, impact, echo
        s.slam_from, s.misreg, s.split_angle = slam_from, misreg, split_angle
        s.ent_dur = ent_dur if ent_dur is not None else {'slam': 0.075, 'type': 0.16, 'flicker': 0.16, 'split': 0.07, 'stamp': 0.1}[ent]
        s.rs = rs
        rng = np.random.default_rng(seed)
        s.rng_seed = seed
        px = size * rs; s.px = px
        f = font(fpath, round(px))
        asc, desc = f.getmetrics()
        bottom_pad = 1.9 if (drips or ex == 'melt') else 0.35
        pl, pt, pr, pb = int(0.3 * px), int(0.25 * px), int(0.3 * px), int(bottom_pad * px)
        xs = []; x = 0.
        for ch in text: xs.append(x); x += f.getlength(ch) + tracking * px
        Ww, Hh = int(x + pl + pr), int(asc + desc + pt + pb)
        img = Image.new('L', (Ww, Hh), 0); d = ImageDraw.Draw(img)
        for ch, xx in zip(text, xs): d.text((pl + xx, pt), ch, font=f, fill=255)
        A = np.asarray(img, np.float32) / 255.
        s.letter_x = [pl + xx for xx in xs] + [pl + x - tracking * px]
        s.origin_px = (pl, pt + asc)
        adv, ink = measure(text, fpath, size, tracking)
        s.adv, s.ink = adv, ink
        s.pivot = ((ink[0] + ink[2]) / 2, (ink[1] + ink[3]) / 2)  # world, rel. to origin
        s.origin = origin
        # --- distress (worn retro print)
        if distress > 0:
            n1 = fbm(Hh, Ww, max(2., px * 0.018), 3, rng)
            n2 = fbm(Hh, Ww, max(3., px * 0.06), 3, rng)
            Ab = cv2.GaussianBlur(A, (0, 0), max(0.6, px * 0.004))
            A = sstep(0.36, 0.64, Ab + (n1 - 0.5) * 0.30 * distress).astype(np.float32)
            holes = sstep(0.70, 0.78, n2 * 0.6 + n1 * 0.4)
            A *= (1 - 0.75 * distress * holes).astype(np.float32)
        s.A = A
        s.shape = A.shape
        s.tex = (0.82 + 0.18 * fbm(Hh, Ww, max(3., px * 0.05), 3, rng)).astype(np.float32)
        s.drips = make_drips(A, drips, px, rng, drip_t if drip_t is not None else t_in + 0.08, drip_spread, drip_len) if drips else []
        # melt column profile
        cols = fbm(1, Ww, max(4., px * 0.05), 3, rng)[0]
        spikes = np.zeros(Ww, np.float32)
        for _ in range(int(len(text) * 2.5)):
            c = rng.uniform(0, Ww); wd = px * rng.uniform(0.01, 0.05)
            spikes += rng.uniform(0.4, 1.0) * np.exp(-((np.arange(Ww) - c) / wd) ** 2)
        s.melt_cols = (0.25 + 0.5 * cols + np.clip(spikes, 0, 1)).astype(np.float32)
        s.ink_px = (pl, pt + asc + ink[1] * rs, pl + x, pt + asc + ink[3] * rs)
        s._static = None
        s.half_masks = None
        if ent == 'split':
            yy, xx_ = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
            cx, cy = s.origin_px[0] + s.pivot[0] * rs, s.origin_px[1] + s.pivot[1] * rs
            nxv, nyv = -math.sin(split_angle), math.cos(split_angle)
            side = (xx_ - cx) * nxv + (yy - cy) * nyv
            up = sstep(-1.0, 1.0, -side).astype(np.float32)
            s.half_masks = (up, 1 - up)

    # ------------------------------------------------------------ appearance in sprite space
    def rgba(s, t):
        dynamic = bool(s.drips) or (s.ex == 'melt' and t >= s.t_out) or (s.ent == 'type' and t < s.t_in + s.ent_dur + 0.02)
        if not dynamic and s._static is not None: return s._static
        A = s.A
        if s.drips:
            D, stains = draw_drips(s.shape, s.drips, t)
        else:
            D, stains = None, []
        if s.style == 'blood':
            U = np.maximum(A, D) if D is not None else A
            out = blood_shade(U, s.px / 160.)
        else:
            rgb = s.tex[..., None] * BONE[None, None, :]
            if stains:
                st = np.zeros(s.shape, np.float32)
                for (x, y, w, a) in stains:
                    cv2.circle(st, (int(x), int(y)), int(w * 2.2), float(a), -1, cv2.LINE_AA)
                st = cv2.GaussianBlur(st, (0, 0), max(1., s.px * 0.025))
                st = np.clip(st * 1.6, 0, 1)[..., None]
                stain_col = np.array([0.45, 0.03, 0.035], np.float32)
                rgb = rgb * (1 - 0.92 * st) + stain_col * st * s.tex[..., None]
            out = np.empty(s.shape + (4,), np.float32)
            out[..., :3] = rgb * A[..., None]; out[..., 3] = A
            if D is not None:
                B = blood_shade(D, s.px / 160.)
                out = B + out * (1 - D[..., None])
        # typewriter reveal
        if s.ent == 'type':
            u = (t - s.t_in) / s.ent_dur
            n = len(s.text)
            k = int(math.floor(cl(u) * n + 1e-6)) + (1 if u >= 0 else 0)
            k = min(k, n)
            xcut = s.letter_x[k] if k < len(s.letter_x) else s.letter_x[-1]
            if u < 1.0 + 0.02:
                mask = np.zeros(s.shape[1], np.float32); mask[:int(xcut)] = 1.
                if k > 0 and k <= n:  # newest letter flickers
                    x0 = int(s.letter_x[k - 1])
                    fr = int(t * FPS)
                    mask[x0:int(xcut)] = 0.55 + 0.45 * ((fr * 7 + k * 3) % 3 != 0)
                out = out * mask[None, :, None]
        if s.ex == 'melt' and t >= s.t_out:
            u = cl((t - s.t_out) / s.ex_dur)
            out = s._melt(out, u)
        if not dynamic: s._static = out
        return out

    def _melt(s, rgba, u):
        Hh, Ww = s.shape
        top, bot = s.ink_px[1], s.ink_px[3]
        yy = np.arange(Hh, dtype=np.float32)
        wy = np.clip((yy - top) / (bot - top + 1e-3), 0, 1.6) * 0.75 + 0.25
        amt = (u ** 1.6) * s.px * 0.9
        dy = amt * wy[:, None] * s.melt_cols[None, :]
        mx = np.broadcast_to(np.arange(Ww, dtype=np.float32)[None, :], (Hh, Ww)).copy()
        wob = (np.sin(yy * 0.02 + u * 6.0) * s.px * 0.012 * u)[:, None]
        my = (yy[:, None] - dy).astype(np.float32)
        out = cv2.remap(rgba, (mx + wob).astype(np.float32), my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        fade = 1 - sstep(0.45, 1.0, u)
        # darken toward blood as it melts
        if s.style != 'blood':
            k = sstep(0.0, 0.6, u) * 0.75
            tint = np.array([1 - 0.45 * k, 1 - 0.95 * k, 1 - 0.93 * k, 1], np.float32)
            out = out * tint
        return out * fade

    # ------------------------------------------------------------ transform
    def visible(s, t):
        start = s.t_in - (s.ent_dur if s.ent in ('slam', 'split') else 0)
        if t < start: return False
        if s.ex == 'cut' and t >= s.t_out: return False
        if s.ex in ('melt', 'fade') and t >= s.t_out + s.ex_dur: return False
        return True

    def base_M(s, scale=1., rot=0., dx=0., dy=0.):
        ox, oy = s.origin; px_, py_ = s.pivot
        return (T(ox + px_ + dx, oy + py_ + dy) @ Rot(rot) @ Sc(scale) @ T(-px_, -py_) @ Sc(1 / s.rs) @ T(-s.origin_px[0], -s.origin_px[1]))

    def draws(s, t):
        """list of (matrices[list], alpha, tint, mask) to draw"""
        out = []
        alpha = 1.
        if s.ex == 'fade' and t >= s.t_out: alpha *= 1 - e_in2((t - s.t_out) / s.ex_dur)
        u_in = (t - s.t_in)
        if s.ent == 'slam':
            d = s.ent_dur
            def sc(tt):
                u = cl((tt - (s.t_in - d)) / d)
                return 1 + (s.slam_from - 1) * (1 - u) ** 2.2
            if u_in < 0:
                a = sstep(0, 0.5, (t - (s.t_in - d)) / d)
                Ms = [s.base_M(sc(t - k / 6 / FPS * 1.0)) for k in range(6)]
                out.append((Ms, alpha * a, None, None))
                return out
            # landed: small rebound
            bump = 1 - 0.045 * math.sin(min(u_in / 0.12, 1) * math.pi) * dec(t, s.t_in, 0.2)
            M = s.base_M(bump)
        elif s.ent == 'split':
            d = s.ent_dur
            u = cl((t - (s.t_in - d)) / d)
            off = (1 - e_out3(u)) * s.size * 1.1 + s.size * 0.035 * dec(t, s.t_in, 0.12)
            ca, sa = math.cos(s.split_angle), math.sin(s.split_angle)
            a = sstep(0, 0.4, u)
            M1 = s.base_M(1., 0., -ca * off, -sa * off)
            M2 = s.base_M(1., 0., ca * off, sa * off)
            if s.misreg and u >= 1:
                Mr = s.base_M(1.0, 0., s.size * 0.022, s.size * 0.016)
                out.append(([Mr], alpha * 0.85, np.array([0.30, 0.0, 0.01], np.float32), None))
            out.append(([M1], alpha * a, None, 0))
            out.append(([M2], alpha * a, None, 1))
            if u_in >= 0 and s.echo: out += s._echo(t)
            return out
        elif s.ent == 'stamp':
            if u_in < 0: return out
            k = 1 + 0.10 * (1 - e_out3(u_in / s.ent_dur))
            M = s.base_M(k)
        elif s.ent == 'flicker':
            if u_in < 0: return out
            if u_in < s.ent_dur:
                fr = int(t * FPS)
                on = [1, 0, 1, 1, 0, 0, 1, 0.4, 1, 1, 0.6, 1][fr % 12]
                alpha *= on
            M = s.base_M(1.)
        else:  # type
            if u_in < 0: return out
            M = s.base_M(1.)
        if s.misreg and s.style != 'blood':
            Mr = s.base_M(1.0, 0., s.size * 0.022, s.size * 0.016) @ np.linalg.inv(s.base_M(1.)) @ M
            out.append(([Mr], alpha * 0.85, np.array([0.30, 0.0, 0.01], np.float32), None))
        out.append(([M], alpha, None, None))
        if s.echo and s.ent in ('slam', 'stamp') and u_in >= 0: out += s._echo(t)
        return out

    def _echo(s, t):
        u = (t - s.t_in) / 0.32
        if u >= 1 or u < 0: return []
        k = e_out3(u)
        M = s.base_M(1 + 0.30 * k * min(1.6, s.impact))
        col = np.array([0.75, 0.05, 0.05], np.float32) if s.style != 'blood' else np.array([0.9, 0.12, 0.08], np.float32)
        return [([M], 0.5 * (1 - u) ** 2, col, None)]


# ---------------------------------------------------------------- splatter
def make_splat(R, rng, kind='radial', length=6.):
    if kind == 'radial':
        Sz = int(R * 6.5); Hh = Ww = Sz
    else:
        Ww, Hh = int(R * (length + 3)), int(R * 3.2)
    img = np.zeros((Hh, Ww), np.uint8)
    F = 16; sh = 4
    cx, cy = Ww / 2, Hh / 2
    def circ(x, y, r):
        if r < 0.5: r = 0.5
        cv2.circle(img, (int(x * F), int(y * F)), int(r * F), 255, -1, cv2.LINE_AA, sh)
    def ell(x, y, a, b, ang):
        cv2.ellipse(img, (int(x * F), int(y * F)), (max(1, int(a * F)), max(1, int(b * F))), ang, 0, 360, 255, -1, cv2.LINE_AA, sh)
    if kind == 'radial':
        n = 220; th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rr = np.ones(n) * R * 0.55
        for k in range(1, 7):
            rr += R * 0.07 / k ** 0.6 * np.sin(k * th + rng.uniform(0, 6.3))
        for _ in range(int(rng.integers(9, 16))):
            a0 = rng.uniform(0, 2 * np.pi); wdt = rng.uniform(0.03, 0.10); hgt = R * rng.uniform(0.15, 0.85)
            dth = np.angle(np.exp(1j * (th - a0)))
            rr += hgt * np.exp(-(dth / wdt) ** 2)
        pts = np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1)
        cv2.fillPoly(img, [(pts * F).astype(np.int32)], 255, cv2.LINE_AA, sh)
        for _ in range(int(rng.integers(8, 15))):
            a = rng.uniform(0, 2 * np.pi); dd = R * rng.uniform(0.45, 0.85)
            circ(cx + dd * math.cos(a), cy + dd * math.sin(a), R * rng.uniform(0.06, 0.2))
        for _ in range(int(rng.integers(160, 260))):
            a = rng.uniform(0, 2 * np.pi); dd = R * (0.75 + rng.exponential(0.55))
            if dd > R * 3.1: continue
            sz = R * 0.11 * rng.random() ** 1.8 * max(0.25, 1.5 - dd / (2.2 * R))
            el = 1 + rng.uniform(0, 1.8) * (dd / R - 0.7)
            ell(cx + dd * math.cos(a), cy + dd * math.sin(a), sz * el, sz, math.degrees(a))
    else:
        L = R * length
        for i in range(int(rng.integers(40, 70))):
            u = rng.uniform(-0.5, 0.5); x = cx + u * L
            y = cy + rng.normal(0, R * 0.12) * (1.4 - abs(u))
            sz = R * 0.22 * rng.random() ** 1.2 * (1.2 - abs(u))
            ell(x, y, sz * rng.uniform(1.5, 4.5), sz * 0.7, rng.normal(0, 6))
        for _ in range(int(rng.integers(150, 240))):
            u = rng.uniform(-0.55, 0.55); x = cx + u * L
            y = cy + rng.normal(0, R * 0.55)
            sz = R * 0.07 * rng.random() ** 1.8
            ell(x, y, sz * rng.uniform(1, 3.5), sz, rng.normal(0, 10))
        pts = []
        for i in range(60):
            u = i / 59 - 0.5
            pts.append((cx + u * L, cy - R * 0.08 * (1 - (2 * u) ** 2) - R * 0.02))
        for i in range(60):
            u = 0.5 - i / 59
            pts.append((cx + u * L, cy + R * 0.08 * (1 - (2 * u) ** 2) + R * 0.02))
        cv2.fillPoly(img, [(np.array(pts) * F).astype(np.int32)], 255, cv2.LINE_AA, sh)
    A = img.astype(np.float32) / 255.
    A = cv2.GaussianBlur(A, (0, 0), max(0.8, R * 0.025))
    A = sstep(0.32, 0.6, A).astype(np.float32)
    return A


class Splat:
    def __init__(s, center, R, t_in, t_out=99., seed=0, kind='radial', angle=0., z=0, dim=0.6, rs=1.5,
                 ex='fade', ex_dur=0.4, length=6., grow=0.07, screen=False):
        rng = np.random.default_rng(seed)
        s.center, s.R, s.t_in, s.t_out, s.angle, s.z, s.rs = center, R, t_in, t_out, angle, z, rs
        s.ex, s.ex_dur, s.grow, s.screen = ex, ex_dur, grow, screen
        A = make_splat(R * rs, rng, kind, length)
        s.rgba_ = blood_shade(A, R * rs / 140., wet=0.9, dark=dim)
        s.c = (A.shape[1] / 2, A.shape[0] / 2)
    def visible(s, t):
        if t < s.t_in: return False
        if s.ex == 'cut' and t >= s.t_out: return False
        if t >= s.t_out + s.ex_dur: return False
        return True
    def rgba(s, t): return s.rgba_
    def draws(s, t):
        u = (t - s.t_in) / s.grow
        k = 0.35 + 0.65 * e_out3(u)
        a = sstep(0, 0.3, u)
        if t >= s.t_out: a *= 1 - cl((t - s.t_out) / s.ex_dur)
        M = T(*s.center) @ Rot(s.angle) @ Sc(k / s.rs) @ T(-s.c[0], -s.c[1])
        return [([M], a, None, None)]


# ---------------------------------------------------------------- camera
class Camera:
    def __init__(s, init): s.init = np.array(init, float); s.keys = []
    def key(s, t, d, cx, cy, z, ease=e_outexpo): s.keys.append((t, d, np.array([cx, cy, z], float), ease)); return s
    def at(s, t):
        prev = None; st = s.init
        for (t0, d, tgt, ease) in sorted(s.keys, key=lambda k: k[0]):
            if t0 > t: break
            if prev is not None: st = s._interp(prev, t0)
            prev = (t0, d, ease, st, tgt)
        return s._interp(prev, t) if prev is not None else st
    def cut_between(s, a, b):
        return any(d <= 0 and a < t0 <= b for (t0, d, _, _) in s.keys)

    @staticmethod
    def _interp(p, t):
        t0, d, ease, a, b = p
        u = 1. if d <= 0 else ease((t - t0) / d)
        return a + (b - a) * u


# ---------------------------------------------------------------- compositing
def draw_items(canvas, rgba, Ms, alpha, tint=None, mask=None):
    if alpha <= 0.003: return
    h, w = rgba.shape[:2]
    corners = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]], np.float64)
    xs, ys = [], []
    for M in Ms:
        c = M @ corners; xs += list(c[0]); ys += list(c[1])
    x0, x1 = max(0, int(math.floor(min(xs)))), min(W, int(math.ceil(max(xs))))
    y0, y1 = max(0, int(math.floor(min(ys)))), min(H, int(math.ceil(max(ys))))
    if x1 <= x0 or y1 <= y0: return
    src = rgba
    if mask is not None: src = src * mask[..., None]
    if tint is not None: src = src[..., 3]
    s = math.sqrt(abs(np.linalg.det(Ms[0][:2, :2])))
    pre = np.eye(3)
    if s < 0.72:
        f = min(1., s * 1.35)
        src = cv2.resize(src, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        pre = Sc(w / src.shape[1], h / src.shape[0])
    acc = None
    for M in Ms:
        Mr = (T(-x0, -y0) @ M @ pre)[:2]
        o = cv2.warpAffine(src, Mr, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        acc = o if acc is None else acc + o
    if len(Ms) > 1: acc *= 1. / len(Ms)
    roi = canvas[y0:y1, x0:x1]
    if tint is not None:
        a = acc * alpha
        roi *= (1 - a)[..., None]; roi += a[..., None] * tint[None, None, :]
    else:
        acc *= alpha
        roi *= (1 - acc[..., 3:4]); roi += acc[..., :3]


# ---------------------------------------------------------------- the piece
class Scene:
    pass


def build():
    sc = Scene()
    items = []; ev = dict(shake=[], punch=[], chroma=[], flash=[], glitch=[], invert=[], glow=[])
    cam = Camera((960, 540, 1.0))

    def row(specs, cx, cy, gap):
        ms = [measure(sp['text'], sp['font'], sp['size'], sp.get('tracking', 0.)) for sp in specs]
        xs = []; x = 0.
        for (adv, ink), sp in zip(ms, specs):
            xs.append(x); x += adv + sp.get('gap', gap)
        left = xs[0] + ms[0][1][0]; right = xs[-1] + ms[-1][1][2]
        big = max(range(len(specs)), key=lambda i: specs[i]['size'])
        ink = ms[big][1]
        base = cy - (ink[1] + ink[3]) / 2
        off = cx - (left + right) / 2
        res = []
        for xx, (adv, ink) in zip(xs, ms):
            o = (xx + off, base)
            res.append(dict(origin=o, center=(o[0] + (ink[0] + ink[2]) / 2, o[1] + (ink[1] + ink[3]) / 2), ink=ink))
        return res

    def impact(t, k):
        ev['shake'].append((t, 9 * k, 0.6 * k, 0.16 + 0.04 * k))
        ev['punch'].append((t, 0.028 * k, 0.11))
        ev['chroma'].append((t, 0.0045 * k, 0.12))

    # ---------------- intro: heartbeat + blade line
    for t, k in [(1.033, 0.35), (1.196, 0.55), (1.347, 0.85)]:
        ev['glow'].append((t, k, 0.14))
    sc.blade = (0.98, 1.654)

    # ---------------- shot 1 : NOW THERE'S / ONLY
    r1 = row([dict(text='NOW', font=FN, size=150), dict(text="THERE'S", font=FN, size=150)], 960, 450, 150 * 0.32)
    r2 = row([dict(text='ONLY', font=FN, size=250, tracking=0.10)], 960, 668, 0)
    end1 = 2.914 - 0.075
    items.append(Word('NOW', FN, 150, r1[0]['origin'], 1.654, end1, 'slam', seed=1, impact=0.9, slam_from=2.6))
    items.append(Word("THERE'S", FN, 150, r1[1]['origin'], 1.960, end1, 'type', ent_dur=0.15, seed=2, impact=0.4))
    items.append(Word('ONLY', FN, 250, r2[0]['origin'], 2.281, end1, 'slam', tracking=0.10, seed=3, impact=1.0))
    impact(1.654, 0.9); impact(1.974, 0.35); impact(2.281, 1.0); impact(2.601, 0.45)
    ev['punch'].append((2.601, 0.03, 0.12))
    c_now = r1[0]['center']; c_r1 = (960, 450)
    cam.key(1.584, 0, c_now[0], c_now[1], 2.0).key(1.585, 0.38, c_now[0] + 10, c_now[1], 2.08, lin)
    cam.key(1.955, 0.16, c_r1[0], c_r1[1], 1.36).key(2.12, 0.15, c_r1[0], c_r1[1], 1.39, lin)
    cam.key(2.245, 0.15, 960, 560, 1.02).key(2.40, 0.52, 960, 560, 1.07, lin)

    # ---------------- shot 2 : ONE / THING / I CAN DO
    ONE_c = (960, 300)
    rO = row([dict(text='ONE', font=FE, size=330)], ONE_c[0], ONE_c[1], 0)
    rT = row([dict(text='THING', font=FN, size=165, tracking=0.06)], 960, 572, 0)
    rI = row([dict(text='I', font=FN, size=150), dict(text='CAN', font=FN, size=150), dict(text='DO', font=FN, size=150)], 960, 772, 150 * 0.34)
    m1 = 6.072
    items.append(Word('ONE', FE, 330, rO[0]['origin'], 2.914, m1, 'slam', ex='melt', ex_dur=0.48, style='blood', drips=9,
                      drip_t=2.95, drip_spread=1.4, drip_len=1.1, seed=11, impact=2.0, slam_from=2.0, distress=0.15, z=12))
    items.append(Splat((ONE_c[0] - 30, ONE_c[1] + 10), 170, 2.914, m1, seed=21, dim=0.38, z=1))
    items.append(Splat((ONE_c[0] + 330, ONE_c[1] - 90), 70, 2.93, m1, seed=22, dim=0.34, z=1))
    items.append(Splat((ONE_c[0] - 380, ONE_c[1] + 120), 55, 2.95, m1, seed=23, dim=0.34, z=1))
    impact(2.914, 2.0); ev['invert'].append((2.914, 1)); ev['flash'].append((2.914, 0.55, 0.22)); ev['glitch'].append((2.914, 3, 0.7, 5))
    ev['punch'].append((3.239, 0.02, 0.1))
    items.append(Word('THING', FN, 165, rT[0]['origin'], 3.547, m1, 'slam', ex='melt', tracking=0.06, seed=12, impact=1.0))
    impact(3.547, 1.0)
    items.append(Word('I', FN, 150, rI[0]['origin'], 4.165, m1, 'stamp', ex='melt', seed=13, impact=0.6))
    impact(4.165, 0.7)
    items.append(Word('CAN', FN, 150, rI[1]['origin'], 4.801, m1, 'slam', ex='melt', seed=14, impact=0.9))
    impact(4.801, 0.9)
    items.append(Word('DO', FN, 150, rI[2]['origin'], 5.439, m1, 'slam', ex='melt', seed=15, impact=1.3, drips=4, drip_t=5.5, drip_len=0.8))
    items.append(Splat((rI[2]['center'][0] + 10, rI[2]['center'][1] + 5), 85, 5.439, m1, seed=24, dim=0.32, z=1))
    impact(5.439, 1.3); ev['flash'].append((5.439, 0.2, 0.15))
    ev['punch'].append((6.072, 0.015, 0.2)); ev['punch'].append((6.391, 0.01, 0.2))
    cam.key(end1, 0, ONE_c[0], ONE_c[1] + 20, 1.62).key(end1 + 0.01, 0.7, ONE_c[0], ONE_c[1] + 22, 1.72, lin)
    mid = (ONE_c[1] + rT[0]['center'][1]) / 2
    cam.key(3.52, 0.17, 960, mid + 10, 1.24).key(3.70, 0.45, 960, mid + 12, 1.27, lin)
    cam.key(4.14, 0.18, 960, 545, 1.0).key(4.33, 2.25, 960, 545, 1.08, lin)
    cam.key(6.072, 0.5, 960, 575, 1.14, lin)

    # ---------------- shot 3 : FIGHT / UNTIL THE END
    rF = row([dict(text='FIGHT', font=FN, size=280, tracking=0.05)], 960, 405, 0)
    rU = row([dict(text='UN', font=FN, size=150, gap=0), dict(text='TIL', font=FN, size=150),
              dict(text='THE', font=FN, size=150), dict(text='END', font=FN, size=150)], 960, 672, 150 * 0.34)
    m3 = 8.95
    ang = math.radians(-14)
    sc.slash = (6.575, 6.70, ang, rF[0]['center'])
    items.append(Word('FIGHT', FN, 280, rF[0]['origin'], 6.699, m3, 'split', ex='cut', tracking=0.05, seed=31,
                      impact=1.8, drips=5, drip_t=6.78, drip_len=0.9, split_angle=ang))
    items.append(Splat(rF[0]['center'], 95, 6.67, m3, seed=41, kind='line', angle=ang, dim=0.34, z=1, length=7.5, grow=0.05))
    impact(6.699, 1.8); ev['glitch'].append((6.699, 4, 1.0, 7)); ev['flash'].append((6.699, 0.35, 0.18))
    items.append(Word('UN', FN, 150, rU[0]['origin'], 7.024, m3, 'slam', seed=32, impact=0.5, slam_from=1.8))
    impact(7.024, 0.55)
    items.append(Word('TIL', FN, 150, rU[1]['origin'], 7.332, m3, 'slam', seed=33, impact=0.8, slam_from=1.8))
    impact(7.332, 0.8)
    items.append(Word('THE', FN, 150, rU[2]['origin'], 7.645, m3, 'type', ent_dur=0.10, seed=34))
    impact(7.663, 0.35)
    items.append(Word('END', FN, 150, rU[3]['origin'], 7.964, m3, 'slam', seed=35, impact=1.5, drips=5, drip_t=8.0, drip_len=1.0))
    items.append(Splat((rU[3]['center'][0], rU[3]['center'][1]), 100, 7.964, m3, seed=42, dim=0.34, z=1))
    impact(7.964, 1.5); ev['glitch'].append((7.964, 3, 0.6, 9)); ev['flash'].append((7.964, 0.3, 0.16))
    ev['punch'].append((8.290, 0.012, 0.1))
    cam.key(6.575, 0, rF[0]['center'][0], rF[0]['center'][1] + 10, 1.45).key(6.58, 0.42, rF[0]['center'][0], rF[0]['center'][1] + 12, 1.5, lin)
    cam.key(7.00, 0.2, 960, 545, 1.12).key(7.21, 0.72, 960, 545, 1.15, lin)
    cam.key(7.94, 0.14, 960, 545, 1.0).key(8.09, 0.4, 960, 545, 1.035, lin)

    # ---------------- shot 4 : LIKE I / PROMISED / TO  (lives below shot 3: whip pan down)
    Y4 = 1250
    rL = row([dict(text='LIKE', font=FN, size=135), dict(text='I', font=FN, size=135)], 960, Y4 + 300, 135 * 0.34)
    PS = 268
    rP = row([dict(text='PRO', font=FE, size=PS, gap=0), dict(text='MI', font=FE, size=PS, gap=0), dict(text='SED', font=FE, size=PS)], 960, Y4 + 548, 0)
    rTo = row([dict(text='TO', font=FN, size=150, tracking=0.08)], 960, Y4 + 800, 0)
    end4 = MUSIC_END
    items.append(Word('LIKE', FN, 135, rL[0]['origin'], 8.515, end4, 'flicker', seed=51, ex='cut'))
    ev['glitch'].append((8.515, 2, 0.6, 11))
    items.append(Word('I', FN, 135, rL[1]['origin'], 8.922, end4, 'slam', seed=52, impact=0.6, slam_from=1.9))
    impact(8.922, 0.6)
    for i, (txt, tt, imp, nd, sd) in enumerate([('PRO', 9.236, 2.2, 4, 61), ('MI', 9.561, 1.2, 3, 62), ('SED', 9.863, 1.8, 4, 63)]):
        items.append(Word(txt, FE, PS, rP[i]['origin'], tt, end4, 'slam', ex='fade', ex_dur=0.72, style='blood', drips=nd,
                          drip_t=tt + 0.05, drip_spread=0.9, drip_len=1.15, seed=sd, impact=imp, slam_from=1.9 + 0.2 * (i == 0),
                          distress=0.15, z=12))
        impact(tt, imp)
    pc = [r['center'] for r in rP]
    items.append(Splat((pc[0][0] - 40, pc[0][1] + 10), 190, 9.236, end4, seed=71, dim=0.38, z=1, ex_dur=0.7))
    items.append(Splat((pc[1][0] + 20, pc[1][1] - 40), 110, 9.561, end4, seed=72, dim=0.34, z=1, ex_dur=0.7))
    items.append(Splat((pc[2][0] + 60, pc[2][1] + 20), 170, 9.863, end4, seed=73, dim=0.38, z=1, ex_dur=0.7))
    items.append(Splat((pc[2][0] + 520, pc[2][1] - 230), 60, 9.88, end4, seed=74, dim=0.34, z=1, ex_dur=0.7))
    ev['invert'].append((9.236, 1)); ev['flash'].append((9.236, 0.6, 0.25)); ev['glitch'].append((9.236, 2, 0.8, 13))
    ev['flash'].append((9.561, 0.25, 0.14)); ev['glitch'].append((9.863, 3, 0.9, 15)); ev['flash'].append((9.863, 0.4, 0.2))
    items.append(Word('TO', FN, 150, rTo[0]['origin'], 10.188, end4, 'slam', tracking=0.08, seed=53, impact=0.8))
    impact(10.188, 0.8)
    impact(10.495, 1.6); ev['flash'].append((10.495, 0.35, 0.3))
    ev['punch'].append((10.815, 0.015, 0.12))
    sc.lens = (10.495, 81)
    yL = rL[0]['center'][1]
    cam.key(8.455, 0.13, 960, yL + 15, 1.6, e_inout).key(8.59, 0.3, 960, yL + 15, 1.66, lin)
    cam.key(8.90, 0.14, (rL[0]['center'][0] + rL[1]['center'][0]) / 2 + 30, yL + 20, 1.55).key(9.05, 0.17, 960, yL + 20, 1.57, lin)
    cam.key(9.205, 0.10, 760, Y4 + 500, 1.34).key(9.31, 0.25, 765, Y4 + 500, 1.38, lin)
    cam.key(9.53, 0.10, 860, Y4 + 515, 1.22).key(9.64, 0.2, 865, Y4 + 515, 1.25, lin)
    cam.key(9.835, 0.11, 960, Y4 + 530, 1.08).key(9.95, 0.22, 960, Y4 + 530, 1.11, lin)
    cam.key(10.16, 0.22, 960, Y4 + 560, 0.98).key(10.39, 0.66, 960, Y4 + 560, 1.02, lin)
    cam.key(11.05, 0.8, 960, Y4 + 548, 1.07, lin)

    # kick / offbeat pulses
    for k in KICKS: ev['punch'].append((k, 0.012, 0.09)); ev['glow'].append((k, 0.55, 0.16))
    for o in OFFS: ev['punch'].append((o, 0.005, 0.07)); ev['glow'].append((o, 0.22, 0.10))

    items.sort(key=lambda it: it.z)
    sc.items, sc.ev, sc.cam = items, ev, cam
    return sc


# ---------------------------------------------------------------- frame-level fx
class FX:
    def __init__(s):
        rng = np.random.default_rng(7)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        s.vign = (1 - 0.62 * np.clip(r / 1.25, 0, 1) ** 2.1).astype(np.float32)[..., None]
        s.glow = np.exp(-(((xx - W / 2) / (W * 0.42)) ** 2 + ((yy - H / 2) / (H * 0.48)) ** 2)).astype(np.float32)[..., None]
        PADX, PADY = 40, 40
        n = fbm(H + 2 * PADY, W + 2 * PADX, 160, 5, rng)
        n2 = fbm(H + 2 * PADY, W + 2 * PADX, 14, 3, rng)
        base = (0.014 + 0.030 * n ** 1.6 + 0.008 * n2)
        s.bg = np.stack([base * 1.25, base * 0.82, base * 0.78], -1).astype(np.float32)
        s.pad = (PADX, PADY)
        s.grain = []
        for i in range(10):
            g = rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
            g += rng.normal(0, 0.6, (H, W)).astype(np.float32)
            s.grain.append(g[..., None])
        s.nx, s.ny, s.nr, s.nf = Noise1(1), Noise1(2), Noise1(3), Noise1(4)
        s.wx, s.wy = Noise1(5), Noise1(6)
        # ash particles
        s.ash = rng.random((110, 6))


def shake_at(sc, fx, t):
    sx = sy = sr = 0.
    for (t0, a, ar, tau) in sc.ev['shake']:
        if t < t0 or t > t0 + tau * 6: continue
        e = dec(t, t0, tau)
        sx += a * e * fx.nx(t * 1.0, 9.0 + t0); sy += a * e * fx.ny(t, 8.0 + t0); sr += ar * e * fx.nr(t, 7.0)
    return sx, sy, math.radians(sr)


def punch_at(sc, t):
    z = 1.
    for (t0, a, tau) in sc.ev['punch']:
        if t0 <= t < t0 + tau * 8: z += a * dec(t, t0, tau)
    return z


def cam_matrix(sc, fx, t, shake=True):
    cx, cy, z = sc.cam.at(t)
    z *= punch_at(sc, t)
    sx, sy, sr = shake_at(sc, fx, t) if shake else (0, 0, 0)
    wx, wy = fx.wx(t, 1.5) * 1.2, fx.wy(t, 1.3) * 1.6
    return T(W / 2 + sx + wx, H / 2 + sy + wy) @ Rot(sr) @ Sc(z) @ T(-cx, -cy), (wx, wy), z


def render(sc, fx, fi):
    t = fi / FPS + VLEAD
    rng = np.random.default_rng(1000 + fi)
    C, (wx, wy), z = cam_matrix(sc, fx, t)
    PX, PY = fx.pad
    ox, oy = int(round(PX - wx * 0.5)), int(round(PY - wy * 0.5))
    frame = fx.bg[oy:oy + H, ox:ox + W].copy()
    # glow pulses
    g = 0.10
    for (t0, k, tau) in sc.ev['glow']:
        if t0 <= t < t0 + tau * 8: g += k * dec(t, t0, tau)
    if t < 1.0: g *= 0.25 + 0.75 * sstep(0.2, 1.0, t)
    if t > MUSIC_END: g *= 1 - sstep(MUSIC_END, MUSIC_END + 0.5, t)
    frame += fx.glow * (np.array([0.075, 0.006, 0.006], np.float32) * g)

    # blade line intro
    b0, b1 = sc.blade
    if b0 <= t < b1 + 0.08:
        lay = np.zeros((H, W), np.float32)
        u = e_in2((t - b0) / (b1 - b0))
        half = 40 + u * (W * 0.6)
        yb = H / 2 + fx.ny(t, 3) * 0.6
        th = 1 + 2 * u
        cv2.line(lay, (int(W / 2 - half), int(yb)), (int(W / 2 + half), int(yb)), 1.0, int(th), cv2.LINE_AA)
        a = (0.35 + 0.65 * u) * (1 - sstep(b1, b1 + 0.07, t)) * (0.85 + 0.15 * math.sin(t * 90))
        gl = cv2.GaussianBlur(lay, (0, 0), 6 + 10 * u)
        frame += (lay[..., None] * np.array([1.0, 0.85, 0.8], np.float32) + gl[..., None] * np.array([2.2, 0.15, 0.1], np.float32)) * a

    # scene items
    for it in sc.items:
        if not it.visible(t): continue
        rg = it.rgba(t)
        for (Ms, a, tint, mask) in it.draws(t):
            Ms2 = [C @ M for M in Ms]
            mk = None
            if mask is not None and getattr(it, 'half_masks', None) is not None: mk = it.half_masks[mask]
            draw_items(frame, rg, Ms2, a, tint, mk)

    # slash streak (screen space)
    if hasattr(sc, 'slash'):
        s0, s1, ang, wc = sc.slash
        if s0 <= t < s1 + 0.2:
            p = C @ np.array([wc[0], wc[1], 1.])
            ca, sa = math.cos(ang), math.sin(ang)
            L = 1500
            u = e_out3((t - s0) / (s1 - s0))
            head = -L + 2 * L * u
            tail = -L + 2 * L * e_in2((t - s0 - 0.03) / (s1 - s0 + 0.12))
            lay = np.zeros((H, W), np.float32)
            pa = (int(p[0] + ca * tail), int(p[1] + sa * tail)); pb = (int(p[0] + ca * head), int(p[1] + sa * head))
            cv2.line(lay, pa, pb, 1.0, 5, cv2.LINE_AA)
            a = 1 - sstep(s1, s1 + 0.16, t)
            gl = cv2.GaussianBlur(lay, (0, 0), 14)
            frame += (lay[..., None] * np.array([1.0, 0.9, 0.85], np.float32) + gl[..., None] * np.array([3.0, 0.2, 0.15], np.float32)) * a

    # camera motion blur (whip)
    C2, _, _ = cam_matrix(sc, fx, t - 1 / FPS, shake=False)
    C1, _, _ = cam_matrix(sc, fx, t, shake=False)
    v = (C1 @ np.array([960, 540 + 1250 * 0.5, 1.]) - C2 @ np.array([960, 540 + 1250 * 0.5, 1.]))
    vlen = 0. if sc.cam.cut_between(t - 1 / FPS, t) else abs(v[1]) * 0.8
    if vlen > 6:
        k = int(min(vlen, 260)) | 1
        ker = np.zeros((k, 1), np.float32); ker[:] = 1. / k
        frame = cv2.filter2D(frame, -1, ker)

    # ash
    for p in fx.ash:
        x0, y0, sp, sz, ph, br = p
        x = (x0 * W + 25 * math.sin(t * (0.4 + ph) + ph * 6)) % W
        y = (y0 * (H + 60) - t * (12 + sp * 30)) % (H + 60) - 30
        val = (0.05 + 0.20 * br ** 2)
        cv2.circle(frame, (int(x), int(y)), int(1 + sz * 1.6), (val * 1.0, val * 0.8, val * 0.7), -1)

    # bloom
    small = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 0.30, 0, None)
    bl = cv2.GaussianBlur(br, (0, 0), 9) * 0.55 + cv2.GaussianBlur(br, (0, 0), 28) * 0.45
    frame += cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([0.9, 0.55, 0.5], np.float32)

    # lens blood (screen space, in front)
    if hasattr(sc, 'lens'):
        lt, ls = sc.lens
        if lt <= t:
            if not hasattr(fx, 'lens_items'):
                lr = np.random.default_rng(ls)
                fx.lens_items = [Splat((x, y), r, lt, MUSIC_END, seed=int(lr.integers(1e6)), dim=0.85, rs=1.0, ex_dur=0.75, grow=0.04)
                                 for (x, y, r) in [(140, 160, 34), (1790, 900, 46), (1720, 120, 22), (230, 960, 26), (1580, 1010, 16)]]
                for li in fx.lens_items:
                    li.drips = None
            for li in fx.lens_items:
                if li.visible(t):
                    for (Ms, a, tint, mask) in li.draws(t):
                        Mslide = T(0, 60 * max(0, t - lt) ** 1.5) @ Ms[0]
                        draw_items(frame, li.rgba(t), [Mslide], a * 0.95)

    # chromatic aberration
    c = 0.0012
    for (t0, a, tau) in sc.ev['chroma']:
        if t0 <= t < t0 + tau * 8: c += a * dec(t, t0, tau)
    if c > 0.0005:
        R = cv2.warpAffine(frame[..., 0], (T(W / 2, H / 2) @ Sc(1 + c) @ T(-W / 2, -H / 2))[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        B = cv2.warpAffine(frame[..., 2], (T(W / 2, H / 2) @ Sc(1 - c) @ T(-W / 2, -H / 2))[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frame[..., 0] = R; frame[..., 2] = B

    # glitch slices
    for (t0, nfr, st, gs) in sc.ev['glitch']:
        f0 = int(math.ceil((t0 - VLEAD) * FPS - 1e-6))
        if f0 <= fi < f0 + nfr:
            gr = np.random.default_rng(gs * 100 + fi)
            for _ in range(int(gr.integers(5, 11))):
                y = int(gr.integers(0, H - 10)); hh = int(gr.integers(6, 70))
                dx = int(gr.normal(0, 60 * st))
                band = frame[y:y + hh]
                band[:] = np.roll(band, dx, axis=1)
                if gr.random() < 0.4:
                    band *= gr.uniform(1.15, 1.5)

    for (t0, nfr) in sc.ev['invert']:
        f0 = int(math.ceil((t0 - VLEAD) * FPS - 1e-6))
        if f0 <= fi < f0 + nfr:
            v = np.clip(frame.max(-1, keepdims=True) * 1.9 - 0.06, 0, 1)
            paper = np.array([0.93, 0.87, 0.78], np.float32); ink = np.array([0.30, 0.015, 0.02], np.float32)
            frame = paper * (1 - v) + ink * v

    # flashes
    fl = 0.
    for (t0, a, tau) in sc.ev['flash']:
        if t0 <= t < t0 + tau * 8: fl += a * dec(t, t0, tau)
    if fl > 0.003:
        frame = frame + (fx.glow * 0.45 + 0.20) * np.array([0.42, 0.012, 0.016], np.float32) * fl
    # end fade
    if t > MUSIC_END + 0.55:
        frame *= 1 - sstep(MUSIC_END + 0.55, DUR - 0.05, t)

    # film: flicker, vignette, grain, scratches, dust
    flick = 1 + 0.035 * fx.nf(t, 9.0) + (rng.random() < 0.04) * (-0.08)
    frame *= flick
    frame *= fx.vign
    lum = np.clip(frame.mean(-1, keepdims=True), 0, 1)
    frame += fx.grain[int(rng.integers(0, len(fx.grain)))] * (0.016 + 0.07 * lum)
    sr = np.random.default_rng(fi // 3)
    if sr.random() < 0.45:
        for _ in range(int(sr.integers(1, 3))):
            x = int(sr.uniform(80, W - 80) + rng.normal(0, 1.5)); y0 = int(sr.uniform(-200, H * 0.5)); y1 = int(y0 + sr.uniform(H * 0.4, H * 1.2))
            val = sr.uniform(0.04, 0.10) * (1 if sr.random() < 0.6 else -1)
            cv2.line(frame, (x, y0), (x + int(sr.normal(0, 4)), y1), (val, val * 0.95, val * 0.9), 1, cv2.LINE_AA)
    for _ in range(int(rng.poisson(1.6))):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H)); r = int(rng.integers(1, 4))
        val = rng.uniform(0.08, 0.25) * (1 if rng.random() < 0.5 else -0.6)
        cv2.ellipse(frame, (x, y), (r, int(r * rng.uniform(0.6, 2.5))), rng.uniform(0, 180), 0, 360, (val, val * 0.95, val * 0.9), -1, cv2.LINE_AA)
    if rng.random() < 0.06:
        x, y = int(rng.uniform(100, W - 100)), int(rng.uniform(100, H - 100))
        pts = np.cumsum(rng.normal(0, 6, (14, 2)), 0) + [x, y]
        cv2.polylines(frame, [pts.astype(np.int32)], False, (0.1, 0.09, 0.085), 1, cv2.LINE_AA)
    # grade: crush, warm shadows
    frame = np.clip(frame, 0, 1)
    frame = frame ** 1.06
    frame = frame * 0.985 + np.array([0.008, 0.003, 0.003], np.float32)
    return (np.clip(frame, 0, 1) * 255 + rng.random((H, W, 1)) * 0.9).astype(np.uint8)


# ---------------------------------------------------------------- main
SC = FXO = None

def _init():
    global SC, FXO
    if SC is None:
        SC = build(); FXO = FX()

def _job(fi):
    return fi, render(SC, FXO, fi).tobytes()


if __name__ == '__main__':
    import multiprocessing as mp
    mode = sys.argv[1] if len(sys.argv) > 1 else 'video'
    _init()
    if mode == 'still':
        times = [float(x) for x in sys.argv[2].split(',')]
        out = sys.argv[3]
        ims = []
        for tt in times:
            fi = int(round(tt * FPS))
            im = render(SC, FXO, fi)
            cv2.putText(im, f'{tt:.3f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            ims.append(im)
        cols = 3 if len(ims) > 2 else len(ims)
        rows_ = (len(ims) + cols - 1) // cols
        th, tw = H // 2, W // 2
        sheet = np.zeros((rows_ * th, cols * tw, 3), np.uint8)
        for i, im in enumerate(ims):
            r_, c_ = divmod(i, cols)
            sheet[r_ * th:(r_ + 1) * th, c_ * tw:(c_ + 1) * tw] = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        if len(ims) == 1: cv2.imwrite(out.replace('.png', '_full.png'), cv2.cvtColor(ims[0], cv2.COLOR_RGB2BGR))
    else:
        out = sys.argv[2]
        f0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        f1 = int(sys.argv[4]) if len(sys.argv) > 4 else NF
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
               '-ss', f'{f0 / FPS:.4f}', '-i', AUDIO, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
               '-pix_fmt', 'yuv420p', '-tune', 'grain', '-c:a', 'aac', '-b:a', '320k', '-shortest', '-movflags', '+faststart', out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        ctx = mp.get_context('fork')
        with ctx.Pool(os.cpu_count()) as pool:
            for k, (fi, buf) in enumerate(pool.imap(_job, range(f0, f1), chunksize=2)):
                p.stdin.write(buf)
                if k % 60 == 0: print('frame', fi, flush=True)
        p.stdin.close(); p.wait()
        print('done', out)
