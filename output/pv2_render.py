"""Lyric PV #2: 'This could be the day' -- melancholy / despair. Black + red, dying eclipse, words burning to ash."""
import numpy as np, cv2, math, os, sys, subprocess
from PIL import Image, ImageDraw
import pv_render as pv
from pv_render import (W, H, FPS, T, Sc, Rot, cl, sstep, e_out3, e_out5, e_inout, e_in2, lin, dec, Noise1, fbm, font, measure,
                blood_shade, make_drips, draw_drips, draw_items)

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
FI = HERE + '/fonts/IM_Fell_English_ital_1.ttf'
AUDIO = HERE + '/work2/audio.wav'
DUR = 7.29
NF = int(round(DUR * FPS))
VLEAD = 0.010
ASH = np.array([0.80, 0.76, 0.70], np.float32)

KICKS = [0.615, 1.248, 1.881, 2.502, 3.135, 3.768, 4.390, 5.033, 5.666, 6.300]
DAY_T = 3.760
BIG_T = 4.390
MUSIC_END = 6.45

ENV = np.load(HERE + '/work2/env.npz')
_vt, _vr = ENV['t'], cv2.GaussianBlur(ENV['vr'].astype(np.float32).reshape(1, -1), (0, 0), 4).ravel()
_ar = cv2.GaussianBlur(ENV['ar'].astype(np.float32).reshape(1, -1), (0, 0), 3).ravel()
def venv(t): return float(np.interp(t, _vt, _vr))
def aenv(t): return float(np.interp(t, _vt, _ar))


# ---------------------------------------------------------------- word with soft entrance + burn-to-ash
class Word2:
    def __init__(s, text, size, center, t_in, style='ash', rs=2.0, seed=0, burn=None, burn_mode='ltr',
                 drips=0, drip_t=None, n_part=900, ent_blur=9., ent_drop=0.14, ent_scale=1.0, flash=0.6):
        s.text, s.size, s.t_in, s.style, s.rs = text, size, t_in, style, rs
        s.burn, s.flash, s.ent_blur, s.ent_drop, s.ent_scale = burn, flash, ent_blur, ent_drop, ent_scale
        rng = np.random.default_rng(seed)
        px = size * rs; s.px = px
        f = font(FI, round(px))
        asc, desc = f.getmetrics()
        pb = 1.6 if drips else 0.45
        pl, pt, pr, pbp = int(0.45 * px), int(0.3 * px), int(0.45 * px), int(pb * px)
        adv = f.getlength(text)
        Ww, Hh = int(adv + pl + pr), int(asc + desc + pt + pbp)
        img = Image.new('L', (Ww, Hh), 0); ImageDraw.Draw(img).text((pl, pt), text, font=f, fill=255)
        A = np.asarray(img, np.float32) / 255.
        n1 = fbm(Hh, Ww, max(2., px * 0.02), 3, rng)
        Ab = cv2.GaussianBlur(A, (0, 0), max(0.6, px * 0.004))
        A = sstep(0.38, 0.62, Ab + (n1 - 0.5) * 0.22).astype(np.float32)
        s.A, s.shape = A, A.shape
        s.tex = (0.80 + 0.20 * fbm(Hh, Ww, max(3., px * 0.05), 3, rng)).astype(np.float32)
        ys, xs = np.nonzero(A > 0.5)
        s.ink = (xs.min(), ys.min(), xs.max(), ys.max())
        s.anchor = ((s.ink[0] + s.ink[2]) / 2, (s.ink[1] + s.ink[3]) / 2)
        s.center = center
        s.drips = make_drips(A, drips, px, rng, drip_t if drip_t is not None else t_in + 0.6, 0.9, 1.3) if drips else []
        for d in s.drips: d.dur *= 1.8; d.fall = True
        # burn field: 0..1, low burns first
        nb = fbm(Hh, Ww, max(3., px * 0.07), 4, rng)
        xn = (np.arange(Ww, dtype=np.float32) - s.ink[0]) / max(1, s.ink[2] - s.ink[0])
        yn = (np.arange(Hh, dtype=np.float32) - s.ink[1]) / max(1, s.ink[3] - s.ink[1])
        if burn_mode == 'ltr': g = np.broadcast_to(np.clip(xn, 0, 1)[None, :], (Hh, Ww))
        else: g = np.broadcast_to(np.clip(1 - yn, 0, 1)[:, None], (Hh, Ww))
        s.field = (0.55 * g + 0.45 * nb).astype(np.float32)
        # ash particles sampled from the glyph
        ys, xs = np.nonzero(A > 0.45)
        k = min(n_part, len(xs)); idx = rng.choice(len(xs), k, replace=False)
        s.pp = np.stack([xs[idx] + rng.random(k), ys[idx] + rng.random(k)], 1).astype(np.float32)
        s.pf = s.field[ys[idx], xs[idx]]
        s.pv = np.stack([rng.normal(0, 1, k), rng.normal(0, 1, k), rng.uniform(0.6, 1.4, k), rng.uniform(0, 6.28, k),
                         rng.uniform(0.7, 1.6, k)], 1).astype(np.float32)
        s._cache = None

    def burn_p(s, t):
        if s.burn is None: return -1.
        b0, b1 = s.burn
        return -0.08 + 1.2 * cl((t - b0) / (b1 - b0)) ** 0.9

    def M(s, t, cam):
        u = t - s.t_in
        drop = -s.ent_drop * s.size * (1 - e_out5(cl((u + 0.06) / 0.7)))
        sc = 1 + (s.ent_scale - 1) * (1 - e_out3(cl((u + 0.06) / 0.9)))
        sink = 0.
        if s.burn is not None and t > s.burn[0]: sink = (t - s.burn[0]) ** 2 * 6.
        cx, cy = s.center
        return cam @ T(cx, cy + drop + sink) @ Sc(sc / s.rs) @ T(-s.anchor[0], -s.anchor[1])

    def alpha(s, t):
        u = t - s.t_in
        return cl((u + 0.06) / 0.06) * 0.45 + 0.55 * e_out3(cl(u / 0.22))

    def rgba(s, t):
        A = s.A
        D = None
        if s.drips:
            D, _ = draw_drips(s.shape, s.drips, t)
        p = s.burn_p(t)
        key = (p < -0.05) and not s.drips
        if key and s._cache is not None:
            out = s._cache
        else:
            if s.style == 'blood':
                U = np.maximum(A, D) if D is not None else A
                out = blood_shade(U, s.px / 170., wet=0.75, dark=0.95)
                # inner ember glow
                g = cv2.GaussianBlur(U, (0, 0), s.px * 0.02)
                out[..., 0] += 0.10 * g * U
            else:
                out = np.empty(s.shape + (4,), np.float32)
                out[..., :3] = (s.tex * A)[..., None] * ASH[None, None, :]
                out[..., 3] = A
                if D is not None:
                    B = blood_shade(D, s.px / 170.)
                    out = B + out * (1 - D[..., None])
            if key: s._cache = out
        if p > -0.05:
            edge = 0.07
            keep = sstep(p, p + 0.02, s.field)
            band = (sstep(p - 0.005, p + 0.03, s.field) * (1 - sstep(p + 0.03, p + edge + 0.04, s.field))).astype(np.float32)
            out = out * keep[..., None]
            a = out[..., 3]
            emb = band * np.maximum(a, (A > 0.3).astype(np.float32) * keep)
            out = out.copy()
            out[..., 0] += emb * 1.25; out[..., 1] += emb * 0.28; out[..., 2] += emb * 0.12
            out[..., 3] = np.maximum(out[..., 3], emb * 0.9)
        u = t - s.t_in
        bl = s.ent_blur * (1 - e_out3(cl((u + 0.06) / 0.45)))
        if bl > 0.4:
            out = cv2.GaussianBlur(out, (0, 0), bl * s.rs)
        return out

    def particles(s, t, cam):
        """burnt-off flakes: returns screen positions (N,2), intensity (N,), ember (N,)"""
        p = s.burn_p(t)
        if p < 0: return None
        b0, b1 = s.burn
        rel = b0 + np.clip((s.pf + 0.08) / 1.2, 0, 1) ** (1 / 0.9) * (b1 - b0)
        tau = t - rel
        m = tau > 0
        if not m.any(): return None
        tau = tau[m]; pp = s.pp[m]; v = s.pv[m]
        life = 1.6 * v[:, 4]
        alive = tau < life
        if not alive.any(): return None
        tau, pp, v, life = tau[alive], pp[alive], v[alive], life[alive]
        sz = s.size / 90.
        dx = (v[:, 0] * 28 * tau + 14 * np.sin(tau * 2.2 * v[:, 2] + v[:, 3]) * tau) * sz
        dy = (-v[:, 1].clip(-0.5, 3) * 10 * tau - 22 * tau + 70 * tau ** 1.6 * v[:, 2]) * sz
        Mw = s.M(t, cam)
        P = np.stack([pp[:, 0], pp[:, 1], np.ones(len(pp), np.float32)], 0)
        Q = (Mw @ P)[:2].T
        sc = math.sqrt(abs(np.linalg.det(cam[:2, :2])))
        Q[:, 0] += dx * sc; Q[:, 1] += dy * sc
        fade = (1 - tau / life) ** 1.3
        ember = np.exp(-tau / (0.35 * v[:, 2]))
        return Q, fade, ember


# ---------------------------------------------------------------- eclipse
def periodic_noise(nt, nr, at, ar, beta, rng):
    w = rng.normal(0, 1, (nr, nt))
    F = np.fft.fft2(w)
    kr = np.fft.fftfreq(nr)[:, None] * nr; kt = np.fft.fftfreq(nt)[None, :] * nt
    k = np.sqrt((kt * at) ** 2 + (kr * ar) ** 2) + 1.
    F *= k ** (-beta)
    n = np.real(np.fft.ifft2(F)).astype(np.float32)
    return (n - n.mean()) / (n.std() + 1e-6)


class Eclipse:
    def __init__(s, center, R):
        rng = np.random.default_rng(5)
        s.center, s.R = center, R
        s.NT, s.NR = 1024, 512
        s.streak = periodic_noise(s.NT, s.NR, 0.30, 12.0, 1.1, rng)
        s.fine = periodic_noise(s.NT, s.NR, 0.5, 2.0, 0.85, rng)
        s.h, s.w = H // 2, W // 2
        yy, xx = np.mgrid[0:s.h, 0:s.w].astype(np.float32)
        s.xx, s.yy = xx * 2 + 0.5, yy * 2 + 0.5

    def layer(s, t, cam, inten, rim, halo, radius_k=1.0):
        c = cam @ np.array([s.center[0], s.center[1], 1.])
        zs = math.sqrt(abs(np.linalg.det(cam[:2, :2])))
        R = s.R * zs * radius_k
        dx, dy = s.xx - c[0], s.yy - c[1]
        r = np.sqrt(dx * dx + dy * dy) + 1e-3
        th = np.arctan2(dy, dx)
        d = r - R
        mt = ((th / (2 * np.pi) + 0.5 + t * 0.006) * s.NT).astype(np.float32)
        rho = np.log(np.maximum(r, 1.) / R)
        mr = ((rho * 0.9 - t * 0.035) * s.NR).astype(np.float32)
        st = cv2.remap(s.streak, mt, mr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        fn = cv2.remap(s.fine, mt * 1.0, (mr * 1.0 + t * 25).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        dpos = np.maximum(d, 0)
        streak = np.clip(0.50 + 0.55 * st + 0.08 * fn, 0, None)
        w_cor = R * (0.055 + 0.06 * np.clip(st + 0.3, 0, 2.5))
        outside = sstep(-1.5, 2.5, d)
        corona = np.exp(-dpos / np.maximum(w_cor, 1.)) * streak ** 1.6 * outside
        rimv = np.exp(-(d / (2.2 + 0.006 * R)) ** 2) * (0.75 + 0.35 * fn)
        halov = np.exp(-dpos / (R * 0.45)) * outside
        inner = np.exp(np.minimum(d, 0) / (R * 0.025)) * (d < 0)
        I = inten * (corona * 1.0 + inner * 0.25) + rim * rimv + halo * halov
        disk = sstep(-3, 2, d)  # 0 inside the moon
        out = np.empty((s.h, s.w, 3), np.float32)
        hot = np.clip(I - 0.9, 0, None)
        out[..., 0] = I * 0.62 + hot * 0.5
        out[..., 1] = I * 0.035 + hot * 0.22
        out[..., 2] = I * 0.04 + hot * 0.16
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
        disk = cv2.resize(disk.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        return out, disk


# ---------------------------------------------------------------- ash field
class AshField:
    def __init__(s, n=520, seed=9):
        r = np.random.default_rng(seed)
        s.x0 = r.uniform(-0.1, 1.1, n); s.y0 = r.uniform(0, 1, n)
        s.z = r.uniform(0.25, 1.0, n) ** 1.6 * 1.9 + 0.15
        s.ph = r.uniform(0, 6.28, n); s.fr = r.uniform(0.3, 0.9, n)
        s.br = r.uniform(0.25, 1.0, n); s.ember = r.random(n) < 0.10
        s.n = n

    def draw(s, frame, t, cam_shift, gain, burst=0.):
        z = s.z
        y = (s.y0 * (H + 120) + t * (14 + 30 * z) + cam_shift[1] * 0.2 * z) % (H + 120) - 60
        x = (s.x0 * W + 30 * z * np.sin(t * s.fr + s.ph) + cam_shift[0] * 0.2 * z) % (W + 100) - 50
        if burst > 0:
            dx, dy = x - W / 2, y - H / 2
            rr = np.sqrt(dx * dx + dy * dy) + 1
            x = x + dx / rr * burst * 90 * z; y = y + dy / rr * burst * 90 * z
        layers = {0: np.zeros((H, W), np.float32), 1: np.zeros((H, W), np.float32), 2: np.zeros((H, W), np.float32)}
        emb = np.zeros((H, W), np.float32)
        for i in range(s.n):
            zi = z[i]
            L = 0 if zi < 0.6 else (1 if zi < 1.4 else 2)
            rad = max(1, int(1.0 + zi * 1.8))
            val = s.br[i] * (0.22 + 0.22 * min(zi, 1.2)) * gain
            xi, yi = int(x[i]), int(y[i])
            if s.ember[i]:
                fl = 0.6 + 0.4 * math.sin(t * 7 * s.fr[i] + s.ph[i])
                cv2.circle(emb, (xi, yi), max(1, rad - 1), val * 3.2 * fl, -1)
            else:
                cv2.ellipse(layers[L], (xi, yi), (rad, max(1, int(rad * 0.6))), (s.ph[i] * 57 + t * 40 * s.fr[i]) % 180, 0, 360, val, -1)
        acc = cv2.GaussianBlur(layers[0], (0, 0), 1.2) + layers[1] + cv2.GaussianBlur(layers[2], (0, 0), 3.5) * 1.6
        frame += acc[..., None] * np.array([0.85, 0.80, 0.76], np.float32)
        e = cv2.GaussianBlur(emb, (0, 0), 1.0) + cv2.GaussianBlur(emb, (0, 0), 5) * 0.8
        frame += e[..., None] * np.array([1.0, 0.18, 0.08], np.float32)


# ---------------------------------------------------------------- scene
class Scene: pass

def build():
    sc = Scene()
    RC = (960, 548); R = 336
    sc.ecl = Eclipse(RC, R)
    words = []
    line_y = 452
    specs = [("This", 2.490), ("could", 2.815), ("be", 3.135), ("the", 3.455)]
    size = 84
    ms = [measure(t, FI, size) for t, _ in specs]
    gap = size * 0.30
    xs = []; x = 0.
    for adv, ink in ms: xs.append(x); x += adv + gap
    left = xs[0] + ms[0][1][0]; right = xs[-1] + ms[-1][1][2]
    off = RC[0] - (left + right) / 2
    for i, ((txt, tt), (adv, ink)) in enumerate(zip(specs, ms)):
        cx = off + xs[i] + (ink[0] + ink[2]) / 2
        b0 = 4.42 + 0.16 * i
        words.append(Word2(txt, size, (cx, line_y), tt, seed=100 + i, burn=(b0, b0 + 0.95), n_part=700))
    day = Word2('day', 236, (RC[0] + 6, 628), DAY_T, style='blood', seed=200, burn=(5.88, 6.47), burn_mode='btu',
                drips=3, drip_t=4.45, n_part=2200, ent_blur=16., ent_drop=0.05, ent_scale=1.10, flash=1.0)
    for d in day.drips: d.L *= 1.6
    words.append(day)
    sc.words, sc.day = words, day
    sc.ash = AshField()
    # camera: slow push, slight sink
    cam = pv.Camera((960, 540, 1.0))
    cam.key(0.0, DAY_T, 960, 548, 1.06, lin)
    cam.key(DAY_T, 0.6, 960, 560, 1.10, e_out3)
    cam.key(DAY_T + 0.6, MUSIC_END - DAY_T - 0.6, 962, 600, 1.21, lin)
    cam.key(MUSIC_END, DUR - MUSIC_END, 962, 612, 1.24, lin)
    sc.cam = cam
    return sc


def ecl_params(t):
    """intensity of corona, rim, halo, radius factor"""
    hb = 0.
    for i, k in enumerate(KICKS):
        if k < DAY_T - 0.05:
            hb += (0.25 + 0.12 * i) * dec(t, k, 0.22)
    pre = 0.05 + 0.10 * sstep(0.4, 3.6, t)
    if t < DAY_T - 0.02:
        inten = 0.04 + 0.10 * sstep(0.4, 2.6, t) + 0.12 * hb + 0.08 * sstep(2.4, 3.7, t)
        rim = pre * 1.6 + hb * 0.65
        halo = 0.03 + 0.07 * hb + 0.03 * sstep(0.4, 2.6, t)
        rk = 0.985 + 0.015 * sstep(0, 3.7, t)
        return inten, rim, halo, rk
    u = t - DAY_T
    ign = e_out3(u / 0.35)
    flare = 1.6 * dec(t, DAY_T, 0.22) + 1.0 * dec(t, BIG_T, 0.30)
    v = venv(t)
    inten = 0.55 * ign + 0.55 * v + 0.35 * flare
    for k in KICKS:
        if k > DAY_T + 0.1: inten += 0.22 * dec(t, k, 0.18)
    rim = 0.55 + 0.5 * flare + 0.3 * v
    halo = 0.10 + 0.10 * v + 0.18 * flare
    rk = 1.0 + 0.045 * e_out3(cl((t - 3.74) / 0.36)) + 0.02 * dec(t, BIG_T, 0.3)
    if t > 5.9:
        k = 1 - 0.75 * sstep(5.9, MUSIC_END, t)
        inten *= k; halo *= k; rim *= 0.4 + 0.6 * k
    if t >= MUSIC_END:
        kk = dec(t, MUSIC_END, 0.08) * 0.8 + 0.2 * (1 - sstep(MUSIC_END, DUR - 0.2, t))
        inten *= kk * 0.6; rim = 0.12 * (1 - sstep(MUSIC_END, DUR - 0.1, t)) + 0.4 * dec(t, MUSIC_END, 0.06); halo *= kk
    return inten, rim, halo, rk


class FX2:
    def __init__(s):
        rng = np.random.default_rng(17)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        s.vign = (1 - 0.72 * np.clip(r / 1.2, 0, 1) ** 1.8).astype(np.float32)[..., None]
        n = fbm(H + 80, W + 80, 200, 5, rng)
        base = 0.008 + 0.020 * n ** 1.8
        s.bg = np.stack([base * 1.3, base * 0.75, base * 0.72], -1).astype(np.float32)
        s.grain = []
        for i in range(10):
            g = rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR) + rng.normal(0, 0.6, (H, W)).astype(np.float32)
            s.grain.append(g[..., None])
        s.wx, s.wy, s.nf, s.sx, s.sy = Noise1(21), Noise1(22), Noise1(23), Noise1(24), Noise1(25)


def cam_at(sc, fx, t):
    cx, cy, z = sc.cam.at(t)
    z *= 1 + 0.012 * dec(t, DAY_T, 0.25) + 0.010 * dec(t, BIG_T, 0.2)
    for k in KICKS: z *= 1 + 0.004 * dec(t, k, 0.12)
    sh = 4.0 * dec(t, DAY_T, 0.3) + 3.0 * dec(t, BIG_T, 0.25)
    sx, sy = fx.sx(t, 6) * sh, fx.sy(t, 6) * sh
    wx, wy = fx.wx(t, 1.2) * 1.0, fx.wy(t, 1.0) * 1.4
    return T(W / 2 + sx + wx, H / 2 + sy + wy) @ Sc(z) @ T(-cx, -cy), (cx, cy)


def render(sc, fx, fi):
    t = fi / FPS + VLEAD
    rng = np.random.default_rng(3000 + fi)
    C, (ccx, ccy) = cam_at(sc, fx, t)
    frame = fx.bg[40:40 + H, 40:40 + W].copy()
    if t < 0.5: frame *= 0.3 + 0.7 * sstep(0, 0.5, t)
    # eclipse
    inten, rim, halo, rk = ecl_params(t)
    lay, disk = sc.ecl.layer(t, C, inten, rim, halo, rk)
    frame *= (0.25 + 0.75 * disk)[..., None]
    frame += lay
    # ash behind text
    gain = 0.35 + 0.65 * sstep(0.3, 1.5, t)
    if t > MUSIC_END: gain *= 1 - 0.4 * sstep(MUSIC_END, DUR, t)
    burst = 0.6 * (1 - math.exp(-max(0, t - DAY_T) / 0.4)) * (t > DAY_T) + 0.4 * (1 - math.exp(-max(0, t - BIG_T) / 0.4)) * (t > BIG_T)
    sc.ash.draw(frame, t, (ccx - 960, ccy - 548), gain, burst)
    # words
    plist = []
    for w in sc.words:
        if t < w.t_in - 0.06: continue
        if w.burn is not None and t > w.burn[1] + 0.05:
            pass
        else:
            rg = w.rgba(t)
            M = w.M(t, C)
            a = w.alpha(t)
            # soft red under-glow
            if w.style != 'blood':
                draw_items(frame, rg, [M @ T(w.size * 0.02 * w.rs, w.size * 0.02 * w.rs)], a * 0.55, np.array([0.22, 0.0, 0.01], np.float32))
            draw_items(frame, rg, [M], a)
            fl = w.flash * dec(t, w.t_in, 0.28)
            if fl > 0.01:
                draw_items(frame, rg, [M @ T(rg.shape[1] / 2, rg.shape[0] / 2) @ Sc(1 + 0.06 * (1 - dec(t, w.t_in, 0.28))) @ T(-rg.shape[1] / 2, -rg.shape[0] / 2)],
                           min(1., fl) * 0.6, np.array([1.0, 0.22, 0.15], np.float32) if w.style == 'blood' else np.array([1.0, 0.85, 0.75], np.float32))
        pr = w.particles(t, C)
        if pr is not None: plist.append(pr)
    # burnt flakes
    if plist:
        acc = np.zeros((H, W), np.float32); emb = np.zeros((H, W), np.float32)
        for Q, fade, ember in plist:
            xi = Q[:, 0].astype(np.int32); yi = Q[:, 1].astype(np.int32)
            ok = (xi >= 1) & (xi < W - 1) & (yi >= 1) & (yi < H - 1)
            xi, yi, f_, e_ = xi[ok], yi[ok], fade[ok], ember[ok]
            np.add.at(acc, (yi, xi), f_ * (1 - e_) * 0.55)
            np.add.at(emb, (yi, xi), f_ * e_ * 1.6)
        acc = cv2.GaussianBlur(acc, (0, 0), 0.9) * 2.2
        emb = cv2.GaussianBlur(emb, (0, 0), 0.9) * 2.2 + cv2.GaussianBlur(emb, (0, 0), 4) * 3.0
        frame += acc[..., None] * np.array([0.62, 0.58, 0.55], np.float32) + emb[..., None] * np.array([1.0, 0.20, 0.08], np.float32)

    # bloom
    small = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 0.22, 0, None)
    bl = cv2.GaussianBlur(br, (0, 0), 8) * 0.5 + cv2.GaussianBlur(br, (0, 0), 30) * 0.6
    frame += cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([1.0, 0.45, 0.42], np.float32)
    # global red breath at the big hits
    fl = 0.30 * dec(t, DAY_T, 0.35) + 0.18 * dec(t, BIG_T, 0.3)
    if fl > 0.004:
        frame += np.array([0.20, 0.004, 0.006], np.float32) * fl
    # chromatic softness
    c = 0.0010 + 0.0035 * dec(t, DAY_T, 0.3) + 0.0025 * dec(t, BIG_T, 0.25)
    Rr = cv2.warpAffine(frame[..., 0], (T(W / 2, H / 2) @ Sc(1 + c) @ T(-W / 2, -H / 2))[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    Bb = cv2.warpAffine(frame[..., 2], (T(W / 2, H / 2) @ Sc(1 - c) @ T(-W / 2, -H / 2))[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    frame[..., 0] = Rr; frame[..., 2] = Bb
    # end fade
    frame *= 1 - sstep(DUR - 0.45, DUR - 0.02, t)
    # film
    flick = 1 + 0.025 * fx.nf(t, 7.0) + (rng.random() < 0.03) * (-0.07)
    frame *= flick * fx.vign
    lum = np.clip(frame.mean(-1, keepdims=True), 0, 1)
    frame += fx.grain[int(rng.integers(0, len(fx.grain)))] * (0.014 + 0.06 * lum)
    sr = np.random.default_rng(fi // 4 + 77)
    if sr.random() < 0.3:
        x = int(sr.uniform(100, W - 100) + rng.normal(0, 1.2)); y0 = int(sr.uniform(-200, H * 0.4)); y1 = int(y0 + sr.uniform(H * 0.5, H * 1.2))
        val = sr.uniform(0.03, 0.07) * (1 if sr.random() < 0.6 else -1)
        cv2.line(frame, (x, y0), (x + int(sr.normal(0, 4)), y1), (val, val * 0.95, val * 0.9), 1, cv2.LINE_AA)
    for _ in range(int(rng.poisson(1.0))):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H)); r = int(rng.integers(1, 3))
        val = rng.uniform(0.06, 0.18) * (1 if rng.random() < 0.5 else -0.6)
        cv2.ellipse(frame, (x, y), (r, int(r * rng.uniform(0.6, 2.5))), rng.uniform(0, 180), 0, 360, (val, val * 0.95, val * 0.9), -1, cv2.LINE_AA)
    frame = np.clip(frame, 0, 1) ** 1.05
    frame = frame * 0.99 + np.array([0.006, 0.002, 0.002], np.float32)
    return (np.clip(frame, 0, 1) * 255 + rng.random((H, W, 1)) * 0.9).astype(np.uint8)


SC = FXO = None
def _init():
    global SC, FXO
    if SC is None: SC = build(); FXO = FX2()
def _job(fi): return fi, render(SC, FXO, fi).tobytes()


if __name__ == '__main__':
    import multiprocessing as mp
    mode = sys.argv[1]
    _init()
    if mode == 'still':
        times = [float(x) for x in sys.argv[2].split(',')]
        out = sys.argv[3]
        ims = []
        for tt in times:
            im = render(SC, FXO, int(round(tt * FPS)))
            cv2.putText(im, f'{tt:.3f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            ims.append(im)
        cols = 3 if len(ims) > 2 else len(ims); rows_ = (len(ims) + cols - 1) // cols
        th, tw = H // 2, W // 2
        sheet = np.zeros((rows_ * th, cols * tw, 3), np.uint8)
        for i, im in enumerate(ims):
            r_, c_ = divmod(i, cols)
            sheet[r_ * th:(r_ + 1) * th, c_ * tw:(c_ + 1) * tw] = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        if len(ims) == 1: cv2.imwrite(out.replace('.png', '_full.png'), cv2.cvtColor(ims[0], cv2.COLOR_RGB2BGR))
    else:
        out = sys.argv[2]
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
               '-i', AUDIO, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p',
               '-tune', 'grain', '-c:a', 'aac', '-b:a', '320k', '-shortest', '-movflags', '+faststart', out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with mp.get_context('fork').Pool(os.cpu_count()) as pool:
            for k, (fi, buf) in enumerate(pool.imap(_job, range(NF), chunksize=2)):
                p.stdin.write(buf)
                if k % 60 == 0: print('frame', fi, flush=True)
        p.stdin.close(); p.wait(); print('done', out)
