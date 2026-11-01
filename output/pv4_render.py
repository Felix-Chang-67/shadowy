"""Lyric PV #4: 'Sacrificing everything I knew. / This could be the day I die for you.'
Blood sacrifice: SACRIFICING is built syllable by syllable in blood, each syllable spraying; its blood runs down and
stains EVERYTHING. A wall of blood pours over the frame and drains away. The vow returns and bleeds harder; the drum
fill lands as bursts of blood, and at the end every word bleeds out."""
import numpy as np, cv2, math, os, sys, subprocess
import pv_render as pv
import pv2_render as pv2
from pv_render import (W, H, FPS, T, Sc, Rot, cl, sstep, e_out3, e_out5, e_inout, e_in2, lin, dec, Noise1, fbm, measure,
                blood_shade, draw_items, Splat)
from pv3_render import FX3, row_origins

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = HERE + '/work4/audio.wav'
FC = HERE + '/fonts/Cinzel_wght_700.ttf'
FCI = HERE + '/fonts/Cormorant_Garamond_ital_wght_1_500.ttf'
DUR = 12.49
NF = int(round(DUR * FPS))
VLEAD = 0.010

SAC = [("SA", 1.020), ("CRI", 1.350), ("FI", 1.680), ("CING", 2.000)]
EVE = [("EV", 2.320), ("ERY", 2.950), ("THING", 3.570)]
KNEW = [("I", 4.060), ("knew", 4.530)]
L2A = [("This", 6.110), ("could", 6.360), ("be", 6.730), ("the", 7.180), ("day", 7.360)]
L2B = [("I", 7.990), ("DIE", 8.620), ("FOR", 9.260), ("YOU", 9.630)]
CUR0, CUR1, CUR2 = 5.52, 5.98, 6.42      # curtain falls / covers / has drained
BAND_BACK = 9.932
FILL = [9.932, 10.559, 10.879, 11.192, 11.349, 11.505]
FILL_SMALL = [10.089, 10.246, 11.041, 11.668]
MELT_T = 11.668
END_T = 11.80
KICKS1 = [0.459, 1.178, 1.457, 1.741, 1.968, 2.363, 2.560, 3.007, 3.210, 3.628, 4.069, 4.272, 4.528, 4.905, 5.155, 5.439]

ENV = np.load(HERE + '/work4/env.npz')
_t = ENV['t']
_vr = cv2.GaussianBlur(ENV['vr'].astype(np.float32).reshape(1, -1), (0, 0), 4).ravel()
def venv(t): return float(np.interp(t, _t, _vr))
def fill_dec(t, tau, k=1.0, small=0.0):
    return sum(k * dec(t, h, tau) for h in FILL) + sum(small * dec(t, h, tau) for h in FILL_SMALL)


# ---------------------------------------------------------------- italic words that can bleed out
class BleedSoft(pv2.Word2):
    def __init__(s, *a, melt_t=99., melt_dur=0.7, **kw):
        kw.setdefault('pad_bottom', 1.3)
        super().__init__(*a, **kw)
        s.melt_t, s.melt_dur = melt_t, melt_dur
        rng = np.random.default_rng(kw.get('seed', 0) + 999)
        Ww = s.shape[1]
        cols = fbm(1, Ww, max(4., s.px * 0.05), 3, rng)[0]
        sp = np.zeros(Ww, np.float32)
        for _ in range(int(len(s.text) * 2.5)):
            c = rng.uniform(0, Ww); wd = s.px * rng.uniform(0.01, 0.05)
            sp += rng.uniform(0.4, 1.0) * np.exp(-((np.arange(Ww) - c) / wd) ** 2)
        s.melt_cols = (0.25 + 0.5 * cols + np.clip(sp, 0, 1)).astype(np.float32)

    def rgba(s, t):
        out = super().rgba(t)
        if t < s.melt_t: return out
        u = cl((t - s.melt_t) / s.melt_dur)
        Hh, Ww = s.shape
        top, bot = s.ink[1], s.ink[3]
        yy = np.arange(Hh, dtype=np.float32)
        wy = np.clip((yy - top) / (bot - top + 1e-3), 0, 1.6) * 0.75 + 0.25
        dy = (u ** 1.5) * s.px * 0.95 * wy[:, None] * s.melt_cols[None, :]
        mx = np.broadcast_to(np.arange(Ww, dtype=np.float32)[None, :], (Hh, Ww))
        my = (yy[:, None] - dy).astype(np.float32)
        o = cv2.remap(out, np.ascontiguousarray(mx), my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        k = sstep(0.0, 0.5, u) * 0.85
        if s.style != 'blood':
            o = o * np.array([1 - 0.45 * k, 1 - 0.95 * k, 1 - 0.93 * k, 1], np.float32)
        return o * (1 - sstep(0.55, 1.0, u))


# ---------------------------------------------------------------- screen-space blood: curtain, runs, flying droplets
class ScreenBlood:
    def __init__(s, seed=11):
        r = np.random.default_rng(seed)
        s.h, s.w = H // 2, W // 2
        s.yy = np.arange(s.h, dtype=np.float32)[:, None]
        s.n1 = (fbm(1, s.w, 18, 4, r)[0] - 0.5).astype(np.float32)
        sp = np.zeros(s.w, np.float32)
        for _ in range(26):
            c = r.uniform(0, s.w); wd = r.uniform(4.5, 11)
            sp = np.maximum(sp, r.uniform(0.35, 1.0) * np.exp(-((np.arange(s.w) - c) / wd) ** 2))
        s.fingers = (sp ** 1.3).astype(np.float32)
        s.n2 = (fbm(1, s.w, 24, 4, r)[0] - 0.5).astype(np.float32)
        # residual runs left by the curtain + new runs at DIE
        s.runs = []
        for _ in range(16):
            s.runs.append(dict(x=r.uniform(20, s.w - 20), w=r.uniform(2.5, 7.5), t0=CUR2 - r.uniform(0.0, 0.25),
                               v=r.uniform(30, 70), y0=r.uniform(10, 50), Lmax=r.uniform(20, 75)))
        for _ in range(9):
            s.runs.append(dict(x=r.uniform(20, s.w - 20), w=r.uniform(2.0, 6.0), t0=8.62 + r.uniform(0.0, 0.6),
                               v=r.uniform(40, 80), y0=-5, Lmax=r.uniform(30, 85)))
        s.drops = []

    def add_burst(self, t0, cx, cy, n, speed, seed, up=0.6):
        r = np.random.default_rng(seed)
        ang = r.uniform(0, 2 * np.pi, n)
        ang = np.where(r.random(n) < up, -np.pi / 2 + r.normal(0, 0.9, n), ang)
        sp = speed * r.uniform(0.35, 1.0, n) ** 0.7
        self.drops.append(dict(t0=t0, x=cx + r.normal(0, 25, n), y=cy + r.normal(0, 18, n), vx=np.cos(ang) * sp, vy=np.sin(ang) * sp,
                               sz=r.uniform(5.0, 15.0, n) * r.uniform(0.6, 1.0, n), life=r.uniform(0.45, 0.9, n)))

    def mask(s, t):
        m = np.zeros((s.h, s.w), np.float32)
        Hh = s.h
        # curtain
        if CUR0 <= t < CUR2 + 0.05:
            u = cl((t - CUR0) / (CUR1 - CUR0))
            front = (Hh + 200) * e_in2(u) * 1.0 + s.fingers * 140 * (1 - 0.3 * u) + s.n1 * 30 - 120
            v = cl((t - CUR1) / (CUR2 - CUR1))
            top = -40 + (Hh + 260) * (v ** 1.6) - s.fingers * 70 * v - s.n2 * 50 * v
            band = ((s.yy < front[None, :]) & (s.yy > top[None, :])).astype(np.float32)
            m = np.maximum(m, band)
        # top meniscus + runs left on the glass
        if t >= CUR2 - 0.3:
            img = np.zeros((s.h, s.w), np.uint8)
            F = 4
            for rr in s.runs:
                if t < rr['t0']: continue
                tau = t - rr['t0']
                L = rr['y0'] + rr['Lmax'] * (1 - math.exp(-tau * rr['v'] / rr['Lmax']))
                x, w = rr['x'], rr['w']
                pts = np.array([[x - w * 1.6, -2], [x - w * 0.55, 6], [x - w * 0.45, L - w], [x + w * 0.45, L - w], [x + w * 0.55, 6], [x + w * 1.6, -2]])
                cv2.fillPoly(img, [(pts * F).astype(np.int32)], 255, cv2.LINE_AA, 2)
                cv2.circle(img, (int(x * F), int((L - w * 0.6) * F)), int(w * 0.75 * F), 255, -1, cv2.LINE_AA, 2)
            ttop = 10 + 6 * np.sin(np.arange(s.w) * 0.05)
            k = sstep(CUR2 - 0.3, CUR2, t) * (1 - sstep(MELT_T + 0.3, DUR, t) * 0.0)
            m = np.maximum(m, img.astype(np.float32) / 255.)
            m = np.maximum(m, (s.yy < (ttop * k)[None, :]).astype(np.float32))
        # flying droplets
        if s.drops:
            img = np.zeros((s.h, s.w), np.uint8)
            F = 4
            for d in s.drops:
                tau = t - d['t0']
                if tau <= 0 or tau > 1.0: continue
                alive = tau < d['life']
                x = (d['x'] + d['vx'] * tau) * 0.5; y = (d['y'] + d['vy'] * tau + 0.5 * 2600 * tau * tau) * 0.5
                vx = d['vx']; vy = d['vy'] + 2600 * tau
                spd = np.sqrt(vx * vx + vy * vy) + 1
                for i in np.nonzero(alive)[0]:
                    if not (-20 < x[i] < s.w + 20 and -20 < y[i] < s.h + 20): continue
                    sz = d['sz'][i] * 0.5 * (1 - 0.5 * tau / d['life'][i])
                    el = 1 + min(3.0, spd[i] / 900)
                    ang = math.degrees(math.atan2(vy[i], vx[i]))
                    cv2.ellipse(img, (int(x[i] * F), int(y[i] * F)), (max(1, int(sz * el * F)), max(1, int(sz * F))), ang, 0, 360, 255, -1, cv2.LINE_AA, 2)
            m = np.maximum(m, img.astype(np.float32) / 255.)
        return m

    def composite(s, frame, t, dark=0.62):
        m = s.mask(t)
        if m.max() <= 0.01: return frame
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        m = sstep(0.35, 0.65, m).astype(np.float32)
        rg = blood_shade(m, 0.8, wet=1.0, dark=dark)
        rg = cv2.resize(rg, (W, H), interpolation=cv2.INTER_LINEAR)
        return rg[..., :3] + frame * (1 - rg[..., 3:4])


# ---------------------------------------------------------------- scene
class Scene: pass

def build():
    sc = Scene()
    pv2.FI = FCI
    caps, soft, splats = [], [], []
    sb = ScreenBlood()
    # ---- line 1
    SZ1 = 182
    r = row_origins([(p, FC, SZ1, 0.04) for p, _ in SAC], 960, 0, 0)
    y1 = 372
    sac_c = []
    for i, ((p, tt), (o, ink)) in enumerate(zip(SAC, r)):
        base = y1 - (-126 + 2.5) / 2 * SZ1 / 175
        w = pv.Word(p, FC, SZ1, (o[0], base), tt, CUR1, 'slam', ex='cut', style='blood', tracking=0.04, seed=800 + i,
                    impact=1.2, slam_from=1.7, drips=3 if p != 'FI' else 2, drip_t=tt + 0.08, drip_spread=1.4, drip_len=1.6,
                    distress=0.2, z=12, ent_dur=0.075)
        for d in w.drips: d.dur *= 1.5; d.L *= 1.25
        caps.append(w)
        c = (o[0] + (ink[0] + ink[2]) / 2, y1)
        sac_c.append(c)
        splats.append(Splat((c[0] + 10, c[1] + 30), 95 + 15 * (i == 3), tt, CUR1, seed=820 + i, dim=0.36, z=1, ex='cut', grow=0.06))
        sb.add_burst(tt, c[0], c[1], 26, 1300, 850 + i)
    SZ2 = 140
    r = row_origins([(p, FC, SZ2, 0.08) for p, _ in EVE], 960, 0, 0)
    y2 = 600
    for i, ((p, tt), (o, ink)) in enumerate(zip(EVE, r)):
        base = y2 - (-100.75 + 2.75) / 2
        w = pv.Word(p, FC, SZ2, (o[0], base), tt, CUR1, 'slam', ex='cut', tracking=0.08, seed=860 + i, impact=0.9,
                    slam_from=1.5, drips=2, drip_t=tt + 0.3, drip_spread=1.2, drip_len=1.0, distress=0.4, z=11)
        caps.append(w)
    r = row_origins([(p, FCI, 96) for p, _ in KNEW], 960, 0, 96 * 0.30)
    y3 = 770
    for i, ((p, tt), (o, ink)) in enumerate(zip(KNEW, r)):
        c = (o[0] + (ink[0] + ink[2]) / 2, y3 + (ink[1] + ink[3]) / 2 - (-70 + 1.25) / 2)
        soft.append(BleedSoft(p, 96, c, tt, seed=880 + i, flash=0.3, ent_blur=9., ent_drop=0.12, melt_t=5.12 + 0.1 * i, melt_dur=0.7))
    # ---- line 2 (the vow, bleeding)
    r = row_origins([(p, FCI, 104) for p, _ in L2A], 960, 0, 104 * 0.28)
    ya = 372
    for i, ((p, tt), (o, ink)) in enumerate(zip(L2A, r)):
        c = (o[0] + (ink[0] + ink[2]) / 2, ya + (ink[1] + ink[3]) / 2 - (-76 + 29.25) / 2)
        soft.append(BleedSoft(p, 104, c, tt, seed=900 + i, flash=0.35, ent_blur=8., ent_drop=0.12, melt_t=MELT_T + 0.04 * i, melt_dur=0.72))
    SZB = 166
    r = row_origins([(p, FC, SZB, 0.06) for p, _ in L2B], 960, 0, SZB * 0.36)
    yb = 588
    die_c = None
    for i, ((p, tt), (o, ink)) in enumerate(zip(L2B, r)):
        base = yb - (ink[1] + ink[3]) / 2
        if p == 'DIE':
            w = pv.Word(p, FC, SZB, (o[0], base), tt, MELT_T, 'slam', ex='melt', ex_dur=0.72, style='blood', tracking=0.06,
                        seed=920 + i, impact=1.4, slam_from=1.6, drips=6, drip_t=tt + 0.06, drip_spread=1.6, drip_len=1.7,
                        distress=0.2, z=12)
            for d in w.drips: d.dur *= 1.6
            die_c = (o[0] + (ink[0] + ink[2]) / 2, yb)
        else:
            w = pv.Word(p, FC, SZB, (o[0], base), tt, MELT_T, 'slam', ex='melt', ex_dur=0.72, tracking=0.06, seed=920 + i,
                        impact=0.8, slam_from=1.5, drips=(2 if p in ('FOR', 'YOU') else 1), drip_t=BAND_BACK + 0.1 * i,
                        drip_spread=1.0, drip_len=0.9, distress=0.4, z=11)
        caps.append(w)
    splats.append(Splat((die_c[0] - 20, die_c[1] + 70), 150, 8.62, MELT_T + 0.05, seed=940, dim=0.32, z=1, ex='fade', ex_dur=0.6))
    sb.add_burst(8.62, die_c[0], die_c[1], 34, 1400, 941)
    # drum fill: bursts of blood around the vow
    pos = [(470, 640), (1450, 520), (760, 420), (1180, 690), (560, 480), (1330, 610)]
    for i, (h, p) in enumerate(zip(FILL, pos)):
        R = 120 + 30 * (h in (10.559, 11.349))
        splats.append(Splat(p, R, h, MELT_T + 0.05, seed=960 + i, dim=0.40 + 0.04 * (i % 2), z=1, ex='fade', ex_dur=0.6, grow=0.05))
        sb.add_burst(h, p[0], p[1], 30 + 10 * (h in (10.559, 11.349)), 1600, 970 + i)
    sc.caps = sorted(caps + splats, key=lambda it: it.z)
    sc.soft = soft
    sc.sb = sb
    sc.ash = pv2.AshField(n=300, seed=33)
    # camera: follow the syllables of SACRIFICING, open to the card; second card opens like the last clip
    cam = pv.Camera((sac_c[0][0], y1, 1.6))
    cam.key(0.0, 0.9, sac_c[0][0] + 10, y1, 1.62, lin)
    cam.key(SAC[0][1] - 0.05, 0.12, sac_c[0][0] + 60, y1 + 10, 1.55)
    cam.key(SAC[1][1] - 0.04, 0.12, (sac_c[0][0] + sac_c[1][0]) / 2 + 40, y1 + 10, 1.42)
    cam.key(SAC[2][1] - 0.04, 0.12, (sac_c[0][0] + sac_c[2][0]) / 2 + 30, y1 + 15, 1.30)
    cam.key(SAC[3][1] - 0.04, 0.14, 960, y1 + 25, 1.16)
    cam.key(EVE[0][1] - 0.04, 0.18, 960, 470, 1.08)
    cam.key(2.6, 1.4, 960, 488, 1.10, lin)
    cam.key(KNEW[0][1] - 0.04, 0.25, 960, 560, 1.0)
    cam.key(4.4, 1.6, 960, 572, 1.05, lin)
    cam.key(CUR1, 0, 960, ya, 1.26)
    cam.key(CUR1 + 0.01, L2B[0][1] - CUR1 - 0.25, 960, ya + 5, 1.32, lin)
    cam.key(L2B[0][1] - 0.04, 0.26, 960, 480, 1.04, e_out3)
    cam.key(8.3, BAND_BACK - 8.3, 960, 486, 1.10, lin)
    cam.key(BAND_BACK, 0.2, 960, 484, 1.0, e_out3)
    cam.key(BAND_BACK + 0.2, MELT_T - BAND_BACK - 0.2, 960, 490, 1.05, lin)
    cam.key(MELT_T, DUR - MELT_T, 960, 530, 1.12, lin)
    sc.cam = cam
    # bursts were placed in world space: convert to screen at their moment
    for d in sb.drops:
        cx, cy, z = cam.at(d['t0'])
        d['x'] = (d['x'] - cx) * z + W / 2; d['y'] = (d['y'] - cy) * z + H / 2
        d['vx'] = d['vx'] * z; d['vy'] = d['vy'] * z
    return sc


def cam_at(sc, fx, t):
    cx, cy, z = sc.cam.at(t)
    for k in KICKS1: z *= 1 + 0.005 * dec(t, k, 0.09)
    for _, tt in SAC: z *= 1 + 0.020 * dec(t, tt, 0.12)
    z *= 1 + 0.025 * dec(t, 8.62, 0.18)
    z *= 1 + fill_dec(t, 0.12, 0.028, 0.010)
    sh = sum(9 * dec(t, tt, 0.16) for _, tt in SAC) + sum(4 * dec(t, tt, 0.14) for _, tt in EVE)
    sh += 8 * dec(t, 8.62, 0.22) + fill_dec(t, 0.16, 11, 4) + 5 * dec(t, CUR0 + 0.25, 0.4)
    for k in KICKS1: sh += 1.5 * dec(t, k, 0.08)
    sx, sy = fx.sx(t, 8) * sh, fx.sy(t, 8) * sh
    rot = math.radians(fx.wx(t, 6) * (fill_dec(t, 0.15, 0.35) + 0.3 * dec(t, 8.62, 0.2)))
    wx, wy = fx.wx(t, 1.2) * 1.0, fx.wy(t, 1.0) * 1.4
    return T(W / 2 + sx + wx, H / 2 + sy + wy) @ Rot(rot) @ Sc(z) @ T(-cx, -cy), (cx, cy)


def render(sc, fx, fi):
    t = fi / FPS + VLEAD
    rng = np.random.default_rng(9000 + fi)
    C, (ccx, ccy) = cam_at(sc, fx, t)
    frame = fx.bg[40:40 + H, 40:40 + W].copy()
    if t < 0.45: frame *= 0.35 + 0.65 * sstep(0, 0.45, t)
    # smooth red light
    g = 0.06 + 0.10 * sstep(0.4, 1.2, t) * (t < CUR1)
    g += sum(0.12 * dec(t, tt, 0.3) for _, tt in SAC)
    g += 0.05 * (CUR2 <= t) + 0.18 * dec(t, 8.62, 0.5)
    g += (0.14 * (t >= BAND_BACK) + fill_dec(t, 0.3, 0.14, 0.05)) * (t < END_T)
    if t > END_T: g *= 1 - sstep(END_T, DUR, t)
    frame += fx.glow_low * (np.array([0.11, 0.007, 0.008], np.float32) * g)
    # ash only in the quiet vow section
    ag = sstep(CUR2, CUR2 + 0.6, t) * (1 - sstep(BAND_BACK, BAND_BACK + 0.8, t))
    if ag > 0.01: sc.ash.draw(frame, t, (0, ccy - 480), ag * 0.8, 0.)

    # dark halo for legibility
    sh = np.zeros((H, W, 1), np.float32)
    for w in sc.soft:
        if t < w.t_in - 0.06 or (t > CUR1 and w.t_in < CUR0) or t > w.melt_t + w.melt_dur: continue
        draw_items(sh, w.rgba(t), [w.M(t, C)], w.alpha(t), np.array([1.], np.float32))
    for it in sc.caps:
        if isinstance(it, Splat) or not it.visible(t): continue
        rg = it.rgba(t)
        for (Ms, a, tint, mask) in it.draws(t):
            if tint is None: draw_items(sh, rg, [C @ M for M in Ms], a, np.array([1.], np.float32))
    shb = cv2.GaussianBlur(cv2.resize(sh[..., 0], (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3.5)
    shb = np.clip(cv2.resize(shb, (W, H), interpolation=cv2.INTER_LINEAR) * 1.6, 0, 1)

    # splats (behind), then halo, then caps
    for it in sc.caps:
        if not isinstance(it, Splat) or not it.visible(t): continue
        for (Ms, a, tint, mask) in it.draws(t):
            draw_items(frame, it.rgba(t), [C @ M for M in Ms], a, tint)
    frame *= (1 - 0.8 * shb)[..., None]
    for it in sc.caps:
        if isinstance(it, Splat) or not it.visible(t): continue
        rg = it.rgba(t)
        for (Ms, a, tint, mask) in it.draws(t):
            draw_items(frame, rg, [C @ M for M in Ms], a, tint)
    for w in sc.soft:
        if t < w.t_in - 0.06 or (t > CUR1 and w.t_in < CUR0) or t > w.melt_t + w.melt_dur: continue
        rg = w.rgba(t); M = w.M(t, C); a = w.alpha(t)
        draw_items(frame, rg, [M @ T(w.size * 0.02 * w.rs, w.size * 0.02 * w.rs)], a * 0.45, np.array([0.25, 0.0, 0.01], np.float32))
        draw_items(frame, rg, [M], a)
        fl = w.flash * dec(t, w.t_in, 0.3)
        if fl > 0.01:
            draw_items(frame, rg, [M @ T(rg.shape[1] / 2, rg.shape[0] / 2) @ Sc(1 + 0.05 * (1 - dec(t, w.t_in, 0.3))) @ T(-rg.shape[1] / 2, -rg.shape[0] / 2)],
                       min(1., fl) * 0.45, np.array([1.0, 0.86, 0.76], np.float32))

    # blood on the glass: curtain, runs, flying droplets
    frame = sc.sb.composite(frame, t, dark=0.62 if t < CUR2 else 0.7)
    # under the full curtain, darken toward deep blood
    if CUR1 - 0.15 < t < CUR2:
        k = sstep(CUR1 - 0.15, CUR1 + 0.05, t) * (1 - sstep(CUR1 + 0.15, CUR2, t))
        frame *= 1 - 0.45 * k

    # bloom
    small = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 0.28, 0, None)
    bl = cv2.GaussianBlur(br, (0, 0), 8) * 0.45 + cv2.GaussianBlur(br, (0, 0), 28) * 0.45
    frame += cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([1.0, 0.5, 0.45], np.float32)
    c = 0.0009 + sum(0.0025 * dec(t, tt, 0.12) for _, tt in SAC) + fill_dec(t, 0.12, 0.0022)
    Rr = cv2.warpAffine(frame[..., 0], (T(W / 2, H / 2) @ Sc(1 + c) @ T(-W / 2, -H / 2))[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    Bb = cv2.warpAffine(frame[..., 2], (T(W / 2, H / 2) @ Sc(1 - c) @ T(-W / 2, -H / 2))[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    frame[..., 0] = Rr; frame[..., 2] = Bb
    frame *= 1 - sstep(DUR - 0.5, DUR - 0.02, t)
    # film
    flick = 1 + 0.02 * fx.nf(t, 7.0)
    frame *= flick * fx.vign
    lum = np.clip(frame.mean(-1, keepdims=True), 0, 1)
    frame += fx.grain[int(rng.integers(0, len(fx.grain)))] * (0.015 + 0.06 * lum)
    sr = np.random.default_rng(fi // 4 + 299)
    if sr.random() < 0.3:
        x = int(sr.uniform(100, W - 100) + rng.normal(0, 1.2)); y0 = int(sr.uniform(-200, H * 0.4)); y1 = int(y0 + sr.uniform(H * 0.5, H * 1.2))
        val = sr.uniform(0.03, 0.06) * (1 if sr.random() < 0.6 else -1)
        cv2.line(frame, (x, y0), (x + int(sr.normal(0, 4)), y1), (val, val * 0.95, val * 0.9), 1, cv2.LINE_AA)
    for _ in range(int(rng.poisson(1.0))):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H)); r = int(rng.integers(1, 3))
        val = rng.uniform(0.06, 0.15) * (1 if rng.random() < 0.5 else -0.6)
        cv2.ellipse(frame, (x, y), (r, int(r * rng.uniform(0.6, 2.5))), rng.uniform(0, 180), 0, 360, (val, val * 0.95, val * 0.9), -1, cv2.LINE_AA)
    frame = np.clip(frame, 0, 1) ** 1.05
    frame = frame * 0.99 + np.array([0.006, 0.002, 0.002], np.float32)
    return (np.clip(frame, 0, 1) * 255 + rng.random((H, W, 1)) * 0.9).astype(np.uint8)


SC = FXO = None
def _init():
    global SC, FXO
    if SC is None: SC = build(); FXO = FX3()
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
               '-i', AUDIO, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-tune', 'grain', '-c:a', 'aac', '-b:a', '320k', '-shortest', '-movflags', '+faststart', out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with mp.get_context('fork').Pool(os.cpu_count()) as pool:
            for k, (fi, buf) in enumerate(pool.imap(_job, range(NF), chunksize=2)):
                p.stdin.write(buf)
                if k % 60 == 0: print('frame', fi, flush=True)
        p.stdin.close(); p.wait(); print('done', out)
