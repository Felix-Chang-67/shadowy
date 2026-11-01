"""Lyric PV #3: 'This could be the day I die for you. / Don't let it be the day.'
Part A mournful & resolute (soft italic, falling ash, burns away) -> Part B the plea hardens (slammed caps, blood DAY)
-> the held 'day': cornered beast, cracks spread, embers rise -> 12.34 drop: frame splits, claws tear on every hit."""
import numpy as np, cv2, math, os, sys, subprocess
import pv_render as pv
import pv2_render as pv2
from pv_render import (W, H, FPS, T, Sc, Rot, cl, sstep, e_out3, e_out5, e_inout, e_in2, lin, dec, Noise1, fbm, measure,
                blood_shade, draw_items, FN, FE)
from pv2_render import Word2, AshField, FI

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = HERE + '/work3/audio.wav'
DUR = 14.49
NF = int(round(DUR * FPS))
VLEAD = 0.010

A_WORDS = [("This", 0.940), ("could", 1.255), ("be", 1.600), ("the", 1.940), ("day", 2.240),
           ("I", 2.890), ("die", 3.480), ("for", 4.110), ("you", 4.470)]
HIT_A = 4.790
B_WORDS = [("DON'T", 5.690), ("LET", 6.000), ("IT", 6.320), ("BE", 6.640), ("THE", 6.970)]
DAY_T = 7.270
CRACK_T0 = 7.55
DROP = 12.341
HITS = [12.672, 12.980, 13.305, 13.607]
END_T = 13.690
KICKS_A = [4.348, 4.598, 4.790, 5.108, 5.364, 5.596, 5.863, 6.089, 6.322, 6.653]

ENV = np.load(HERE + '/work3/env.npz')
_t = ENV['t']
_vr = cv2.GaussianBlur(ENV['vr'].astype(np.float32).reshape(1, -1), (0, 0), 4).ravel()
_hi = cv2.GaussianBlur(ENV['hi'].astype(np.float32).reshape(1, -1), (0, 0), 6).ravel()
def venv(t): return float(np.interp(t, _t, _vr))
def hienv(t): return float(np.interp(t, _t, _hi))


def tension(t):
    """0..1 build over the held 'day' (riser)"""
    return float(sstep(CRACK_T0, DROP, t) ** 1.6)


# ---------------------------------------------------------------- word with custom entrances (rise / jitter)
class DayWord(pv.Word):
    def draws(s, t):
        u = t - s.t_in
        if u < -0.02: return []
        rise = (1 - e_out5(cl((u + 0.02) / 0.45))) * s.size * 0.32
        a = sstep(-0.02, 0.16, u)
        sc = 0.90 + 0.10 * e_out3(cl((u + 0.02) / 0.5))
        # tremble grows with tension, bursts at the drop and on each hit
        amp = 2.5 * tension(t) + 9 * dec(t, DROP, 0.2) + sum(5 * dec(t, h, 0.12) for h in HITS)
        jx = amp * math.sin(t * 61.0) * math.cos(t * 23.0)
        jy = amp * math.sin(t * 47.0 + 1.3)
        burst = 0.10 * (1 - e_out3(cl((t - DROP) / 0.35))) * (t >= DROP) + sum(0.035 * dec(t, h, 0.1) for h in HITS)
        M = s.base_M(sc + burst, 0., jx, rise + jy)
        out = []
        # red misregistration shadow (retro print) in the fight part
        if t >= DROP:
            off = 0.03 * s.size * (0.5 + dec(t, DROP, 0.3) + sum(dec(t, h, 0.12) for h in HITS))
            out.append(([s.base_M(sc + burst, 0., jx + off, rise + jy + off * 0.6)], 0.7, np.array([0.55, 0.02, 0.02], np.float32), None))
        out.append(([M], a, None, None))
        if u >= 0 and u < 0.4:
            k = e_out3(u / 0.4)
            out.append(([s.base_M(sc * (1 + 0.18 * k), 0., jx, rise + jy)], 0.4 * (1 - k) ** 2, np.array([0.9, 0.1, 0.06], np.float32), None))
        return out


class FightWord(pv.Word):
    """slammed caps that also jolt on the drop/hits"""
    def draws(s, t):
        out = super().draws(t)
        if t < DROP: return out
        k = 0.06 * dec(t, DROP, 0.22) + sum(0.025 * dec(t, h, 0.1) for h in HITS)
        if k < 1e-3: return out
        res = []
        for (Ms, a, tint, mask) in out:
            res.append(([M @ np.linalg.inv(s.base_M(1.)) @ s.base_M(1 + k) @ np.linalg.inv(s.base_M(1.)) @ s.base_M(1.) for M in Ms], a, tint, mask))
        return res


# ---------------------------------------------------------------- cracks
class Cracks:
    def __init__(s, center, box, seed=31):
        rng = np.random.default_rng(seed)
        s.paths = []  # (pts Nx2 world, dist N, ts, width)
        cx, cy = center; bw, bh = box
        n_roots = 11
        starts = np.sort(CRACK_T0 + (rng.random(n_roots) ** 1.4) * 3.6)
        for i in range(n_roots):
            a = 2 * np.pi * i / n_roots + rng.normal(0, 0.12)
            p0 = np.array([cx + 0.5 * bw * math.cos(a) * 0.98, cy + 0.5 * bh * math.sin(a) * 0.98])
            s._grow(p0, a, 0., starts[rng.permutation(n_roots)[i]], 3.2, 0, rng, (cx, cy))
        # cumulative growth table (world px)
        s.tt = np.linspace(0, DUR, 3000)
        v = 40 + 270 * np.array([sstep(9.0, DROP, x) ** 1.5 for x in s.tt])
        v[s.tt >= DROP] = 30.
        v[(s.tt >= DROP) & (s.tt < DROP + 0.12)] = 2600.
        s.cum = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(s.tt))])
        # vertex birth times (for ember spawning)
        s.verts = []
        for (pts, dist, ts, w) in s.paths:
            c0 = np.interp(ts, s.tt, s.cum)
            birth = np.interp(c0 + dist, s.cum, s.tt)
            for p, b in zip(pts[::3], birth[::3]): s.verts.append((p[0], p[1], b))
        s.verts = np.array(s.verts, np.float32)

    def _grow(s, p, a, d0, ts, w, depth, rng, c):
        L = rng.uniform(320, 820) * (0.5 ** depth)
        step = 11.
        pts = [p.copy()]; dist = [d0]
        out_a = a
        for k in range(int(L / step)):
            a += rng.normal(0, 0.38)
            # keep heading outward
            oa = math.atan2(p[1] - c[1], p[0] - c[0])
            da = math.atan2(math.sin(oa - a), math.cos(oa - a))
            a += 0.18 * da
            p = p + step * np.array([math.cos(a), math.sin(a)])
            d0 += step
            pts.append(p.copy()); dist.append(d0)
            if depth < 2 and rng.random() < 0.028:
                s._grow(p.copy(), a + rng.choice([-1, 1]) * rng.uniform(0.45, 1.0), d0, ts, w * 0.6, depth + 1, rng, c)
        s.paths.append((np.array(pts, np.float32), np.array(dist, np.float32), ts, w))

    def length_at(s, t, ts):
        if t <= ts: return 0.
        return float(np.interp(t, s.tt, s.cum) - np.interp(ts, s.tt, s.cum))

    def draw(s, frame, t, C, inten, hot):
        if inten <= 0.002: return
        F = 4
        img = np.zeros((H, W), np.uint8)
        zs = math.sqrt(abs(np.linalg.det(C[:2, :2])))
        for (pts, dist, ts, w) in s.paths:
            Lv = s.length_at(t, ts)
            if Lv <= 0: continue
            m = dist - dist[0] <= Lv
            if m.sum() < 2: continue
            P = pts[m]
            Q = (C[:2, :2] @ P.T).T + C[:2, 2]
            th = max(1, int(round(w * zs * (0.6 + 0.6 * hot))))
            cv2.polylines(img, [(Q * F).astype(np.int32)], False, 255, th, cv2.LINE_AA, 2)
        core = img.astype(np.float32) / 255.
        g1 = cv2.GaussianBlur(core, (0, 0), 4)
        g2 = cv2.GaussianBlur(cv2.resize(core, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 6)
        g2 = cv2.resize(g2, (W, H), interpolation=cv2.INTER_LINEAR)
        col_core = np.array([1.0, 0.30 + 0.45 * hot, 0.22 + 0.35 * hot], np.float32)
        frame += (core[..., None] * col_core * (0.7 + 0.5 * hot) + g1[..., None] * np.array([1.0, 0.06, 0.04], np.float32)
                  + g2[..., None] * np.array([1.5, 0.07, 0.05], np.float32)) * inten


# ---------------------------------------------------------------- rising embers (from cracks + ambient)
class Embers:
    def __init__(s, cracks, n=1400, seed=41):
        r = np.random.default_rng(seed)
        V = cracks.verts
        idx = r.integers(0, len(V), n)
        s.src = V[idx]
        s.delay = r.uniform(0.05, 3.5, n)
        s.period = r.uniform(1.3, 2.4, n)
        s.vy = r.uniform(70, 210, n); s.vx = r.normal(0, 25, n)
        s.ph = r.uniform(0, 6.28, n); s.br = r.uniform(0.4, 1.0, n) ** 2
        # ambient embers for the fight
        m = 700
        s.ax = r.uniform(0, W, m); s.ay = r.uniform(0, H, m); s.avy = r.uniform(120, 520, m)
        s.aph = r.uniform(0, 6.28, m); s.abr = r.uniform(0.3, 1.0, m) ** 2; s.az = r.uniform(0.5, 2.0, m)

    def draw(s, frame, t, C, inten, amb):
        acc = np.zeros((H, W), np.float32)
        if inten > 0.01:
            born = s.src[:, 2] + s.delay
            m = t > born
            if m.any():
                tau = ((t - born[m]) % s.period[m])
                life = s.period[m]
                P = s.src[m, :2]
                Q = (C[:2, :2] @ P.T).T + C[:2, 2]
                x = Q[:, 0] + s.vx[m] * tau + 14 * np.sin(tau * 3 + s.ph[m])
                y = Q[:, 1] - s.vy[m] * tau * (1 + 0.8 * tau)
                f = s.br[m] * np.sin(np.pi * tau / life) * inten
                xi, yi = x.astype(np.int32), y.astype(np.int32)
                ok = (xi > 0) & (xi < W) & (yi > 0) & (yi < H)
                np.add.at(acc, (yi[ok], xi[ok]), f[ok])
        if amb > 0.01:
            y = (s.ay - s.avy * t * (0.6 + 0.4 * amb)) % (H + 40) - 20
            x = (s.ax + 30 * np.sin(t * 2 + s.aph) + 40 * np.sin(t * 0.7 + s.aph * 3)) % W
            xi, yi = x.astype(np.int32), y.astype(np.int32)
            ok = (xi > 0) & (xi < W) & (yi > 0) & (yi < H)
            np.add.at(acc, (yi[ok], xi[ok]), (s.abr * amb * (0.6 + 0.4 * np.sin(t * 9 + s.aph)))[ok])
        if acc.max() <= 0: return
        e = cv2.GaussianBlur(acc, (0, 0), 1.1) * 7 + cv2.GaussianBlur(acc, (0, 0), 4.5) * 18
        frame += e[..., None] * np.array([1.0, 0.24, 0.08], np.float32)


# ---------------------------------------------------------------- claw slashes
class Claw:
    def __init__(s, t0, direction, seed, scale=1.0, cx=W / 2):
        r = np.random.default_rng(seed)
        s.t0 = t0
        cx, cy = cx + r.normal(0, 30), H / 2 + r.normal(0, 40)
        ang = math.radians(r.uniform(52, 66)) * direction
        L = r.uniform(900, 1150) * scale
        d = np.array([math.cos(ang), math.sin(ang)]) * (1 if direction > 0 else -1)
        if d[1] < 0: d = -d
        nrm = np.array([-d[1], d[0]])
        s.streaks = []
        for k in (-1, 0, 1):
            off = nrm * k * r.uniform(68, 84) * scale
            p0 = np.array([cx, cy]) - d * L / 2 + off + d * r.uniform(-40, 40)
            p2 = np.array([cx, cy]) + d * L / 2 * (0.9 if k else 1.0) + off
            p1 = (p0 + p2) / 2 + nrm * r.uniform(30, 70) * direction
            s.streaks.append((p0, p1, p2, r.uniform(10, 15) * scale * (1.15 if k == 0 else 0.9)))

    def _poly(s, p0, p1, p2, w, a, b):
        ss = np.linspace(a, b, 48)
        P = ((1 - ss) ** 2)[:, None] * p0 + (2 * (1 - ss) * ss)[:, None] * p1 + (ss ** 2)[:, None] * p2
        D = np.gradient(P, axis=0); D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-6
        N = np.stack([-D[:, 1], D[:, 0]], 1)
        hw = w * np.sin(np.pi * ss) ** 0.7 + 0.5
        return np.concatenate([P + N * hw[:, None], (P - N * hw[:, None])[::-1]])

    def draw(s, frame, t, scar_fade=1.0):
        u = t - s.t0
        if u < 0: return
        head = e_out3(cl(u / 0.075)); tail = e_in2(cl((u - 0.02) / 0.2)) * 0.0
        F = 4
        img = np.zeros((H, W), np.uint8)
        for (p0, p1, p2, w) in s.streaks:
            if head <= 0.01: continue
            poly = s._poly(p0, p1, p2, w, tail, head)
            cv2.fillPoly(img, [(poly * F).astype(np.int32)], 255, cv2.LINE_AA, 2)
        m = img.astype(np.float32) / 255.
        hotk = math.exp(-max(0, u - 0.06) / 0.12)
        scar = (0.32 + 0.5 * math.exp(-u / 0.4)) * scar_fade
        g = cv2.GaussianBlur(m, (0, 0), 9)
        g2 = cv2.GaussianBlur(cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 7)
        g2 = cv2.resize(g2, (W, H), interpolation=cv2.INTER_LINEAR)
        # carve: darken under the scar, then light it
        frame *= (1 - 0.6 * m[..., None])
        frame += m[..., None] * (np.array([1.0, 0.85, 0.75], np.float32) * 1.4 * hotk + np.array([0.48, 0.02, 0.025], np.float32) * scar)
        frame += (g[..., None] * np.array([1.6, 0.12, 0.08], np.float32) + g2[..., None] * np.array([3.2, 0.15, 0.1], np.float32)) * (0.25 * scar + 1.0 * hotk)


# ---------------------------------------------------------------- scene
class Scene: pass

def row_origins(specs, cx, base_y, gap):
    ms = [measure(sp[0], sp[1], sp[2], sp[3] if len(sp) > 3 else 0.) for sp in specs]
    xs = []; x = 0.
    for adv, ink in ms: xs.append(x); x += adv + gap
    left = xs[0] + ms[0][1][0]; right = xs[-1] + ms[-1][1][2]
    off = cx - (left + right) / 2
    return [((xx + off, base_y), ink) for xx, (adv, ink) in zip(xs, ms)]


def build():
    sc = Scene()
    # ---- part A (soft italic)
    items_a = []
    r1 = row_origins([(w, FI, 94) for w, _ in A_WORDS[:5]], 960, 470, 94 * 0.30)
    r2 = row_origins([("I", FI, 94), ("die", FI, 170), ("for", FI, 94), ("you", FI, 94)], 960, 680, 94 * 0.34)
    for i, ((w, tt), ((ox, oy), ink)) in enumerate(zip(A_WORDS[:5], r1)):
        c = (ox + (ink[0] + ink[2]) / 2, oy + (ink[1] + ink[3]) / 2)
        b0 = HIT_A + 0.02 + 0.05 * i
        items_a.append(Word2(w, 94, c, tt, seed=300 + i, burn=(b0, b0 + 0.5), n_part=650, flash=0.5))
    for i, ((w, tt), ((ox, oy), ink)) in enumerate(zip(A_WORDS[5:], r2)):
        c = (ox + (ink[0] + ink[2]) / 2, oy + (ink[1] + ink[3]) / 2)
        b0 = HIT_A + 0.06 + 0.06 * i
        if w == 'die':
            wd = Word2(w, 170, c, tt, style='blood', seed=320, burn=(b0, b0 + 0.55), n_part=1400, drips=3, drip_t=3.62,
                       ent_blur=12., ent_drop=0.10, ent_scale=1.06, flash=1.0)
            for d in wd.drips: d.dur *= 0.7
        else:
            wd = Word2(w, 94, c, tt, seed=310 + i, burn=(b0, b0 + 0.5), n_part=650, flash=0.5)
        items_a.append(wd)
    sc.items_a = items_a
    sc.ash = AshField(n=420, seed=12)

    # ---- part B (hard caps + blood DAY)
    items_b = []
    rb = row_origins([(w, FN, 100, 0.06) for w, _ in B_WORDS], 960, 368, 100 * 0.36)
    for i, ((w, tt), (o, ink)) in enumerate(zip(B_WORDS, rb)):
        items_b.append(FightWord(w, FN, 100, o, tt, END_T, 'slam', tracking=0.06, seed=400 + i, impact=0.7 + 0.1 * i,
                                 slam_from=1.9, z=10))
    rd = row_origins([("DAY", FE, 360, 0.02)], 960, 0, 0)
    (ox, oy), ink = rd[0]
    day_cy = 668
    base = day_cy - (ink[1] + ink[3]) / 2
    day = DayWord('DAY', FE, 360, (ox, base), DAY_T, END_T, 'stamp', style='blood', tracking=0.02, rs=1.5, drips=6,
                  drip_t=7.6, drip_spread=3.0, drip_len=1.25, seed=450, distress=0.2, z=12, echo=False)
    for d in day.drips: d.dur *= 2.0
    items_b.append(day)
    sc.items_b = items_b
    sc.day_center = (960, day_cy)
    sc.cracks = Cracks((960, day_cy - 10), (820, 330))
    sc.embers = Embers(sc.cracks)
    sc.claws = [Claw(h, d, 500 + i, k, x) for i, (h, d, k, x) in enumerate(zip(HITS, (1, -1, 1, -1), (0.8, 0.8, 0.85, 1.15), (330, 1590, 520, 960)))]
    # split line (screen space) for the drop
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = math.radians(-9)
    n = np.array([-math.sin(a), math.cos(a)])
    jag = np.interp(xx[0], np.linspace(0, W, 40), np.random.default_rng(3).normal(0, 9, 40)).astype(np.float32)
    sc.split_n = n
    sc.split_side = ((xx - W / 2) * n[0] + (yy - H / 2 - 30) * n[1] + jag[None, :]).astype(np.float32)

    # ---- camera
    cam = pv.Camera((960, 540, 1.0))
    cam.key(0.0, HIT_A, 960, 572, 1.07, lin)
    cam.key(HIT_A, 0.85, 960, 585, 1.11, lin)
    cam.key(5.42, 0.26, 960, 470, 1.0, e_inout)
    cam.key(5.69, DAY_T - 5.69, 960, 485, 1.04, lin)
    cam.key(DAY_T, 0.6, 960, 545, 1.06, e_out3)
    cam.key(DAY_T + 0.6, 12.02 - DAY_T - 0.6, 960, 590, 1.20, lin)
    cam.key(12.02, DROP - 12.02, 960, 620, 1.36, e_in2)
    cam.key(DROP, 0, 960, 520, 0.98)
    cam.key(DROP + 0.001, END_T - DROP, 960, 520, 1.05, lin)
    sc.cam = cam
    return sc


def cam_at(sc, fx, t):
    cx, cy, z = sc.cam.at(t)
    z *= 1 + 0.010 * dec(t, 3.48, 0.3) + 0.012 * dec(t, HIT_A, 0.25) + 0.008 * dec(t, DAY_T, 0.3)
    for k in KICKS_A: z *= 1 + 0.004 * dec(t, k, 0.1)
    for h in HITS: z *= 1 + 0.035 * dec(t, h, 0.09)
    z *= 1 + 0.06 * dec(t, DROP, 0.12)
    sh = (2.0 * dec(t, 3.48, 0.25) + 3.0 * dec(t, HIT_A, 0.25) + 5.0 * tension(t) * (t < DROP)
          + 26 * dec(t, DROP, 0.22) + sum(17 * dec(t, h, 0.15) for h in HITS))
    for w, tt in B_WORDS: sh += 5 * dec(t, tt, 0.12)
    sx, sy = fx.sx(t, 9) * sh, fx.sy(t, 9) * sh
    rot = math.radians(fx.wx(t, 7) * (0.9 * dec(t, DROP, 0.25) + sum(0.6 * dec(t, h, 0.15) for h in HITS)))
    wx, wy = fx.wx(t, 1.2) * 1.0, fx.wy(t, 1.0) * 1.4
    return T(W / 2 + sx + wx, H / 2 + sy + wy) @ Rot(rot) @ Sc(z) @ T(-cx, -cy), (cx, cy)


def paper_negative(frame):
    v = np.clip(frame.max(-1, keepdims=True) * 1.9 - 0.06, 0, 1)
    paper = np.array([0.93, 0.87, 0.78], np.float32); ink = np.array([0.30, 0.015, 0.02], np.float32)
    return paper * (1 - v) + ink * v


def render(sc, fx, fi):
    t = fi / FPS + VLEAD
    rng = np.random.default_rng(5000 + fi)
    C, (ccx, ccy) = cam_at(sc, fx, t)
    frame = fx.bg[40:40 + H, 40:40 + W].copy()
    if t < 0.6: frame *= 0.3 + 0.7 * sstep(0, 0.6, t)
    ten = tension(t)
    fight = 1.0 if (DROP <= t < END_T) else 0.0
    # background red light: dim & low in part A, rising through B, bright in the fight
    g = (0.05 + 0.10 * dec(t, 3.48, 0.4) + 0.25 * dec(t, HIT_A, 0.35)) * (t < 5.6)
    g += (0.05 + 0.16 * ten) * (5.6 <= t < DROP)
    for w, tt in B_WORDS: g += 0.18 * dec(t, tt, 0.15)
    g += (0.30 + 0.6 * dec(t, DROP, 0.35) + sum(0.3 * dec(t, h, 0.18) for h in HITS)) * fight
    frame += fx.glow_low * (np.array([0.11, 0.007, 0.008], np.float32) * g)

    # falling ash (part A, fading through B)
    ag = sstep(0.4, 1.6, t) * (1 - sstep(6.6, 8.0, t))
    if ag > 0.01:
        sc.ash.draw(frame, t, (0, ccy - 540), ag * 0.9, 0.)
    # cracks
    if t >= CRACK_T0 and t < END_T:
        inten = (0.16 + 0.5 * ten + 0.25 * venv(t)) * (t < DROP) + fight * (0.62 + 0.9 * dec(t, DROP, 0.3) + sum(0.35 * dec(t, h, 0.15) for h in HITS))
        hot = min(1., 0.15 + 0.85 * ten + fight)
        if t >= DROP: inten *= 1.0
        sc.cracks.draw(frame, t, C, inten, hot)
    # embers rising
    if t >= CRACK_T0 and t < END_T + 0.6:
        ei = (0.25 + 0.9 * ten) * (t < DROP) + fight * 1.1 + (t >= END_T) * 0.8 * (1 - sstep(END_T, END_T + 0.6, t))
        amb = fight * 0.55 + 0.15 * ten * (t < DROP) + (t >= END_T) * 0.5 * (1 - sstep(END_T, END_T + 0.6, t))
        sc.embers.draw(frame, t, C, ei, amb)

    # words, part A
    plist = []
    if t < 8.2:
        for w in sc.items_a:
            if t < w.t_in - 0.06: continue
            if not (w.burn is not None and t > w.burn[1] + 0.05):
                rg = w.rgba(t); M = w.M(t, C); a = w.alpha(t)
                if w.style != 'blood':
                    draw_items(frame, rg, [M @ T(w.size * 0.02 * w.rs, w.size * 0.02 * w.rs)], a * 0.5, np.array([0.22, 0.0, 0.01], np.float32))
                draw_items(frame, rg, [M], a)
                fl = w.flash * dec(t, w.t_in, 0.28)
                if fl > 0.01:
                    draw_items(frame, rg, [M @ T(rg.shape[1] / 2, rg.shape[0] / 2) @ Sc(1 + 0.06 * (1 - dec(t, w.t_in, 0.28))) @ T(-rg.shape[1] / 2, -rg.shape[0] / 2)],
                               min(1., fl) * 0.6, np.array([1.0, 0.22, 0.15], np.float32) if w.style == 'blood' else np.array([1.0, 0.85, 0.75], np.float32))
            pr = w.particles(t, C)
            if pr is not None: plist.append(pr)
    if plist:
        acc = np.zeros((H, W), np.float32); emb = np.zeros((H, W), np.float32)
        for Q, fade, ember in plist:
            xi = Q[:, 0].astype(np.int32); yi = Q[:, 1].astype(np.int32)
            ok = (xi >= 1) & (xi < W - 1) & (yi >= 1) & (yi < H - 1)
            np.add.at(acc, (yi[ok], xi[ok]), (fade * (1 - ember) * 0.55)[ok])
            np.add.at(emb, (yi[ok], xi[ok]), (fade * ember * 1.6)[ok])
        acc = cv2.GaussianBlur(acc, (0, 0), 0.9) * 2.2
        emb = cv2.GaussianBlur(emb, (0, 0), 0.9) * 2.2 + cv2.GaussianBlur(emb, (0, 0), 4) * 3.0
        frame += acc[..., None] * np.array([0.62, 0.58, 0.55], np.float32) + emb[..., None] * np.array([1.0, 0.20, 0.08], np.float32)

    # words, part B
    if 5.5 <= t < END_T:
        sh = np.zeros((H, W, 3), np.float32)
        for it in sc.items_b:
            if not it.visible(t): continue
            rg = it.rgba(t)
            for (Ms, a, tint, mask) in it.draws(t):
                if tint is None: draw_items(sh, rg, [C @ M for M in Ms], a, np.array([1, 1, 1], np.float32))
        shb = cv2.GaussianBlur(cv2.resize(sh[..., 0], (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 4)
        shb = np.clip(cv2.resize(shb, (W, H), interpolation=cv2.INTER_LINEAR) * 1.6, 0, 1)
        frame *= (1 - 0.85 * shb)[..., None]
        for it in sc.items_b:
            if not it.visible(t): continue
            rg = it.rgba(t)
            for (Ms, a, tint, mask) in it.draws(t):
                draw_items(frame, rg, [C @ M for M in Ms], a, tint)

    # claws
    for i, cw in enumerate(sc.claws):
        if cw.t0 <= t < END_T:
            cw.draw(frame, t, 1.0)

    # the frame cracks open on the drop
    if DROP <= t < DROP + 0.75:
        u = t - DROP
        off = 46 * math.exp(-u / 0.10) + 12 * math.exp(-u / 0.45)
        n = sc.split_n
        up = cv2.warpAffine(frame, np.float32([[1, 0, -n[0] * off], [0, 1, -n[1] * off]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        dn = cv2.warpAffine(frame, np.float32([[1, 0, n[0] * off], [0, 1, n[1] * off]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        sd = sc.split_side
        mu = (sd < -off).astype(np.float32)[..., None]; md = (sd > off).astype(np.float32)[..., None]
        gap = np.exp(-np.abs(sd) / (off * 0.55 + 2))[..., None]
        light = gap * (np.array([2.4, 0.9, 0.7], np.float32) * math.exp(-u / 0.12) + np.array([1.4, 0.06, 0.04], np.float32))
        frame = up * mu + dn * md + (1 - mu - md) * 0. + light * (0.6 + 0.4 * math.exp(-u / 0.2))
    # music cut: everything but the scars goes
    if t >= END_T:
        frame = fx.bg[40:40 + H, 40:40 + W] * 0.5
        for cw in sc.claws:
            cw.draw(frame, t, 0.55 * (1 - sstep(END_T, END_T + 0.7, t)))
        if t < END_T + 0.6:
            sc.embers.draw(frame, t, C, 0.8 * (1 - sstep(END_T, END_T + 0.6, t)), 0.5 * (1 - sstep(END_T, END_T + 0.6, t)))

    # bloom
    small = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 0.25, 0, None)
    bl = cv2.GaussianBlur(br, (0, 0), 8) * 0.5 + cv2.GaussianBlur(br, (0, 0), 28) * 0.55
    frame += cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([1.0, 0.5, 0.45], np.float32)
    # chroma
    c = 0.0010 + 0.002 * ten * (t < DROP) + 0.012 * dec(t, DROP, 0.15) + sum(0.007 * dec(t, h, 0.1) for h in HITS)
    for w, tt in B_WORDS: c += 0.002 * dec(t, tt, 0.1)
    Rr = cv2.warpAffine(frame[..., 0], (T(W / 2, H / 2) @ Sc(1 + c) @ T(-W / 2, -H / 2))[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    Bb = cv2.warpAffine(frame[..., 2], (T(W / 2, H / 2) @ Sc(1 - c) @ T(-W / 2, -H / 2))[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    frame[..., 0] = Rr; frame[..., 2] = Bb
    # glitch slices on the drop and hits
    for (t0, nfr, st, gs) in [(DROP, 4, 1.2, 1)] + [(h, 3, 0.8, 2 + i) for i, h in enumerate(HITS)]:
        f0 = int(math.ceil((t0 - VLEAD) * FPS - 1e-6))
        if f0 <= fi < f0 + nfr:
            gr = np.random.default_rng(gs * 100 + fi)
            for _ in range(int(gr.integers(5, 11))):
                y = int(gr.integers(0, H - 10)); hh = int(gr.integers(6, 70))
                band = frame[y:y + hh]
                band[:] = np.roll(band, int(gr.normal(0, 70 * st)), axis=1)
                if gr.random() < 0.4: band *= gr.uniform(1.15, 1.6)
    # paper-negative impact frames
    for t0 in (DROP, HITS[-1]):
        f0 = int(math.ceil((t0 - VLEAD) * FPS - 1e-6))
        if fi == f0: frame = paper_negative(frame)
    # red flashes
    fl = 0.5 * dec(t, DROP, 0.25) + sum(0.25 * dec(t, h, 0.15) for h in HITS) + 0.12 * dec(t, HIT_A, 0.25) + 0.10 * dec(t, 3.48, 0.25)
    if fl > 0.004:
        frame += (fx.glow_low * 0.5 + 0.25) * np.array([0.40, 0.012, 0.016], np.float32) * fl
    # tension: vignette closes in before the drop
    if 11.7 < t < DROP:
        k = sstep(11.7, DROP, t) ** 2
        frame *= (1 - 0.55 * k * (1 - fx.vign_tight))
    # end
    frame *= 1 - sstep(DUR - 0.35, DUR - 0.02, t)
    # film
    flick = 1 + 0.03 * fx.nf(t, 8.0) + (rng.random() < 0.03) * (-0.07) + 0.06 * ten * math.sin(t * 50) * (t < DROP)
    frame *= flick * fx.vign
    lum = np.clip(frame.mean(-1, keepdims=True), 0, 1)
    frame += fx.grain[int(rng.integers(0, len(fx.grain)))] * (0.015 + 0.065 * lum)
    sr = np.random.default_rng(fi // 4 + 99)
    if sr.random() < 0.35:
        x = int(sr.uniform(100, W - 100) + rng.normal(0, 1.2)); y0 = int(sr.uniform(-200, H * 0.4)); y1 = int(y0 + sr.uniform(H * 0.5, H * 1.2))
        val = sr.uniform(0.03, 0.08) * (1 if sr.random() < 0.6 else -1)
        cv2.line(frame, (x, y0), (x + int(sr.normal(0, 4)), y1), (val, val * 0.95, val * 0.9), 1, cv2.LINE_AA)
    for _ in range(int(rng.poisson(1.2))):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H)); r = int(rng.integers(1, 3))
        val = rng.uniform(0.06, 0.18) * (1 if rng.random() < 0.5 else -0.6)
        cv2.ellipse(frame, (x, y), (r, int(r * rng.uniform(0.6, 2.5))), rng.uniform(0, 180), 0, 360, (val, val * 0.95, val * 0.9), -1, cv2.LINE_AA)
    frame = np.clip(frame, 0, 1) ** 1.05
    frame = frame * 0.99 + np.array([0.006, 0.002, 0.002], np.float32)
    return (np.clip(frame, 0, 1) * 255 + rng.random((H, W, 1)) * 0.9).astype(np.uint8)


class FX3(pv2.FX2):
    def __init__(s):
        super().__init__()
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        s.glow_low = np.exp(-(((xx - W / 2) / (W * 0.55)) ** 2 + ((yy - H * 0.62) / (H * 0.55)) ** 2)).astype(np.float32)[..., None]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        s.vign_tight = np.exp(-(r / 0.75) ** 2).astype(np.float32)[..., None]


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
               '-i', AUDIO, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p',
               '-tune', 'grain', '-c:a', 'aac', '-b:a', '320k', '-shortest', '-movflags', '+faststart', out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with mp.get_context('fork').Pool(os.cpu_count()) as pool:
            for k, (fi, buf) in enumerate(pool.imap(_job, range(NF), chunksize=2)):
                p.stdin.write(buf)
                if k % 60 == 0: print('frame', fi, flush=True)
        p.stdin.close(); p.wait(); print('done', out)
