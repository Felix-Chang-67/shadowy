"""Lyric PV #3 (rev): 'This could be the day I die for you. / Don't let it be the day.'
The vow is the centrepiece: whispered italic 'This could be the day' over engraved caps 'I DIE FOR YOU'.
'Don't let it be the day' is a smaller plea beneath. No lightning, no strobes: the struggle is carried by
straining type, a rising heat glow on DIE, embers that climb faster, and camera weight."""
import numpy as np, cv2, math, os, sys, subprocess
import pv_render as pv
import pv2_render as pv2
import pv3_render as pv3
from pv_render import (W, H, FPS, T, Sc, Rot, cl, sstep, e_out3, e_out5, e_inout, e_in2, lin, dec, Noise1, measure, draw_items)
from pv3_render import (FX3, row_origins, A_WORDS, B_WORDS, DAY_T, DROP, HITS, END_T, HIT_A, KICKS_A, tension, venv)

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = HERE + '/work3/audio.wav'
FC = HERE + '/fonts/Cinzel_wght_700.ttf'
FCI = HERE + '/fonts/Cormorant_Garamond_ital_wght_1_500.ttf'
DUR = 14.49
NF = int(round(DUR * FPS))
VLEAD = 0.010
DIE_T = 3.480


def hits_dec(t, tau, k=1.0):
    return sum(k * dec(t, h, tau) for h in HITS)


def strain(t):
    """jitter amplitude (world px) of the vow while the beast is cornered"""
    return 3.2 * tension(t) * (t < DROP) + 7 * dec(t, DROP, 0.22) + hits_dec(t, 0.14, 3.5)


# ---------------------------------------------------------------- words
class VowWord(pv.Word):
    """engraved caps: slam in, then strain against each other through the held note, snap back on the drop"""
    def draws(s, t):
        out = super().draws(t)
        if not out: return out
        px = s.origin[0] + s.pivot[0]; py = s.origin[1] + s.pivot[1]
        side = (px - 960) / 700.
        spread = 0.045 * tension(t) * (t < DROP) - 0.015 * dec(t, DROP, 0.12) * (t >= DROP)
        dx = (px - 960) * spread
        a = strain(t)
        ph = s.rng_seed * 1.7
        jx = a * math.sin(t * 57 + ph) * math.cos(t * 21 + ph)
        jy = a * math.sin(t * 43 + ph * 2)
        k = 1 + 0.07 * dec(t, DROP, 0.25) * (t >= DROP) + hits_dec(t, 0.12, 0.03) + 0.02 * dec(t, HIT_A, 0.2) * (s.text == 'DIE')
        A = T(px + dx + jx, py + jy) @ Sc(k) @ T(-px, -py)
        return [([A @ M for M in Ms], al, tint, mask) for (Ms, al, tint, mask) in out]


class SoftWord(pv2.Word2):
    def __init__(s, *a, jitter=0.6, **kw):
        super().__init__(*a, **kw); s.jit = jitter; s.ph = kw.get('seed', 0) * 0.37
    def M(s, t, cam):
        base = super().M(t, cam)
        a = strain(t) * s.jit
        jx = a * math.sin(t * 51 + s.ph); jy = a * math.cos(t * 39 + s.ph)
        return cam @ T(jx, jy) @ np.linalg.inv(cam) @ base


# ---------------------------------------------------------------- rising embers
class RisingEmbers:
    def __init__(s, n=1100, seed=77):
        r = np.random.default_rng(seed)
        s.n = n
        s.x0 = r.uniform(-40, W + 40, n); s.ph = r.uniform(0, 1, n)
        s.vy = r.uniform(0.55, 1.45, n); s.sw = r.uniform(8, 40, n); s.sf = r.uniform(0.6, 2.2, n); s.sp = r.uniform(0, 6.28, n)
        s.th = r.random(n); s.br = r.uniform(0.35, 1.0, n) ** 2; s.fl = r.uniform(3, 11, n)
        s.big = r.random(n) < 0.2
        # travel = integral of speed (px) so the speed can change smoothly
        s.tt = np.linspace(0, DUR, 4000)
        v = np.array([s.speed(x) for x in s.tt])
        s.trav = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(s.tt))])

    @staticmethod
    def speed(t):
        base = 90 + 120 * tension(t) * (t < DROP)
        if t >= DROP: base = 230
        return base + 520 * dec(t, DROP, 0.35) + hits_dec(t, 0.18, 160)

    @staticmethod
    def density(t):
        d = 0.10 * sstep(5.4, 6.4, t) + 0.55 * tension(t) * (t < DROP)
        if t >= DROP: d = 0.85 + 0.15 * dec(t, DROP, 0.5)
        if t >= END_T: d *= 1 - sstep(END_T, END_T + 0.7, t)
        return d

    def draw(s, frame, t):
        dens = s.density(t)
        if dens <= 0.005: return
        tr = float(np.interp(t, s.tt, s.trav))
        span = H + 120
        y = H + 60 - ((tr * s.vy + s.ph * span) % span)
        x = s.x0 + s.sw * np.sin(t * s.sf + s.sp) + 0.04 * (H + 60 - y) * np.sin(s.sp)
        m = s.th < dens
        hfade = np.clip((y + 40) / (H * 0.55), 0, 1)
        f = s.br * hfade * (0.65 + 0.35 * np.sin(t * s.fl + s.sp)) * m
        acc = np.zeros((H, W), np.float32)
        xi, yi = x.astype(np.int32), y.astype(np.int32)
        ok = (xi > 1) & (xi < W - 1) & (yi > 1) & (yi < H - 1) & (f > 0)
        np.add.at(acc, (yi[ok], xi[ok]), f[ok])
        big = ok & s.big
        np.add.at(acc, (yi[big] + 1, xi[big]), f[big] * 0.8); np.add.at(acc, (yi[big], xi[big] + 1), f[big] * 0.8)
        e = cv2.GaussianBlur(acc, (0, 0), 1.0) * 9 + cv2.GaussianBlur(acc, (0, 0), 5) * 22
        frame += e[..., None] * np.array([1.0, 0.26, 0.08], np.float32)


# ---------------------------------------------------------------- scene
class Scene: pass

def build():
    sc = Scene()
    pv2.FI = FCI  # soft words use Cormorant italic
    # row 1: whispered italic
    r1 = row_origins([(w, FCI, 110) for w, _ in A_WORDS[:5]], 960, 0, 110 * 0.28)
    soft = []
    y1 = 342
    for i, ((w, tt), ((ox, oy), ink)) in enumerate(zip(A_WORDS[:5], r1)):
        c = (ox + (ink[0] + ink[2]) / 2, y1 + (ink[1] + ink[3]) / 2 - (-80 + 31) / 2)
        soft.append(SoftWord(w, 110, c, tt, seed=600 + i, flash=0.35, ent_blur=8., ent_drop=0.12, jitter=0.5))
    # row 2: the vow, engraved caps
    vow_specs = [(w.upper(), FC, 170, 0.06) for w, _ in A_WORDS[5:]]
    r2 = row_origins(vow_specs, 960, 0, 170 * 0.36)
    y2 = 566
    vow = []
    for i, ((w, tt), (o, ink)) in enumerate(zip(A_WORDS[5:], r2)):
        base = y2 - (ink[1] + ink[3]) / 2
        W_ = w.upper()
        if W_ == 'DIE':
            wd = VowWord(W_, FC, 170, (o[0], base), tt, END_T, 'slam', style='blood', tracking=0.06, seed=700 + i, impact=1.3,
                         slam_from=1.55, drips=4, drip_t=3.6, drip_spread=2.0, drip_len=1.0, distress=0.25, z=12, ent_dur=0.085)
            for d in wd.drips: d.dur *= 2.2
        else:
            wd = VowWord(W_, FC, 170, (o[0], base), tt, END_T, 'slam', tracking=0.06, seed=700 + i, impact=0.8,
                         slam_from=1.5, distress=0.45, z=11, ent_dur=0.08)
        vow.append(wd)
    sc.die = vow[1]
    # row 3: the plea, smaller italic; 'day' in blood
    y3 = 800
    words3 = [("Don't", B_WORDS[0][1]), ("let", B_WORDS[1][1]), ("it", B_WORDS[2][1]), ("be", B_WORDS[3][1]), ("the", B_WORDS[4][1]), ("day", DAY_T)]
    r3 = row_origins([(w, FCI, 78) for w, _ in words3], 960, 0, 78 * 0.30)
    plea = []
    for i, ((w, tt), ((ox, oy), ink)) in enumerate(zip(words3, r3)):
        c = (ox + (ink[0] + ink[2]) / 2, y3 + (ink[1] + ink[3]) / 2 - (-55 + 21) / 2)
        if w == 'day':
            wd = SoftWord(w, 78, c, tt, style='blood', seed=650, flash=0.6, ent_blur=10., ent_drop=-0.25, ent_scale=1.12,
                          drips=2, drip_t=7.9, jitter=0.8)
            for d in wd.drips: d.dur *= 2.5; d.L *= 1.4
        else:
            wd = SoftWord(w, 78, c, tt, seed=640 + i, flash=0.3, ent_blur=7., ent_drop=0.10, jitter=0.7)
        plea.append(wd)
    sc.soft = soft + plea
    sc.vow = vow
    sc.ash = pv2.AshField(n=420, seed=12)
    sc.embers = RisingEmbers()
    # camera: start on the whisper, open to the vow, then to the whole card
    cam = pv.Camera((960, y1, 1.34))
    cam.key(0.0, 2.8, 960, y1 + 6, 1.40, lin)
    cam.key(2.80, 0.24, 960, 452, 1.06, e_out3)
    cam.key(3.05, HIT_A - 3.05, 960, 458, 1.10, lin)
    cam.key(5.42, 0.32, 960, 566, 0.97, e_inout)
    cam.key(5.75, DAY_T - 5.75, 960, 570, 1.00, lin)
    cam.key(DAY_T, 12.02 - DAY_T, 960, 590, 1.12, lin)
    cam.key(12.02, DROP - 12.02, 960, 580, 1.20, e_in2)
    cam.key(DROP, 0.12, 960, 566, 0.99, e_out3)
    cam.key(DROP + 0.12, END_T - DROP - 0.12, 960, 566, 1.05, lin)
    sc.cam = cam
    return sc


def heat(t):
    """emissive glow of DIE"""
    if t < DIE_T: return 0.
    h = 0.22 + 0.45 * dec(t, DIE_T, 0.35) + 0.35 * dec(t, HIT_A, 0.4)
    h += (0.10 + 0.55 * tension(t)) * (DAY_T <= t < DROP) + 0.25 * venv(t) * (DAY_T <= t < DROP)
    if t >= DROP: h = 0.60 + 0.55 * dec(t, DROP, 0.45) + hits_dec(t, 0.25, 0.22)
    if t >= END_T: h = 0.65 * dec(t, END_T, 0.25)
    return h


def cam_at(sc, fx, t):
    cx, cy, z = sc.cam.at(t)
    z *= 1 + 0.010 * dec(t, DIE_T, 0.3) + 0.014 * dec(t, HIT_A, 0.3)
    for k in KICKS_A: z *= 1 + 0.003 * dec(t, k, 0.1)
    z *= 1 + 0.035 * dec(t, DROP, 0.18) * (t >= DROP) + hits_dec(t, 0.12, 0.018)
    for _, tt in A_WORDS[5:]: z *= 1 + 0.008 * dec(t, tt, 0.12)
    sh = (2.5 * dec(t, DIE_T, 0.25) + 4.0 * dec(t, HIT_A, 0.3) + 3.0 * tension(t) * (t < DROP)
          + 20 * dec(t, DROP, 0.30) + hits_dec(t, 0.18, 11))
    for _, tt in A_WORDS[5:]: sh += 3 * dec(t, tt, 0.15)
    sx, sy = fx.sx(t, 7) * sh, fx.sy(t, 7) * sh
    rot = math.radians(fx.wx(t, 5) * (0.7 * dec(t, DROP, 0.3) + hits_dec(t, 0.18, 0.4)))
    wx, wy = fx.wx(t, 1.2) * 1.0, fx.wy(t, 1.0) * 1.4
    return T(W / 2 + sx + wx, H / 2 + sy + wy) @ Rot(rot) @ Sc(z) @ T(-cx, -cy), (cx, cy)


def render(sc, fx, fi):
    t = fi / FPS + VLEAD
    rng = np.random.default_rng(7000 + fi)
    C, (ccx, ccy) = cam_at(sc, fx, t)
    frame = fx.bg[40:40 + H, 40:40 + W].copy()
    if t < 0.6: frame *= 0.3 + 0.7 * sstep(0, 0.6, t)
    ten = tension(t)
    on = t < END_T
    # smooth red light (never strobing)
    g = 0.05 + 0.10 * dec(t, DIE_T, 0.5) + 0.18 * dec(t, HIT_A, 0.5)
    g += (0.06 + 0.22 * ten) * (5.6 <= t < DROP)
    if t >= DROP: g = 0.36 + 0.30 * dec(t, DROP, 0.5) + hits_dec(t, 0.3, 0.10)
    if not on: g = 0.36 * dec(t, END_T, 0.2)
    frame += fx.glow_low * (np.array([0.11, 0.007, 0.008], np.float32) * g)
    # falling ash (sorrow) fades out as the struggle begins
    ag = sstep(0.3, 1.4, t) * (1 - sstep(6.4, 8.0, t))
    if ag > 0.01: sc.ash.draw(frame, t, (0, ccy - 450), ag * 0.9, 0.)
    # rising embers (the fight)
    sc.embers.draw(frame, t)

    if on:
        # dark halo behind all type
        sh = np.zeros((H, W), np.float32)
        lay = []
        for w in sc.soft:
            if t < w.t_in - 0.06: continue
            rg = w.rgba(t); M = w.M(t, C); a = w.alpha(t)
            lay.append(('s', w, rg, M, a))
            draw_items(sh[..., None], rg, [M], a, np.array([1.], np.float32))
        for it in sc.vow:
            if not it.visible(t): continue
            rg = it.rgba(t); ds = it.draws(t)
            lay.append(('v', it, rg, ds, None))
            for (Ms, a, tint, mask) in ds:
                if tint is None: draw_items(sh[..., None], rg, [C @ M for M in Ms], a, np.array([1.], np.float32))
        shb = cv2.GaussianBlur(cv2.resize(sh, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3.5)
        shb = np.clip(cv2.resize(shb, (W, H), interpolation=cv2.INTER_LINEAR) * 1.7, 0, 1)
        frame *= (1 - 0.82 * shb)[..., None]
        # heat glow of DIE
        hv = heat(t)
        if hv > 0.01 and sc.die.visible(t):
            dl = np.zeros((H // 2, W // 2, 1), np.float32)
            for (Ms, a, tint, mask) in sc.die.draws(t):
                if tint is None:
                    draw_items_half(dl, sc.die.rgba(t), [Sc(0.5) @ C @ M for M in Ms], a)
            d = dl[..., 0]
            gl = cv2.GaussianBlur(d, (0, 0), 16) * 0.35 + cv2.GaussianBlur(d, (0, 0), 40) * 1.25
            gl = cv2.resize(gl, (W, H), interpolation=cv2.INTER_LINEAR)
            frame += gl[..., None] * np.array([0.75, 0.05, 0.03], np.float32) * hv
        # type
        for kind, w, rg, X, a in lay:
            if kind == 's':
                M = X
                draw_items(frame, rg, [M @ T(w.size * 0.02 * w.rs, w.size * 0.02 * w.rs)], a * 0.45, np.array([0.22, 0.0, 0.01], np.float32)) if w.style != 'blood' else None
                draw_items(frame, rg, [M], a)
                fl = w.flash * dec(t, w.t_in, 0.3)
                if fl > 0.01:
                    draw_items(frame, rg, [M @ T(rg.shape[1] / 2, rg.shape[0] / 2) @ Sc(1 + 0.05 * (1 - dec(t, w.t_in, 0.3))) @ T(-rg.shape[1] / 2, -rg.shape[0] / 2)],
                               min(1., fl) * 0.5, np.array([1.0, 0.25, 0.18], np.float32) if w.style == 'blood' else np.array([1.0, 0.86, 0.76], np.float32))
            else:
                for (Ms, al, tint, mask) in X:
                    draw_items(frame, rg, [C @ M for M in Ms], al, tint)
        # DIE heats: emissive core on top of the blood
        if hv > 0.25:
            for (Ms, a, tint, mask) in sc.die.draws(t):
                if tint is None:
                    draw_items(frame, sc.die.rgba(t), [C @ M for M in Ms], a * min(0.55, (hv - 0.25) * 0.6), np.array([1.0, 0.30, 0.16], np.float32))
    else:
        # music stops: only DIE's afterglow remains
        frame = fx.bg[40:40 + H, 40:40 + W] * 0.5
        k = dec(t, END_T, 0.22)
        if k > 0.01:
            for (Ms, a, tint, mask) in sc.die.draws(END_T - 0.001):
                if tint is None:
                    draw_items(frame, sc.die.rgba(END_T - 0.001), [C @ M for M in Ms], a * k * 0.8)
        sc.embers.draw(frame, t)

    # bloom
    small = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 0.28, 0, None)
    bl = cv2.GaussianBlur(br, (0, 0), 8) * 0.45 + cv2.GaussianBlur(br, (0, 0), 28) * 0.45
    frame += cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([1.0, 0.5, 0.45], np.float32)
    # subtle lens fringe
    c = 0.0009 + 0.0025 * dec(t, DROP, 0.25) * (t >= DROP) + hits_dec(t, 0.15, 0.0012)
    Rr = cv2.warpAffine(frame[..., 0], (T(W / 2, H / 2) @ Sc(1 + c) @ T(-W / 2, -H / 2))[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    Bb = cv2.warpAffine(frame[..., 2], (T(W / 2, H / 2) @ Sc(1 - c) @ T(-W / 2, -H / 2))[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    frame[..., 0] = Rr; frame[..., 2] = Bb
    # walls close in before the drop
    if 11.7 < t < DROP:
        k = sstep(11.7, DROP, t) ** 2
        frame *= (1 - 0.5 * k * (1 - fx.vign_tight))
    frame *= 1 - sstep(DUR - 0.35, DUR - 0.02, t)
    # film
    flick = 1 + 0.02 * fx.nf(t, 7.0)
    frame *= flick * fx.vign
    lum = np.clip(frame.mean(-1, keepdims=True), 0, 1)
    frame += fx.grain[int(rng.integers(0, len(fx.grain)))] * (0.015 + 0.06 * lum)
    sr = np.random.default_rng(fi // 4 + 199)
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


def draw_items_half(canvas, rgba, Ms, alpha):
    """alpha-only splat into a half-res single channel layer"""
    h, w = rgba.shape[:2]
    Hh, Wh = canvas.shape[:2]
    M = Ms[0]
    s = math.sqrt(abs(np.linalg.det(M[:2, :2])))
    src = rgba[..., 3]
    pre = np.eye(3)
    if s < 0.7:
        f = min(1., s * 1.35)
        src = cv2.resize(src, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        pre = Sc(w / src.shape[1], h / src.shape[0])
    o = cv2.warpAffine(src, (M @ pre)[:2], (Wh, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    canvas[..., 0] = np.maximum(canvas[..., 0], o * alpha)


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
