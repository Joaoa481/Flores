#!/usr/bin/env python3
"""Donde florece mi cielo: the whole film (45 s, 60 fps), Firefly Pointillism.
Deterministic and parallel: every particle position is an analytic function of time, any frame renders alone.

One pool of points for the whole film; in every scene each point "is" something:
  L light    2      : the guiding light (+ its halo) → the breeze → the flower's heart
  S stars    1400   : far stars
  W grass    5000   : bokeh → the meadow (focus ripple from the touch) → bent by rain → blown by the gust
  F flower   ~3500  : the seed → stem, leaves, bud, petals → drooping in the rain → lifted again
  C cloud    2600   : the cloud from the right → blown by the gust into stars
  R rain     900    : inside the cloud → rain → blown into fireflies
  M meadow   2880   : dark buds → 320 small flowers blooming across the hill
  X fireflies       : fireflies → gather into the title

  python film.py still 1.0 9.0 17.0       # test frames → out/stills/
  python film.py render 0 45 out/video.mp4   (WORKERS, default 4)
"""
import os, sys, math, subprocess
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from engine import (W, H, SEED, c01, sstep, e_io, e_out, key, pulse, grid_sample, splat, up2, background, tiles,
                    tile_full, LV_SIG, LV_RES, LV_GAIN, EXPO, HALO)
from timeline import *            # noqa: F401,F403

FPS = 60
FONT = os.path.join(HERE, 'fonts', 'Lora.ttf')
BASE = np.array([1180.0, 862.0])          # where the flower grows
TOUCH = BASE + np.array([0.0, -6.0])
C0 = np.array([960.0, 560.0])

FLOWER, FLOWERF = np.array([1.0, .74, .34]), np.array([1.0, .62, .28])   # protagonist (her), exclusive
LIME = np.array([.78, 1.0, .46])                                          # the guiding light (him), exclusive
FIRE = np.array([.86, 1.0, .40])
STAR = np.array([.86, .90, 1.0])
CLOUD_B, CLOUD_S = np.array([.40, .54, .50]), np.array([.66, .82, .74])
TITLE_C = np.array([1.0, .90, .58])
GRASS_A, GRASS_B = np.array([.32, .62, .20]), np.array([.62, .88, .30])


def rot(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------- camera
Z_K = [(0, 1.0), (T_DESC, 1.0), (T_TOUCH, 1.0), (T_GROW, 1.0), (T_BLOOM, 1.36), (T_CLOUD, 1.38), (T_FLARE, 1.02),
       (T_LIFT, 1.0), (T_FIELD, 1.0), (35.5, .72), (DUR, .72)]
X_K = [(0, -150), (T_DESC, -150), (T_TOUCH, 140), (T_GROW, 160), (T_BLOOM, 215), (T_CLOUD, 215), (T_FLARE, 120),
       (T_LIFT, 110), (T_FIELD, 110), (35.5, 60), (DUR, 60)]
Y_K = [(0, -140), (T_DESC, -140), (T_TOUCH, 60), (T_GROW, 80), (T_BLOOM, 60), (T_CLOUD, 55), (T_FLARE, -20),
       (T_LIFT, -20), (T_FIELD, -10), (35.5, -40), (DUR, -40)]


def cam(t):
    z, ox, oy = key(t, Z_K), key(t, X_K), key(t, Y_K)
    ox += 38 * math.exp(-((t - (T_GUST + .9)) / .55) ** 2)            # the gust nudges the camera
    return z, ox, oy


def to_screen(pos, dz, t):
    z, ox, oy = cam(t); k = 1 + .38 * dz; zz = 1 + (z - 1) * k
    return np.stack([C0[0] + (pos[:, 0] - C0[0] - ox * k) * zz, C0[1] + (pos[:, 1] - C0[1] - oy * k) * zz], 1)


def to_world0(scr, t):          # inverse for depth 0 (the title is placed in screen space)
    z, ox, oy = cam(t)
    return np.stack([C0[0] + ox + (scr[:, 0] - C0[0]) / z, C0[1] + oy + (scr[:, 1] - C0[1]) / z], 1)


# ---------------------------------------------------------------- weather and story curves
def storm(t): return float(sstep(T_CLOUD + 1.0, T_RAIN + 1.0, t) * (1 - sstep(T_GUST + .2, T_STARS, t)))
def droop(t): return key(t, [(0, 0), (T_RAIN + .5, 0), (T_HUSH + .3, 1.0), (T_GUST + .6, 1.0), (T_LIFT + 1.2, 0), (DUR, 0)])
def gust_amt(x, t):
    d = .55 * c01((x + 200) / 2400.0)
    return np.exp(-((t - (T_GUST + .45 + d)) / .5) ** 2)
def wind(x, t):                 # lean added to grass and stem (positive = to the right)
    return -.42 * storm(t) + .8 * gust_amt(x, t)


# ---------------------------------------------------------------- the flower
def flower_frame(t):
    g = e_io(c01((t - T_GROW) / (T_BUD - T_GROW)))
    lo = e_io(c01((t - T_LEAF) / 1.6))
    bud = e_io(c01((t - (T_LEAF + .6)) / (T_BUD - T_LEAF - .6)))
    b = e_io(c01((t - T_OPEN) / (T_BLOOM - T_OPEN)))
    d = droop(t)
    sway = .035 * math.sin(.9 * t) + .02 * math.sin(1.7 * t + 1) + .25 * float(wind(np.array([BASE[0]]), t)[0]) * (1 - .5 * d)
    Hs = 300.0 * g
    s = np.linspace(0, 1, 65)
    th = sway + s * (.16 + 1.45 * d) + s ** 2 * .1 * math.sin(.6 * t)
    dx = np.sin(th) * Hs / 64; dy = -np.cos(th) * Hs / 64
    xs = BASE[0] + np.concatenate([[0], np.cumsum((dx[1:] + dx[:-1]) * .5)])
    ys = BASE[1] + np.concatenate([[0], np.cumsum((dy[1:] + dy[:-1]) * .5)])
    tip = np.array([xs[-1], ys[-1]]); tdir = np.array([math.sin(th[-1]), -math.cos(th[-1])])
    head = tip + tdir * (14 + 10 * b)
    return dict(g=g, lo=lo, bud=bud, b=b, d=d, s=s, xs=xs, ys=ys, th=th, tip=tip, head=head, tth=th[-1])


def stem_at(F, u):
    x = np.interp(u, F['s'], F['xs']); y = np.interp(u, F['s'], F['ys']); a = np.interp(u, F['s'], F['th'])
    return np.stack([x, y], 1), a


# ---------------------------------------------------------------- text
def text_mask(lines):
    img = Image.new('L', (W, H), 0); d = ImageDraw.Draw(img)
    for s, size, x0, yc in lines:
        f = ImageFont.truetype(FONT, size); f.set_variation_by_name('Bold')
        bb = d.textbbox((0, 0), 'Hgá', font=f)
        d.text((x0, yc - (bb[3] - bb[1]) / 2 - bb[1]), s, font=f, fill=255)
    return np.array(img) > 110


class Model:
    def __init__(self):
        rng = np.random.default_rng(SEED); self.rng = rng
        R = rng.random
        # ---- stars
        n = 1400; self.S = dict(x=rng.uniform(-500, 2500, n), y=-320 + 900 * R(n) ** 1.6, lvl=np.where(R(n) < .08, 2.4, .2 + .9 * R(n)),
                                lum=.035 + .13 * R(n) ** 2, ph=R(n) * 6.28, rate=.5 + 1.2 * R(n), t0=.6 + 2.6 * R(n))
        # ---- grass: blades in three depth bands
        nbl, per = 1000, 5
        bx = rng.uniform(-750, 2750, nbl); band = rng.choice(3, nbl, p=[.34, .44, .22])
        by = np.where(band == 0, rng.uniform(848, 880, nbl), np.where(band == 1, rng.uniform(875, 960, nbl), rng.uniform(980, 1130, nbl)))
        bh = np.where(band == 0, rng.uniform(16, 38, nbl), np.where(band == 1, rng.uniform(30, 80, nbl), rng.uniform(80, 190, nbl)))
        bi = np.repeat(np.arange(nbl), per)
        n = nbl * per
        self.W = dict(x=bx[bi], y=by[bi], h=bh[bi], lean=rng.normal(0, .2, nbl)[bi], ph=rng.uniform(0, 6.28, nbl)[bi], band=band[bi],
                      s=np.sqrt(R(n)), mix=R(n), r=R(n))
        self.W['dz'] = np.where(self.W['band'] == 0, -.35, np.where(self.W['band'] == 1, 0.0, .55))
        self.W['lvl'] = np.where(self.W['band'] == 0, 1.1, np.where(self.W['band'] == 1, .55, 2.3)) + .3 * R(n)
        self.W['lum'] = np.where(self.W['band'] == 0, .10, np.where(self.W['band'] == 1, .16, .075)) * (.6 + .7 * R(n))
        dist = np.hypot(self.W['x'] - TOUCH[0], (self.W['y'] - TOUCH[1]) * 2.5)
        self.W['ta'] = T_TOUCH + dist / 760.0
        self.W['df'] = np.hypot(self.W['x'] - BASE[0], (self.W['y'] - BASE[1]) * 2.5)
        # ---- flower
        nS, nLf, nP, nC = 620, 2 * 360, 12 * 140, 520
        self.nF = nS + nLf + nP + nC
        F = {}
        F['kind'] = np.concatenate([np.zeros(nS), np.ones(nLf), np.full(nP, 2), np.full(nC, 3)]).astype(int)
        F['u'] = R(self.nF); F['side'] = np.where(R(self.nF) < .5, -1.0, 1.0)
        F['out'] = R(self.nF) < .66
        F['leaf'] = np.concatenate([np.zeros(nS), np.repeat([0, 1], 360), np.zeros(nP + nC)]).astype(int)
        F['pet'] = np.concatenate([np.zeros(nS + nLf), np.repeat(np.arange(12), 140), np.zeros(nC)]).astype(int)
        F['psi'] = R(self.nF) * 2 * math.pi; F['rr'] = np.sqrt(R(self.nF))
        F['ci'] = np.concatenate([np.zeros(nS + nLf + nP), np.arange(nC)])
        F['jit'] = rng.normal(0, 1, (self.nF, 2))
        F['ph'] = R(self.nF) * 6.28; F['rate'] = .8 + .9 * R(self.nF)
        F['spark'] = R(self.nF) < .04
        F['lvl'] = np.where(R(self.nF) < .05, 1.6, .25 + .35 * R(self.nF))
        self.Fp = F; self.nC_ = nC
        # ---- cloud
        n = 2600
        blobs = np.array([(-270, 10, 120), (-110, -50, 150), (60, -70, 160), (230, -10, 135), (10, 25, 170), (-190, 45, 110), (330, 40, 95)], float)
        bk = rng.integers(0, len(blobs), n); nr = rng.normal(0, 1, (n, 2))
        off = blobs[bk, :2] + nr * blobs[bk, 2:3] * .5
        off[:, 1] = np.where(off[:, 1] > 50, 50 + (off[:, 1] - 50) * .3, off[:, 1])
        sharp = R(n) < .45
        self.Cc = dict(off=off, lvl=np.where(sharp, 1.0 + .9 * R(n), 2.2 + 1.1 * R(n)), lum=np.where(sharp, .05, .06) * (.6 + .8 * R(n)), sharp=sharp,
                       ph=R(n) * 6.28, star=np.stack([rng.uniform(-450, 2450, n), -300 + 880 * R(n) ** 1.5], 1),
                       slvl=.25 + .6 * R(n), slum=.015 + .08 * R(n) ** 3, rate=.5 + R(n), r=R(n))
        # ---- rain
        n = 900
        self.Rr = dict(ox=rng.uniform(-330, 330, n), off=R(n), ff=np.stack([rng.uniform(150, 2100, n), 420 + 440 * R(n) ** .7], 1),
                       ph=R(n) * 6.28, rate=.3 + .5 * R(n), dl=.6 * R(n), lum=.08 + .14 * R(n))
        # ---- meadow flowers: 320 rosettes × 9 points
        nf = 320; q = R(nf); fx = rng.uniform(-650, 2650, nf)
        bad = (np.abs(fx - BASE[0]) < 160) & (q < .55); fx[bad] += np.where(R(bad.sum()) < .5, -260, 260)
        fy = 866 + 150 * q ** 1.5; fr = 7 + 14 * q
        rep = np.repeat(np.arange(nf), 9); k = np.tile(np.arange(9), nf)
        palette = np.array([FIRE, GRASS_B, [.95, .95, .62]])
        pc = palette[rng.choice(3, nf, p=[.45, .3, .25])]
        ta = T_FLASH + .15 + np.hypot(fx - BASE[0], (fy - BASE[1]) * 2.5) / 650.0
        self.M = dict(x=fx[rep], y=fy[rep], r=fr[rep], k=k, q=q[rep], col=pc[rep], ta=ta[rep], ph=rng.uniform(0, 6.28, nf)[rep],
                      dz=(-.3 + .9 * q)[rep], rot=rng.uniform(0, 6.28, nf)[rep])
        # ---- fireflies + title
        lines = [(TITLE[0], 62, 140, 196), (TITLE[1], 62, 140, 286)]
        mask = text_mask(lines)
        self.nT = 6200
        self.Tscr = grid_sample(mask, self.nT, rng)
        ys_, xs_ = np.nonzero(mask); self.Tc = np.array([xs_.mean(), ys_.mean()])
        n = self.nT + 1300
        self.X = dict(x=rng.uniform(-600, 2600, n), y=880 - 420 * R(n) ** 2.6, dz=rng.uniform(-.4, .5, n), lvl=rng.uniform(.6, 1.8, n),
                      lum=.035 + .07 * R(n), ph=R(n) * 6.28, rate=.25 + .45 * R(n), w1=.4 + R(n), w2=.3 + .8 * R(n),
                      p1=R(n) * 6.28, p2=R(n) * 6.28, dl=.85 * R(n), shim=R(n) * 6.28)
        self.X['ta'] = T_TOUCH + .4 + np.hypot(self.X['x'] - TOUCH[0], (self.X['y'] - TOUCH[1]) * 1.5) / 700.0
        # nearest-first: title slot i goes to the firefly whose settled position is closest along x (keeps flights short and untangled)
        self.Xord = np.argsort(self.X['x'][:self.nT] + 0.3 * self.X['y'][:self.nT])
        self.Tord = np.argsort(self.Tscr[:, 0] + 0.3 * self.Tscr[:, 1])

    # ------------------------------------------------------------ groups
    def light(self, t, F):
        P0 = np.array([700.0, 300.0])
        wob = np.array([14 * math.sin(.8 * t), 9 * math.sin(1.1 * t + 1)])
        tipoff = F['tip'] + np.array([-62.0, -34.0])
        hover = F['head'] + np.array([-150.0, -105.0])
        if t < T_DESC:
            p = P0 + wob
        elif t < T_TOUCH:
            u = e_io((t - T_DESC) / (T_TOUCH - T_DESC))
            c1, c2 = np.array([1260.0, 230.0]), np.array([1520.0, 700.0])
            p = (1 - u) ** 3 * (P0 + wob) + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u * u * c2 + u ** 3 * TOUCH
        elif t < T_BLOOM:
            u = e_io(c01((t - T_GROW) / 1.2)); p = TOUCH * (1 - u) + tipoff * u
        elif t < T_CLOUD:
            a = -math.pi / 2 - 2 * math.pi * e_io((t - T_BLOOM) / (T_CLOUD - T_BLOOM))
            circ = F['head'] + 135 * np.array([math.cos(a), .55 * math.sin(a)])
            u = e_io(c01((t - T_BLOOM) / .7)); p = tipoff * (1 - u) + circ * u
            u2 = e_io(c01((t - (T_CLOUD - .8)) / .8)); p = p * (1 - u2) + hover * u2
        elif t < T_GUST + .2:
            p = hover + .5 * wob
            u = e_io(c01((t - T_FLARE) / (T_GUST + .2 - T_FLARE))); p = p * (1 - u) + np.array([230.0, 400.0]) * u
        elif t < T_STARS:
            u = e_io((t - T_GUST - .2) / (T_STARS - T_GUST - .2))
            a, c, b = np.array([230.0, 400.0]), np.array([1100.0, 150.0]), np.array([2080.0, 300.0])
            p = (1 - u) ** 2 * a + 2 * (1 - u) * u * c + u * u * b
        elif t < T_FLASH:
            u = e_io((t - T_STARS) / (T_FLASH - T_STARS))
            a, c = np.array([2080.0, 300.0]), np.array([1750.0, 120.0])
            b = F['head']
            p = (1 - u) ** 2 * a + 2 * (1 - u) * u * c + u * u * b
        else:
            p = F['head'].copy()
        lvl = key(t, [(0, 3.9), (T_LIGHT, 3.9), (T_LIGHT + .8, .2), (DUR, .2)])
        lum = key(t, [(0, 0), (T_LIGHT, 2.5), (T_LIGHT + .8, 11.0), (T_CLOUD, 11.0), (T_RAIN + 1, 6.0), (T_FLARE - .05, 6.0), (T_FLASH, 11.0), (DUR, 7.0)])
        lum *= 1 + .1 * math.sin(2.1 * t)
        lum *= 1 + 4.0 * math.exp(-((t - T_FLARE) / .22) ** 2) + 1.5 * math.exp(-((t - T_FLASH) / .1) ** 2)
        lum *= 1 - .25 * sstep(T_OUT, DUR, t)
        pos = np.array([p, p]); return dict(pos=pos, dz=np.zeros(2), lvl=np.array([lvl, min(3.9, lvl + 2.8)]),
                                            lum=np.array([lum, lum * .3]), col=np.array([LIME, LIME]))

    def stars(self, t):
        S = self.S; n = len(S['x'])
        tw = .65 + .35 * np.sin(S['rate'] * t + S['ph'])
        lum = S['lum'] * tw * sstep(S['t0'], S['t0'] + 1.2, t) * (1 - .6 * storm(t))
        lum *= 1 - .2 * sstep(T_OUT, DUR, t)
        return dict(pos=np.stack([S['x'], S['y']], 1), dz=np.full(n, -.7), lvl=S['lvl'], lum=lum, col=np.tile(STAR, (n, 1)))

    def grass(self, t):
        Wd = self.W; n = len(Wd['x'])
        ln = Wd['lean'] + .07 * np.sin(1.1 * t + Wd['ph']) + wind(Wd['x'], t) * (.75 + .25 * np.sin(2.3 * t + Wd['ph']))
        cv = .25 * Wd['s'] * ln
        x = Wd['x'] + np.sin(ln + cv) * Wd['h'] * Wd['s']; y = Wd['y'] - np.cos(ln + cv) * Wd['h'] * Wd['s']
        ap = e_out(c01((t - Wd['ta']) / .9))
        lvl = 3.6 * (1 - ap) + Wd['lvl'] * ap
        flash = 1 + 1.8 * np.exp(-((t - Wd['ta']) / .12) ** 2) + 2.2 * np.exp(-((t - T_FLASH - Wd['df'] / 1400.0) / .09) ** 2)
        lum = Wd['lum'] * (.05 + .95 * ap) * sstep(T_TOUCH - .4, T_TOUCH, t) * flash * (1 - .45 * storm(t)) * (1 + .15 * sstep(T_FLASH, T_FLASH + 1, t))
        col = GRASS_A[None] * (1 - Wd['mix'][:, None]) + GRASS_B[None] * Wd['mix'][:, None]
        return dict(pos=np.stack([x, y], 1), dz=Wd['dz'], lvl=lvl, lum=lum, col=col)

    def flower(self, t, F):
        P = self.Fp; n = self.nF; pos = np.zeros((n, 2)); lum = np.zeros(n)
        k = P['kind']; g, lo, bud, b, d = F['g'], F['lo'], F['bud'], F['b'], F['d']
        # stem: two edges of a thin tapered tube along the grown length
        m = k == 0
        sp, sa = stem_at(F, P['u'][m] * .97)
        nrm = np.stack([np.cos(sa), np.sin(sa)], 1)
        wdt = (5.0 - 2.0 * P['u'][m]) * np.where(P['out'][m], P['side'][m], P['side'][m] * .3 * P['rr'][m])
        pos[m] = sp + nrm * wdt[:, None] + P['jit'][m] * .6
        lum[m] = np.where(P['out'][m], .5, .18) * min(1.0, .12 + 1.1 * g)
        # leaves: lens shapes from the stem
        m = k == 1
        lf = P['leaf'][m]; su = np.where(lf == 0, .33, .52); sg = np.where(lf == 0, -1.0, 1.0)
        bp_, ba = stem_at(F, su)
        ang = ba + sg * ((.18 + .85 * lo) - .5 * d * (1 + .4 * sg))
        ax = np.stack([np.sin(ang), -np.cos(ang)], 1); an = np.stack([np.cos(ang), np.sin(ang)], 1)
        L = 108 * lo * g; q = P['u'][m]
        hw = 30 * lo * np.sin(math.pi * q) ** .9
        bend = 18 * lo * q * q
        across = np.where(P['out'][m], P['side'][m], P['side'][m] * P['rr'][m] * .85)
        pos[m] = bp_ + ax * (L * q)[:, None] + an * (hw * across + sg * bend)[:, None] + P['jit'][m] * .7
        lum[m] = np.where(P['out'][m], .45, .14) * sstep(0, .15, lo)
        # head frame: local u axis across, v axis along the stem direction (the rosette squashes as the head bows)
        tth = F['tth']; Rh = rot(tth)
        squash = 1 - .5 * d
        beff = b * (1 - .3 * d)
        # petals
        m = k == 2
        pk = P['pet'][m]
        a_open = pk * (2 * math.pi / 12) + .13 + .9 * math.sin(math.pi * b) * (1 - b * .3)
        a_bud = -math.pi / 2 + (pk / 11.0 - .5) * .55
        a = a_bud * (1 - beff) + a_open * beff
        Lp = (12 + 24 * bud) * (1 - beff) + np.where(pk % 2 == 0, 80, 66) * beff
        wp = (6 + 5 * bud) * (1 - beff) + 26 * beff
        r0 = 6 + 22 * beff
        cen = (r0 + Lp / 2)[:, None] * np.stack([np.cos(a), np.sin(a)], 1)
        ps = P['psi'][m]; rr = np.where(P['out'][m], 1.0, P['rr'][m] * .9)
        e1 = np.stack([np.cos(a), np.sin(a)], 1); e2 = np.stack([-np.sin(a), np.cos(a)], 1)
        loc = cen + e1 * (Lp / 2 * rr * np.cos(ps))[:, None] + e2 * (wp / 2 * rr * np.sin(ps))[:, None]
        loc[:, 1] *= squash
        pos[m] = F['head'] + loc @ Rh.T + P['jit'][m] * .5
        back = (pk % 2 == 1)
        lum[m] = np.where(P['out'][m], .5, .13) * np.where(back, .7, 1.0) * sstep(T_LEAF + .4, T_BUD - .3, t) * (.3 + .7 * b)
        # centre: a sunflower spiral
        m = k == 3
        ci = P['ci'][m]; Rc = 30 * (.3 + .7 * beff)
        rr = Rc * np.sqrt((ci + .5) / self.nC_); aa = ci * 2.39996
        loc = np.stack([rr * np.cos(aa), rr * np.sin(aa) * squash], 1)
        pos[m] = F['head'] + loc @ Rh.T
        ring = rr > Rc * .82
        lum[m] = np.where(ring, .42, .2) * sstep(T_LEAF + .8, T_BUD, t) * (.3 + .7 * b)
        # seed before growth: everything sits at the touch point, barely glowing
        seed = 1 - sstep(T_GROW - .1, T_GROW + .5, t)
        lum = lum * (1 - seed) + .0007 * seed * sstep(T_TOUCH, T_TOUCH + .6, t)
        pos = pos * (1 - seed) + (TOUCH + P['jit'] * 4) * seed
        # life
        breath = 1 + .22 * np.sin(P['rate'] * t + P['ph'])
        lum = lum * breath * np.where(P['spark'], 2.4, 1.0)
        lum *= 1 - .62 * d
        lum *= 1 + 2.2 * math.exp(-((t - T_FLASH) / .1) ** 2) + .9 * math.exp(-((t - T_BLOOM) / .15) ** 2)
        lum *= 1 + .12 * sstep(T_FLASH, T_FLASH + 1, t)
        col = np.where((k == 3)[:, None], FLOWERF[None], FLOWER[None])
        return dict(pos=pos, dz=np.zeros(n), lvl=P['lvl'], lum=lum, col=np.broadcast_to(col, (n, 3)).copy())

    def cloud_origin(self, t):
        x = key(t, [(0, 2700), (T_CLOUD - .5, 2700), (24.5, 1250), (T_FLARE, 1130), (DUR, 1130)])
        return np.array([x, 300.0])

    def cloud(self, t):
        Cc = self.Cc; n = len(Cc['lvl'])
        O = self.cloud_origin(min(t, T_GUST))
        drift = np.stack([10 * np.sin(.4 * t + Cc['ph']), 6 * np.sin(.55 * t + 2 * Cc['ph'])], 1)
        p0 = O + Cc['off'] + drift
        settle = O + Cc['off']
        dl = .55 * c01((settle[:, 0] - 700) / 1100)
        u = e_io(c01((t - T_GUST - dl) / 1.25))
        sx = np.sin(math.pi * u)
        p = p0 * (1 - u[:, None]) + Cc['star'] * u[:, None] + np.stack([560 * sx, -150 * (Cc['r'] - .3) * sx], 1)
        lvl = Cc['lvl'] * (1 - u) + Cc['slvl'] * u
        tw = .65 + .35 * np.sin(Cc['rate'] * t + Cc['ph'])
        lum0 = Cc['lum'] * sstep(T_CLOUD - .5, T_CLOUD + 1.2, t)
        lum = lum0 * (1 - u) + Cc['slum'] * tw * u
        lum *= 1 - .2 * sstep(T_OUT, DUR, t)
        col0 = np.where(Cc['sharp'][:, None], CLOUD_S[None], CLOUD_B[None])
        col = col0 * (1 - u[:, None]) + STAR[None] * u[:, None]
        dz = -.1 * (1 - u) - .7 * u
        return dict(pos=p, dz=dz, lvl=lvl, lum=lum, col=col)

    def rain(self, t):
        Rr = self.Rr; n = len(Rr['ox'])
        O = self.cloud_origin(min(t, T_GUST))
        fall = (t * 900 + Rr['off'] * 760) % 760
        x = O[0] + Rr['ox'] - .24 * fall; y = O[1] + 70 + fall
        p0 = np.stack([x, y], 1)
        u = e_io(c01((t - T_GUST - .1 - Rr['dl']) / 1.3))
        sx = np.sin(math.pi * u)
        drift = np.stack([22 * np.sin(Rr['rate'] * t + Rr['ph']), 14 * np.sin(.7 * Rr['rate'] * t + 2 * Rr['ph'])], 1)
        # once blown, the drop freezes where it was and flies to its firefly place
        frozen = np.stack([O[0] + Rr['ox'] - .24 * ((T_GUST * 900 + Rr['off'] * 760) % 760),
                           O[1] + 70 + ((T_GUST * 900 + Rr['off'] * 760) % 760)], 1)
        start = frozen if t > T_GUST else p0
        p = start * (1 - u[:, None]) + (Rr['ff'] + drift) * u[:, None] + np.stack([480 * sx, -80 * sx], 1)
        rl = .085 * sstep(T_RAIN, T_RAIN + .7, t)
        fl = Rr['lum'] * (.02 + 1.3 * pulse(Rr['rate'] * t * .5 + Rr['ph'] / 6.28, .08))
        lum = rl * (1 - u) + fl * u
        lum *= 1 - .3 * sstep(T_OUT, DUR, t)
        col = CLOUD_S[None] * (1 - u[:, None]) + FIRE[None] * u[:, None]
        return dict(pos=p, dz=np.full(n, .1), lvl=1.0 * (1 - u) + 1.2 * u, lum=lum, col=col)

    def meadow(self, t):
        M = self.M; n = len(M['x'])
        u = e_out(c01((t - M['ta']) / .8))
        sway = 3 * np.sin(1.1 * t + M['ph']) + 20 * M['q'] * wind(M['x'], t)
        ring = M['k'] < 8
        a = M['rot'] + M['k'] * (2 * math.pi / 8) + .02 * t
        rr = np.where(ring, M['r'] * (.25 + .75 * u), 0.0)
        x = M['x'] + sway + rr * np.cos(a); y = M['y'] - M['r'] * 1.2 + rr * np.sin(a) * .6
        lvl = 3.0 * (1 - u) + (np.where(M['q'] > .85, 1.4, .3 + .4 * (1 - M['q']))) * u
        flash = 1 + 2.0 * np.exp(-((t - M['ta']) / .1) ** 2)
        lum = np.where(ring, .26, .45) * (.5 + .5 * M['q']) * u * flash * (1 + .2 * np.sin(1.3 * t + M['ph']))
        return dict(pos=np.stack([x, y], 1), dz=M['dz'], lvl=lvl, lum=lum, col=M['col'])

    def fireflies(self, t):
        X = self.X; n = len(X['x'])
        p = np.stack([X['x'] + 26 * np.sin(X['w1'] * t + X['p1']) + 10 * np.sin(1.7 * X['w2'] * t),
                      X['y'] + 18 * np.sin(X['w2'] * t + X['p2'])], 1)
        lum = X['lum'] * np.where(np.arange(n) < self.nT, .3, 1.0) * (.01 + 1.5 * pulse(X['rate'] * t + X['ph'] / 6.28, .075)) * sstep(0, .8, t - X['ta'])
        lum = lum * (1 - .92 * storm(t)) * (1 + 2.0 * np.exp(-((t - T_FLASH - np.hypot(X['x'] - BASE[0], X['y'] - BASE[1]) / 1400.0) / .09) ** 2))
        lvl = X['lvl'].copy(); col = np.tile(FIRE, (n, 1)); dz = X['dz'].copy()
        # the title: the first nT fireflies swirl into the letters
        if t > T_TITLE0 - .05:
            i = self.Xord; tgt_scr = self.Tscr[self.Tord]
            tgt = to_world0(tgt_scr + .25 * np.stack([np.sin(3 * t + X['shim'][i]), np.cos(2.6 * t + X['shim'][i])], 1), t)
            dur = T_TITLE - T_TITLE0 - .85
            ue = e_io(c01((t - T_TITLE0 - X['dl'][i]) / dur))
            Tc = to_world0(self.Tc[None], t)[0]
            sw = 2.0 * (1 - ue)
            st = p[i] * (1 - ue[:, None]) + tgt * ue[:, None] - Tc
            cs, sn = np.cos(sw), np.sin(sw)
            p[i] = Tc + np.stack([cs * st[:, 0] - sn * st[:, 1], sn * st[:, 0] + cs * st[:, 1]], 1)
            tl = .30 * (1 + 1.3 * math.exp(-((t - T_TITLE) / .12) ** 2)) * (1 - .55 * sstep(T_OUT, DUR, t))
            lum[i] = lum[i] * (1 - ue) + tl * ue
            lvl[i] = lvl[i] * (1 - ue) + .22 * ue
            dz[i] = dz[i] * (1 - ue)
            col[i] = FIRE[None] * (1 - ue[:, None]) + TITLE_C[None] * ue[:, None]
        return dict(pos=p, dz=dz, lvl=lvl, lum=lum, col=col)

    def evaluate(self, t):
        F = flower_frame(t)
        gs = [self.stars(t), self.grass(t), self.meadow(t), self.cloud(t), self.rain(t), self.fireflies(t), self.flower(t, F), self.light(t, F)]
        return {k: np.concatenate([g[k] for g in gs]) for k in ('pos', 'dz', 'lvl', 'lum', 'col')}


# ---------------------------------------------------------------- render
MODEL = None
def get_model():
    global MODEL
    if MODEL is None: MODEL = Model()
    return MODEL


def render(t, fi=0):
    M = get_model(); sh = .25 / FPS
    a = M.evaluate(t - sh); b = M.evaluate(t + sh)
    p0 = to_screen(a['pos'], a['dz'], t - sh); p1 = to_screen(b['pos'], b['dz'], t + sh)
    lvl = np.clip((a['lvl'] + b['lvl']) * .5, 0, 3.999); lum = np.maximum(0, (a['lum'] + b['lum']) * .5)
    col = (a['col'] + b['col']) * .5
    jump = np.linalg.norm(p1 - p0, axis=1) > 260          # rain wraps around: no trail across the frame
    p0 = np.where(jump[:, None], p1, p0)
    keep = lum > 1e-4
    p0, p1, lvl, lum, col = p0[keep], p1[keep], lvl[keep], lum[keep], col[keep]
    L0 = np.floor(lvl).astype(int); fr = lvl - L0
    bufs = []
    for k in range(5):
        res = LV_RES[k]; h, w = H // res, W // res
        wk = np.where(L0 == k, 1 - fr, 0) + np.where(L0 + 1 == k, fr, 0)
        wk = wk + HALO * (np.where(np.minimum(L0 + 2, 4) == k, 1 - fr, 0) + np.where(np.minimum(L0 + 3, 4) == k, fr, 0))
        m = wk > 1e-5
        if not m.any(): bufs.append(None); continue
        q0, q1, ww, cc = p0[m] / res, p1[m] / res, (wk * lum)[m] * LV_GAIN[k] / (res * res), col[m]
        ln = np.linalg.norm(q1 - q0, axis=1); ns = np.clip(np.ceil(ln / 1.1), 1, 40).astype(int)
        rep = np.repeat(np.arange(len(ns)), ns)
        start = np.repeat(np.cumsum(ns) - ns, ns); j = np.arange(rep.size) - start
        f = (j + .5) / ns[rep]
        q = q0[rep] + (q1[rep] - q0[rep]) * f[:, None]
        bufs.append(splat(h, w, q[:, 0], q[:, 1], ww[rep] / ns[rep], cc[rep]))
    acc = None
    for k in (4, 3, 2):
        res = LV_RES[k]; s = LV_SIG[k] / res
        cur = gaussian_filter(bufs[k], (s, s, 0), truncate=2.6) if bufs[k] is not None else np.zeros((H // res, W // res, 3), np.float32)
        acc = cur if acc is None else cur + up2(acc)
    acc = up2(acc)
    if bufs[1] is not None: acc += gaussian_filter(bufs[1], (LV_SIG[1], LV_SIG[1], 0), truncate=2.6)
    if bufs[0] is not None: acc += gaussian_filter(bufs[0], (LV_SIG[0], LV_SIG[0], 0), truncate=2.8)
    lin = background() + acc * EXPO
    img = 1 - np.exp(-lin)
    img = np.power(np.clip(img, 0, 1), 1 / 2.2).astype(np.float32)
    G_, U = tiles(); rg = np.random.default_rng(fi * 31 + 5)
    k = fi % 4; ox, oy = int(rg.integers(0, 512)), int(rg.integers(0, 512))
    grain = tile_full(G_, k, ox, oy); dith = tile_full(U, (k + 1) % 4, oy, ox)
    lum_ = img.mean(2, keepdims=True)
    img = img + (grain[..., None] * .0045 * (.35 + .65 * (1 - lum_))) + (dith[..., None] - .5) / 255.0 * 1.3
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)


def _work(fi):
    return fi, render(fi / FPS, fi).tobytes()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'render'
    if cmd == 'still':
        sd = os.environ.get('STILLS', os.path.join(HERE, 'out', 'stills')); os.makedirs(sd, exist_ok=True)
        for ts in sys.argv[2:]:
            t = float(ts); fi = int(round(t * FPS))
            Image.fromarray(render(t, fi)).save(os.path.join(sd, 's_%05.2f.jpg' % t), quality=90)
            print('still', t, flush=True)
        return
    if cmd == 'render':
        a, b = float(sys.argv[2]), float(sys.argv[3]); out = sys.argv[4]
        f0, f1 = int(round(a * FPS)), int(round(b * FPS))
        workers = int(os.environ.get('WORKERS', '4'))
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        ff = subprocess.Popen([os.environ.get('FFMPEG', 'ffmpeg'), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                               '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17',
                               '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers, initializer=get_model) as pool:
            for fi, buf in pool.imap(_work, range(f0, f1), chunksize=4):
                ff.stdin.write(buf)
                if fi % 300 == 0: print('frame', fi, '/', f1, flush=True)
        ff.stdin.close(); ff.wait()
        print('wrote', out)


if __name__ == '__main__':
    main()
