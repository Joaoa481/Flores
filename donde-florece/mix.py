#!/usr/bin/env python3
"""Donde florece mi cielo, sound: score (real instrument samples, D Lydian / B minor) + crickets, wind, rain + foley → out/mix.wav.
Every time point comes from timeline.py, the same file the picture uses. Run with the library's .venv and LIB set."""
import os, sys, math
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ['LIB']
sys.path.insert(0, LIB); sys.path.insert(0, HERE)
from core.audio import sampler as S
from core.audio.sfx import SR, bp, lp, hp, add, compress, limit, whoosh
from timeline import *            # noqa: F401,F403

N = int(round(DUR * SR)); tt = np.arange(N) / SR
rng = np.random.default_rng(722)
def env(pts):
    a = np.array(pts, float); return np.interp(tt, a[:, 0], a[:, 1])
def st(x, pan=0.0):
    th = (pan + 1) * math.pi / 4; return np.stack([x * math.cos(th), x * math.sin(th)], 1)

S.seed(7)
ev = []
def E(t, inst, p, d, v, pan=0.0, g=1.0, **kw):
    ev.append(dict(t=t, inst=inst, pitch=p, dur=d, vel=v, pan=pan, gain=g, **kw))
def chord(t, inst, ps, d, v, step=0.0, g=1.0, pan0=-.35):
    for i, p in enumerate(ps): E(t + step * i, inst, p, d, v, pan0 + .7 * i / max(1, len(ps) - 1), g)
def pad(t, d, notes, v=.28, g=.6, att=.6):
    insts = ['contrabass', 'cellos', 'violas', 'violins', 'violins']
    pans = {'contrabass': 0, 'cellos': -.3, 'violas': .05, 'violins': .3}
    for inst, p in zip(insts[5 - len(notes):], notes): E(t, inst, p, d, v, pans[inst], g, attack=att)
B8 = BEAT / 2

# ---- 0: the light, alone (the first sound comes with it after a breath of silence)
E(T_LIGHT, 'glass', 'A5', 2.8, .6, -.1, 1.6); E(T_LIGHT + .02, 'hand_chimes', 'D6', 2.6, .5, -.15, 1.3); E(T_LIGHT, 'harp', 'D3', 3.0, .4, 0, .9)
E(1.9, 'vibraphone', 'F#5', 1.6, .34, -.2, 1.1); E(2.7, 'vibraphone', 'E5', 1.6, .32, -.1, 1.0)
# ---- bar 2–3: the descent. Harp eighths in D Lydian, string pad; the vibraphone falls with the light
arp = ['D3', 'A3', 'E4', 'F#4', 'G#4', 'F#4', 'E4', 'A3']
for k in range(int((T_TOUCH - T_DESC) / B8)):
    E(T_DESC + k * B8, 'harp', arp[k % 8], .9, .24 + .02 * (k % 2 == 0), -.3 + .6 * (k % 8) / 7, .55)
pad(T_DESC, T_TOUCH - T_DESC + 1.0, ['D2', 'D3', 'A3', 'F#4'], .24, .5, 1.2)
for t, p in [(4.6, 'B5'), (5.8, 'A5'), (6.9, 'F#5'), (7.7, 'E5')]: E(t, 'vibraphone', p, 1.2, .24, .2, .5)
# ---- the touch: harp pluck + hand chime, a quick sweep with the focus ripple
E(T_TOUCH, 'harp', 'D3', 2.0, .55, 0, .8); E(T_TOUCH, 'harp', 'A3', 2.0, .45, .1, .7)
E(T_TOUCH, 'hand_chimes', 'D6', 2.2, .4, .15, .6)
for i, p in enumerate(['D5', 'E5', 'F#5', 'G#5', 'A5', 'B5', 'C#6', 'D6']): E(T_TOUCH + .12 + .06 * i, 'glockenspiel', p, .7, .16 + .015 * i, -.7 + .2 * i, .45)
# ---- bars 4–5: growth. Rising harp arpeggios that speed up; flute enters; cymbal swell into the bloom
tq = T_GROW; i = 0
grow = ['D3', 'A3', 'D4', 'E4', 'F#4', 'A4', 'D4', 'E4', 'G#4', 'B4', 'C#5', 'E5']
while tq < T_BLOOM - .1:
    E(tq, 'harp', grow[i % len(grow)], .9, .26 + .012 * min(i, 20), -.5 + (i % 6) / 5, .6)
    frac = (tq - T_GROW) / (T_BLOOM - T_GROW); tq += B8 * (1 - .45 * frac); i += 1
pad(T_GROW, T_BLOOM - T_GROW + .2, ['D2', 'A2', 'E3', 'G#3'], .26, .55, 1.0)
for t, p, d in [(T_LEAF, 'F#5', 1.5), (T_BUD, 'G#5', .8), (T_OPEN, 'A5', 1.6)]: E(t, 'flute', p, d, .34, .15, .65, attack=.08)
E(T_LEAF + .02, 'glass', 'E6', 1.2, .2, -.35, .4); E(T_LEAF + .05, 'glass', 'B5', 1.2, .18, .35, .35)
E(T_BLOOM - 1.5, 'sus_cymbal', 'cresc', None, .42, .1, .42)
E(T_OPEN, 'violins_trem', 'A5', T_BLOOM - T_OPEN, .24, .3, .45, attack=.6)
# ---- the bloom lands
chord(T_BLOOM, 'harp', ['D2', 'A2', 'D3', 'F#3', 'A3', 'E4', 'G#4'], 3.0, .55, step=.03)
chord(T_BLOOM, 'hand_chimes', ['D5', 'F#5', 'A5', 'E6'], 2.6, .42, g=.7)
E(T_BLOOM, 'timpani', 'D3', 1.6, .36, 0, .7); E(T_BLOOM, 'glockenspiel', 'D7', 1.2, .26, .3, .5)
pad(T_BLOOM, T_CLOUD - T_BLOOM + .4, ['D2', 'D3', 'A3', 'F#4', 'C#5'], .28, .55, .3)
# ---- bar 6: breath. The flute sings while the light circles
for t, p, d in [(17.5, 'A5', .8), (18.33, 'B5', .8), (19.17, 'A5', .4), (19.58, 'F#5', 1.2)]: E(t, 'flute', p, d, .36, .15, .7, attack=.06)
for k in range(8): E(T_BLOOM + .2 + k * 2 * B8, 'harp', ['D4', 'A4', 'G#4', 'A4'][k % 4], .8, .22, -.25, .45)
# ---- bars 7–8: the cloud. B minor, low strings, tremolo, soft timpani; everything thins out to near-silence
E(T_CLOUD, 'cellos', 'B2', 3.2, .3, -.3, .65, attack=.8)
E(T_CLOUD, 'contrabass', 'B1', T_HUSH - T_CLOUD, .28, 0, .6, attack=1.2)
E(T_CLOUD + .5, 'violas', 'F#3', 3.0, .24, .1, .5, attack=.9)
E(T_RAIN - .3, 'violins_trem', 'D5', 2.6, .22, .3, .45, attack=.6)
E(T_RAIN + 1.2, 'cellos', 'G2', 2.0, .32, -.3, .7, attack=.4)
E(T_RAIN + 1.2, 'violins_trem', 'B4', 1.8, .24, .3, .45, attack=.4)
for t in (23.4, 23.9, 24.4): E(t, 'timpani', 'F#2', .9, .24, 0, .55)
for t, p in [(21.2, 'F#5'), (22.4, 'D5'), (23.6, 'B4')]: E(t, 'vibraphone', p, 1.4, .18, .2, .4)
E(24.8, 'glass', 'B5', 1.0, .14, .1, .3)
# ---- bar 9: the flare (first sound after the silence), the gust, the cloud blown into stars
E(T_FLARE, 'glass', 'F#6', 2.2, .5, -.1, .8); E(T_FLARE, 'hand_chimes', 'A6', 2.0, .38, -.2, .6)
gl = ['D3', 'E3', 'F#3', 'G#3', 'A3', 'B3', 'D4', 'E4', 'F#4', 'G#4', 'A4', 'B4', 'D5', 'E5', 'F#5', 'A5', 'B5', 'D6']
for i, p in enumerate(gl): E(T_GUST + .15 + 1.0 * (i / len(gl)) ** 1.1, 'harp', p, .9, .32 + .02 * i, -.8 + 1.6 * i / len(gl), .8)
E(T_GUST - .6, 'sus_cymbal', 'cresc', None, .45, .2, .45)
for i, p in enumerate(['A6', 'G#6', 'F#6', 'E6', 'D6', 'C#6', 'B5', 'A5', 'F#5', 'E5']): E(T_GUST + 1.2 + .08 * i, 'glockenspiel', p, .7, .22, -.5 + .1 * i, .5)
for k in range(9): E(T_STARS + .12 * k + .05 * rng.random(), 'glockenspiel', ['D7', 'A6', 'E7', 'F#6', 'B6', 'C#7', 'A6', 'E6', 'D7'][k], .6, .12, -.8 + .2 * k, .35)
# ---- bars 10–11: the flower lifts, the flash, the meadow blooms
pad(T_LIFT, T_TITLE0 - T_LIFT + .4, ['D2', 'D3', 'A3', 'E4', 'G#4'], .28, .55, .9)
E(T_LIFT, 'harp', 'D2', 2.2, .45, -.2, .7); E(T_LIFT + .4, 'flute', 'D5', 1.2, .3, .15, .55, attack=.15)
for i, p in enumerate(['D4', 'F#4', 'A4', 'B4', 'C#5', 'D5', 'E5', 'F#5', 'G#5', 'A5', 'B5', 'D6']):
    E(T_FLASH + .04 * i, 'harp', p, 1.1, .36, -.8 + .14 * i, .8)
chord(T_FLASH, 'hand_chimes', ['D5', 'F#5', 'A5', 'C#6'], 2.4, .44, g=.72)
E(T_FLASH, 'timpani', 'D3', 1.4, .34, 0, .6)
for t, p, d in [(32.5, 'F#5', .8), (33.33, 'A5', .8), (34.17, 'B5', .8), (35.0, 'C#6', .8), (35.83, 'A5', 1.4)]:
    E(t, 'flute', p, d, .38, .15, .72, attack=.06)
for k in range(int((T_TITLE0 - T_FIELD) / B8)):
    E(T_FIELD + k * B8, 'harp', ['D3', 'A3', 'E4', 'G#4', 'A4', 'E4'][k % 6], .8, .26, -.4 + .8 * (k % 6) / 5, .55)
for k in range(16):                          # sparkles as the small flowers open
    E(T_FIELD + .3 + k * .26 + .08 * rng.random(), 'glockenspiel', ['D6', 'F#6', 'A6', 'E6', 'G#6', 'B6', 'C#7', 'D7'][k % 8], .6, .1 + .006 * k, -.9 + .12 * k, .35)
E(T_FIELD, 'violins', 'A4', 4.5, .22, .3, .45, attack=1.0)
# ---- bar 12–13: the title gathers and lands; let it ring out
gl2 = ['A3', 'B3', 'D4', 'E4', 'F#4', 'G#4', 'A4', 'B4', 'D5', 'E5', 'F#5', 'A5']
for i, p in enumerate(gl2): E(T_TITLE0 + .1 + .8 * (i / len(gl2)) ** 1.2, 'harp', p, .9, .28 + .02 * i, -.6 + 1.2 * i / len(gl2), .75)
E(T_TITLE - 1.3, 'sus_cymbal', 'cresc', None, .38, .1, .38)
chord(T_TITLE, 'harp', ['D2', 'A2', 'D3', 'F#3', 'A3', 'E4', 'F#4', 'A4'], 4.0, .52, step=.035)
chord(T_TITLE, 'hand_chimes', ['D5', 'F#5', 'A5', 'E6'], 3.5, .38, g=.65)
E(T_TITLE, 'glass', 'D6', 3.0, .38, .1, .6); E(T_TITLE, 'timpani', 'D3', 2.0, .3, 0, .6)
pad(T_TITLE, DUR - T_TITLE - .3, ['D2', 'D3', 'A3', 'F#4', 'E5'], .26, .55, .4)
E(T_TITLE + .05, 'flute', 'A5', 2.8, .32, .15, .62, attack=.12)
E(41.67, 'flute', 'F#5', 2.2, .28, .15, .55, attack=.15)
E(T_OUT - .67, 'harp', 'D3', 3.0, .38, -.1, .6); E(T_OUT - .67, 'harp', 'A3', 3.0, .3, .1, .5)
E(T_OUT - .2, 'glockenspiel', 'D7', 1.6, .16, .3, .4)

music = S.render(ev, dur=DUR + 3, master=False)[:N]
music = S.room(music, size=.65, mix=.26)

# ---- ambience: crickets (gone in the storm, back after the gust: L-cut under the end of the gust), wind, rain
amb = np.zeros((N, 2))
def cricket(f0, rate, pan, seed, on):
    r = np.random.default_rng(seed); x = np.zeros(N); t = r.uniform(0, 1 / rate)
    while t < DUR:
        for j in range(r.integers(3, 6)):
            i = int((t + j * .032) * SR); n = int(.018 * SR)
            if i + n >= N: break
            k = np.arange(n) / SR; x[i:i + n] += np.sin(2 * np.pi * f0 * k) * np.sin(np.pi * k / .018) ** 2
        t += 1 / rate * (.85 + .3 * r.random())
    return st(x * on, pan)
for k, (f0, rt, pn, sd) in enumerate([(4400, 1.3, -.7, 1), (4900, 1.1, .6, 2), (4650, 1.6, -.2, 3), (5100, .9, .35, 4), (4300, 1.45, .8, 5)]):
    on = env([(0, 0), (1.2 + .4 * k, 0), (2.4 + .4 * k, 1), (T_CLOUD + .5, 1), (T_RAIN, 0), (T_STARS + .8 + .6 * k, 0), (T_STARS + 1.6 + .6 * k, 1), (DUR, 1)])
    amb += cricket(f0, rt, pn, sd, on) * .009
wn = rng.standard_normal(N)
gusty = .6 + .4 * np.sin(2 * np.pi * .37 * tt) * np.sin(2 * np.pi * .13 * tt + 1)
wind = bp(wn, 160, 1300) * env([(0, 0), (T_CLOUD, 0), (T_RAIN, .25), (T_HUSH, .2), (T_FLARE, .08), (T_GUST, .4), (T_GUST + .7, 1.0), (T_STARS, .3), (T_LIFT, .02), (DUR, 0)]) * gusty
amb += np.stack([wind, np.roll(wind, 1301)], 1) * .05
rain_on = env([(0, 0), (T_RAIN - .35, 0), (T_RAIN + .8, .8), (T_FLARE, .8), (T_GUST + .2, .5), (T_GUST + 1.0, 0), (DUR, 0)])
rn = hp(rng.standard_normal(N), 2500) * .5 + bp(rng.standard_normal(N), 700, 2500) * .3
drops = np.zeros(N); idx = rng.integers(0, N - 400, 7000); amp = rng.random(7000) ** 3
for i, a in zip(idx, amp): drops[i:i + 200] += a * np.exp(-np.arange(200) / 30) * np.sin(np.arange(200) * (.6 + a))
rain = (rn + hp(drops, 1500) * .8) * rain_on
amb += np.stack([rain, np.roll(rain, 2203)], 1) * .04

# ---- foley: the air of the moving light
fx = np.zeros((N, 2))
def air(at, d, g, pan=0.0, f=4000): add(fx, lp(whoosh(d, 1.0), f), at, g, pan)
air(T_DESC + 2.0, 2.0, .12, .2, 2500)     # the light's long descent
air(T_TOUCH - .1, .6, .15, .0, 3000)      # the touch
air(T_OPEN - .1, 1.4, .18, .1, 3500)      # the petals swirl open
air(T_GUST - .05, 1.6, .55, -.3)          # the gust
air(T_GUST + .5, 1.2, .35, .5, 3000)      # the cloud streaks away
air(T_TITLE0, 1.2, .22, -.4)              # the fireflies swirl into the title

mix = music + amb + fx
mix = mix - .75 * np.stack([lp(mix[:, c], 150) for c in (0, 1)], 1)   # tame the low end (pads and timpani)
from scipy.signal import welch
f_, P = welch(mix.mean(1), SR, nperseg=8192)
print(f'20–120 Hz vs 120–8k (mean PSD): {10 * np.log10(P[(f_ > 20) & (f_ < 120)].mean() / P[(f_ > 120) & (f_ < 8000)].mean()):.1f} dB')
mix = np.stack([limit(compress(mix[:, c], thr=.4, ratio=2.0)) for c in (0, 1)], 1)
mix *= np.clip((DUR - tt) / .6, 0, 1)[:, None]
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
sf.write(os.path.join(HERE, 'out', 'mix.wav'), mix.astype(np.float32), SR)
used = sorted({e['inst'] for e in ev})
cr = S.credits(used)
open(os.path.join(HERE, 'out', 'instruments.txt'), 'w').write('\n'.join(cr) if isinstance(cr, list) else str(cr))
print('wrote out/mix.wav, events', len(ev), 'instruments', used)
