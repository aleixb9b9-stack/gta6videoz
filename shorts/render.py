"""Genera los Shorts verticales (1080x1920, 30 fps) del pack de lanzamiento de GTA 6.

Uso:  python3 shorts/render.py            -> renderiza todos
      python3 shorts/render.py datos      -> solo uno

Todo se genera aquí (imagen y música), sin material de terceros.
"""
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.signal import lfilter

W, H, FPS = 1080, 1920, 30
BPM = 120
BEAT = 60 / BPM
SR = 44100
HORIZON = 1180
Y_SHIFT = 130  # baja todo el texto hacia el centro útil del Short
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

FONT = "/usr/share/fonts/opentype/inter/InterDisplay-BlackItalic.otf"
FONT_SUB = "/usr/share/fonts/opentype/inter/InterDisplay-ExtraBoldItalic.otf"
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

WHITE = (255, 255, 255)
YELLOW = (255, 230, 0)
PINK = (255, 46, 136)
CYAN = (0, 229, 255)
RED = (255, 40, 60)
DARK = (25, 5, 45)


# ----------------------------------------------------------------- fondo ---

def lerp(a, b, t):
    return a + (b - a) * t


def sky():
    y = np.linspace(0, 1, HORIZON)[:, None]
    stops = [(0.0, (12, 2, 30)), (0.45, (60, 8, 90)), (0.75, (200, 30, 120)), (1.0, (255, 140, 60))]
    img = np.zeros((HORIZON, W, 3), np.float32)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        m = (y >= p0) & (y <= p1)
        t = np.clip((y - p0) / (p1 - p0), 0, 1)
        for ch in range(3):
            img[..., ch] = np.where(m, lerp(c0[ch], c1[ch], t), img[..., ch])
    rng = np.random.default_rng(7)
    for _ in range(140):  # estrellas
        sx, sy = rng.integers(0, W), rng.integers(0, int(HORIZON * 0.45))
        b = rng.uniform(120, 255)
        img[sy:sy + 3, sx:sx + 3] = np.maximum(img[sy:sy + 3, sx:sx + 3], b)
    return img


def sun_layer(phase):
    """Sol synthwave con franjas que suben (phase 0..1, periódico)."""
    r = 360
    cx, cy = W // 2, HORIZON - 120
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = d <= r
    t = np.clip((yy - (cy - r)) / (2 * r), 0, 1)
    col = np.stack([255 + 0 * t, lerp(235, 40, t), lerp(90, 140, t)], -1)
    # franjas en la mitad inferior
    rel = (yy - cy) / r
    band = 26.0
    gaps = ((yy + phase * band) % band) < np.clip(rel * band * 0.9, 0, band)
    alpha = inside & ~((rel > 0) & gaps)
    alpha &= yy < HORIZON
    return col, alpha.astype(np.float32)


def grid_layer(phase):
    layer = Image.new("RGB", (W, H - HORIZON), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    fh = H - HORIZON
    for i in range(0, 24):
        z = (i + 1 - phase) * 0.4
        if z <= 0.05:
            continue
        y = fh * (0.2 / z)
        if y > fh:
            continue
        d.line([(0, y), (W, y)], fill=PINK, width=max(2, int(6 / z)))
    for k in range(-9, 10):
        d.line([(W / 2 + k * 45, 0), (W / 2 + k * 380, fh)], fill=PINK, width=3)
    glow = layer.filter(ImageFilter.GaussianBlur(10))
    arr = np.asarray(layer, np.float32) + np.asarray(glow, np.float32) * 1.6
    floor = np.zeros((fh, W, 3), np.float32)
    g = np.linspace(1, 0, fh)[:, None, None]
    floor += np.array([60, 6, 70], np.float32) * (0.4 + 0.6 * (1 - g))
    return np.clip(floor + arr, 0, 255)


def build_backgrounds(n):
    base = sky()
    frames = []
    for i in range(n):
        ph = i / n
        img = np.zeros((H, W, 3), np.float32)
        img[:HORIZON] = base
        col, a = sun_layer(ph)
        img = img * (1 - a[..., None]) + col * a[..., None]
        # halo del sol
        halo = Image.fromarray((a * 255).astype(np.uint8)).resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(25)).resize((W, H))
        h = np.asarray(halo, np.float32)[..., None] / 255
        img += h * np.array([255, 60, 120]) * 0.55
        img[HORIZON:] = grid_layer(ph)
        img[HORIZON - 3:HORIZON + 3] = (255, 120, 200)
        # viñeta
        yy, xx = np.mgrid[0:H, 0:W]
        v = 1 - 0.55 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2.2
        img *= np.clip(v, 0.35, 1)[..., None]
        frames.append(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)))
    return frames


# ----------------------------------------------------------------- texto ---

def text_img(txt, size, fill=WHITE, glow=PINK, font=FONT, max_w=980, stroke=10):
    while True:
        f = ImageFont.truetype(font, size)
        bbox = f.getbbox(txt, stroke_width=stroke)
        if bbox[2] - bbox[0] <= max_w or size < 30:
            break
        size -= 4
    pad = 60
    w, h = bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    pos = (pad - bbox[0], pad - bbox[1])
    if glow:
        g = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(g).text(pos, txt, font=f, fill=glow + (255,), stroke_width=stroke + 6, stroke_fill=glow + (255,))
        g = g.filter(ImageFilter.GaussianBlur(18))
        base = Image.alpha_composite(base, g)
        base = Image.alpha_composite(base, g)
        d = ImageDraw.Draw(base)
    d.text(pos, txt, font=f, fill=fill + (255,), stroke_width=stroke, stroke_fill=DARK + (255,))
    return base


def emoji_img(ch, size):
    f = ImageFont.truetype(EMOJI, 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox())
    return im.resize((size, int(size * im.height / im.width)), Image.LANCZOS)


def cross_img(w, h, color=RED):
    pad = 50
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for a, b in [((pad, pad), (pad + w, pad + h)), ((pad + w, pad), (pad, pad + h))]:
        d.line([a, b], fill=color + (255,), width=34)
    g = im.filter(ImageFilter.GaussianBlur(14))
    return Image.alpha_composite(Image.alpha_composite(g, g), im)


def pill_img(txt, bg=YELLOW, fg=DARK, size=56):
    f = ImageFont.truetype(FONT, size)
    b = f.getbbox(txt)
    w, h = b[2] - b[0] + 70, b[3] - b[1] + 40
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=bg + (255,))
    d.text((35 - b[0], 20 - b[1]), txt, font=f, fill=fg + (255,))
    return im


# ------------------------------------------------------------- animación ---

def ease_out_back(p):
    c1, c3 = 1.9, 2.9
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


def ease_out_cubic(p):
    return 1 - (1 - p) ** 3


class El:
    def __init__(self, img, cx, cy, t_in, t_out, kind="pop", dur=0.35, pulse=False):
        self.img, self.cx, self.cy = img, cx, cy + Y_SHIFT
        self.t_in, self.t_out, self.kind, self.dur, self.pulse = t_in, t_out, kind, dur, pulse

    def draw(self, frame, t):
        if not (self.t_in <= t < self.t_out):
            return
        p = min(1, (t - self.t_in) / self.dur)
        scale, dx, dy, alpha = 1.0, 0, 0, min(1, p * 3)
        if self.kind == "pop":
            scale = max(0.01, ease_out_back(p))
        elif self.kind == "slam":
            scale = lerp(2.6, 1, ease_out_cubic(p))
        elif self.kind == "left":
            dx = -W * (1 - ease_out_cubic(p))
        elif self.kind == "right":
            dx = W * (1 - ease_out_cubic(p))
        elif self.kind == "up":
            dy = 200 * (1 - ease_out_cubic(p))
        elif self.kind == "draw":  # la X/tachado aparece creciendo
            scale = lerp(0.3, 1, ease_out_back(p))
        if self.pulse:
            beat_p = ((t - self.t_in) % BEAT) / BEAT
            scale *= 1 + 0.045 * math.exp(-beat_p * 6)
        out_left = self.t_out - t
        if out_left < 0.12:
            q = out_left / 0.12
            alpha *= q
            scale *= lerp(1.15, 1, q)
        img = self.img
        if abs(scale - 1) > 0.002:
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
        if alpha < 0.999:
            a = img.getchannel("A").point(lambda v: int(v * alpha))
            img = img.copy()
            img.putalpha(a)
        frame.alpha_composite(img, (int(self.cx + dx - img.width / 2), int(self.cy + dy - img.height / 2)))


def render_video(name, duration, elements, flashes, shakes, audio):
    os.makedirs(OUT, exist_ok=True)
    period = int(BEAT * FPS)  # el fondo hace loop en cada pulso
    bgs = build_backgrounds(period)
    wav = os.path.join(OUT, f"{name}.wav")
    write_wav(wav, audio)
    mp4 = os.path.join(OUT, f"{name}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-shortest", mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(duration * FPS)
    rng = np.random.default_rng(1)
    for i in range(n):
        t = i / FPS
        frame = bgs[i % period].convert("RGBA")
        for e in elements:
            e.draw(frame, t)
        img = frame.convert("RGB")
        # temblor de cámara
        for st in shakes:
            if 0 <= t - st < 0.3:
                k = (1 - (t - st) / 0.3) * 28
                ox, oy = rng.integers(-k, k + 1), rng.integers(-k, k + 1)
                img = img.transform(img.size, Image.AFFINE, (1, 0, ox, 0, 1, oy))
        # destellos en los cambios de escena
        for ft in flashes:
            if 0 <= t - ft < 0.25:
                a = 0.75 * (1 - (t - ft) / 0.25)
                img = Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), a)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    os.remove(wav)
    print("OK", mp4)


# ----------------------------------------------------------------- audio ---

def write_wav(path, x):
    x = np.tanh(x * 1.2)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.8
    data = (x * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def lowpass(x, cutoff):
    """Filtro paso bajo de un polo."""
    a = math.exp(-2 * math.pi * cutoff / SR)
    return lfilter([1 - a], [1, -a], x)


def saw(freq, n, detune=0.0):
    t = np.arange(n) / SR
    f = freq * (1 + detune)
    return 2 * ((t * f) % 1) - 1


def music(duration, impacts=(), risers=(), drop_until=0.0, final_hit=None):
    n = int(duration * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    rng = np.random.default_rng(3)

    def add(sig, t0, gain=1.0, pan=0.0):
        s = int(t0 * SR)
        if s >= n:
            return
        e = min(n, s + len(sig))
        L[s:e] += sig[: e - s] * gain * (1 - max(0, pan))
        R[s:e] += sig[: e - s] * gain * (1 + min(0, pan))

    # sonidos base
    kn = int(0.4 * SR)
    kt = np.arange(kn) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 110 * np.exp(-kt * 30)) / SR) * np.exp(-kt * 9)
    sn = int(0.22 * SR)
    snare = rng.standard_normal(sn) * np.exp(-np.arange(sn) / SR * 18)
    snare = snare - lowpass(snare, 900) * 0.8
    snare += np.sin(2 * np.pi * 190 * np.arange(sn) / SR) * np.exp(-np.arange(sn) / SR * 30) * 0.6
    hn = int(0.05 * SR)
    hat = np.diff(rng.standard_normal(hn + 1)) * np.exp(-np.arange(hn) / SR * 80)

    # progresión La menor - Fa - Do - Sol (un acorde por compás)
    chords = [(55.0, [220.0, 261.63, 329.63]), (43.65, [174.61, 220.0, 261.63]),
              (65.41, [196.0, 261.63, 329.63]), (49.0, [196.0, 246.94, 293.66])]
    bar = BEAT * 4
    nbars = int(math.ceil(duration / bar))
    for b in range(nbars):
        root, triad = chords[b % 4]
        t0 = b * bar
        # pad
        pn = int(bar * SR)
        env = np.minimum(1, np.arange(pn) / (0.3 * SR)) * np.minimum(1, (pn - np.arange(pn)) / (0.2 * SR))
        pad = sum(saw(f, pn, d) for f in triad for d in (-0.004, 0.004)) / 6
        add(lowpass(pad, 1400) * env, t0, 0.18, -0.3)
        add(lowpass(saw(triad[2] * 2, pn, 0.002) / 2, 1800) * env, t0, 0.07, 0.4)
        # bajo en corcheas
        for k in range(8):
            ln = int(BEAT / 2 * SR)
            f = root * (2 if k % 2 else 1)
            note = saw(f, ln) * np.exp(-np.arange(ln) / SR * 6)
            add(lowpass(note, 600), t0 + k * BEAT / 2, 0.45)
        for k in range(4):
            tb = t0 + k * BEAT
            if tb < drop_until:
                continue
            add(kick, tb, 0.95)
            if k in (1, 3):
                add(snare, tb, 0.35)
            add(hat, tb + BEAT / 2, 0.18, 0.35)
            add(hat, tb, 0.08, -0.35)

    # efectos
    for ti in impacts:
        m = int(1.2 * SR)
        tt = np.arange(m) / SR
        boom = np.sin(2 * np.pi * np.cumsum(30 + 80 * np.exp(-tt * 12)) / SR) * np.exp(-tt * 3.5)
        crash = rng.standard_normal(m) * np.exp(-tt * 4)
        crash = crash - lowpass(crash, 3000)
        add(boom, ti, 0.9)
        add(crash, ti, 0.22)
    for ti in risers:  # whoosh de 0.5 s que termina en ti
        m = int(0.5 * SR)
        noise = rng.standard_normal(m)
        env = (np.arange(m) / m) ** 2
        add((noise - lowpass(noise, 1500)) * env, ti - 0.5, 0.3, 0.2)
    if final_hit is not None:
        m = int(2.5 * SR)
        tt = np.arange(m) / SR
        chord = sum(saw(f, m, d) for f in (110, 220, 261.63, 329.63, 440) for d in (-0.005, 0.005)) / 10
        add(lowpass(chord, 2500) * np.exp(-tt * 1.2), final_hit, 0.5)
    return np.stack([L, R], 1)


# ----------------------------------------------------------------- shorts ---

def short_datos():
    """'GTA 6: 3 cosas que DEBES saber' — 24 s, cambios de escena a compás."""
    D = 24.0
    cx = W // 2
    E = []
    # Escena 0 — gancho (0-4)
    E += [El(text_img("GTA 6", 270, YELLOW), cx, 470, 0.0, 4.0, "slam", 0.25, pulse=True),
          El(text_img("SALE EL JUEVES", 110), cx, 700, 0.5, 4.0, "left"),
          El(text_img("3 COSAS QUE", 96, glow=CYAN), cx, 880, 1.5, 4.0, "pop"),
          El(text_img("DEBES SABER", 120, YELLOW), cx, 1010, 2.0, 4.0, "slam", 0.25)]
    # Escena 1 — 00:00 (4-10)
    E += [El(pill_img("1 / 3"), cx, 260, 4.0, 10.0, "pop"),
          El(text_img("00:00", 330, glow=CYAN), cx, 520, 4.0, 10.0, "slam", 0.25, pulse=True),
          El(text_img("SE DESBLOQUEA A", 84), cx, 760, 4.7, 10.0, "up"),
          El(text_img("MEDIANOCHE", 128, YELLOW), cx, 880, 5.0, 10.0, "pop"),
          El(text_img("HORA LOCAL DE TU PAÍS", 64, font=FONT_SUB, glow=CYAN, stroke=8), cx, 1010, 6.0, 10.0, "up")]
    # Escena 2 — PC (10-16)
    E += [El(pill_img("2 / 3"), cx, 260, 10.0, 16.0, "pop"),
          El(text_img("PC", 380), cx, 540, 10.0, 16.0, "slam", 0.25),
          El(cross_img(360, 300), cx, 540, 10.6, 16.0, "draw", 0.3),
          El(text_img("NO SALE EN PC", 120, RED, glow=RED), cx, 800, 11.0, 16.0, "pop"),
          El(text_img("SOLO PS5 Y XBOX SERIES X|S", 62, font=FONT_SUB, glow=CYAN, stroke=8), cx, 940, 12.0, 16.0, "up")]
    # Escena 3 — Online (16-21)
    E += [El(pill_img("3 / 3"), cx, 260, 16.0, 21.0, "pop"),
          El(text_img("ONLINE", 230), cx, 520, 16.0, 21.0, "slam", 0.25),
          El(cross_img(700, 200), cx, 520, 16.6, 21.0, "draw", 0.3),
          El(text_img("NO ESTÁ DE SALIDA", 104, RED, glow=RED), cx, 760, 17.0, 21.0, "pop"),
          El(text_img("SE ESPERA PARA 2027", 80, YELLOW, glow=CYAN), cx, 900, 18.0, 21.0, "up")]
    # Escena 4 — CTA (21-24)
    E += [El(text_img("¿LO JUEGAS", 140), cx, 480, 21.0, D + 1, "slam", 0.25),
          El(text_img("A LAS 00:00?", 140, YELLOW), cx, 640, 21.3, D + 1, "slam", 0.25),
          El(text_img("SÍ O NO EN COMENTARIOS", 66, font=FONT_SUB, glow=CYAN, stroke=8), cx, 820, 22.0, D + 1, "up"),
          El(emoji_img("👇", 170), cx, 1000, 22.4, D + 1, "pop", pulse=True)]
    cuts = [4.0, 10.0, 16.0, 21.0]
    audio = music(D, impacts=[0.0] + cuts, risers=cuts)
    render_video("short-1-3-datos", D, E, flashes=[0.0] + cuts, shakes=[0.0, 10.6, 16.6, 21.0], audio=audio)


def short_cuenta_atras():
    """Cuenta atrás 10→1 y revelación — 16 s, pensado para verse en bucle."""
    D = 16.0
    cx = W // 2
    E = [El(text_img("¿PREPARADO?", 150, YELLOW), cx, 520, 0.0, 2.0, "slam", 0.25, pulse=True),
         El(text_img("GTA 6 EN…", 110, glow=CYAN), cx, 720, 0.6, 2.0, "pop")]
    start = 2.0
    for k in range(10):
        n = 10 - k
        col = RED if n <= 3 else WHITE
        E.append(El(text_img(str(n), 560, col, glow=RED if n <= 3 else PINK), cx, 620, start + k, start + k + 1, "slam", 0.18))
    E.append(El(text_img("GTA 6 EN…", 80, glow=CYAN), cx, 230, start, start + 10, "pop"))
    rev = start + 10  # 12.0
    E += [El(text_img("YA ESTÁ", 200), cx, 430, rev, D + 1, "slam", 0.22),
          El(text_img("AQUÍ", 300, YELLOW), cx, 680, rev + 0.25, D + 1, "slam", 0.22, pulse=True),
          El(text_img("19 · 11 · 2026", 110, glow=CYAN), cx, 900, rev + 0.8, D + 1, "up"),
          El(text_img("PS5 · XBOX SERIES X|S", 66, font=FONT_SUB, glow=CYAN, stroke=8), cx, 1030, rev + 1.3, D + 1, "up"),
          El(text_img("SÍGUEME PARA EL GAMEPLAY", 60, font=FONT_SUB, glow=PINK, stroke=8), cx, 1150, rev + 2.0, D + 1, "pop")]
    ticks = [start + k for k in range(10)]
    audio = music(D, impacts=[rev], risers=[2.0, rev], drop_until=2.0, final_hit=rev)
    render_video("short-2-cuenta-atras", D, E, flashes=[2.0, rev] + ticks[7:], shakes=[rev] + ticks[7:], audio=audio)


SHORTS = {"datos": short_datos, "cuenta": short_cuenta_atras}

if __name__ == "__main__":
    which = sys.argv[1:] or list(SHORTS)
    for w in which:
        SHORTS[w]()
