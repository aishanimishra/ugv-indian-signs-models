import socket, json, threading, base64
import cv2, numpy as np, pygame

PI_IP   = "172.20.10.7"
PI_PORT = 9000

# ─── Shared state ─────────────────────────────────────
latest = {"frame": 0, "fps": 0.0, "detections": [], "img": None}
lock   = threading.Lock()

# ─── Color map ────────────────────────────────────────
def get_color(label):
    if any(x in label for x in ["STOP", "NO_ENTRY", "PROHIBITED"]):
        return (255, 50,  50)
    if "SPEED_LIMIT" in label:
        return (255, 165,  0)
    if "COMPULSARY" in label:
        return (50,  200, 50)
    if any(x in label for x in ["DIP", "CURVE", "BEND",
                                  "GRAVEL", "SLIPPERY", "HUMP"]):
        return (255, 255,  0)
    return (100, 200, 255)

# ─── Receiver thread ──────────────────────────────────
def recv_thread():
    buf = ""
    while True:
        try:
            data = conn.recv(65536).decode()
            if not data:
                break
            buf += data
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                packet = json.loads(line)
                with lock:
                    latest.update(packet)
        except Exception as e:
            print(f"Recv error: {e}")
            break

# ─── Connect ──────────────────────────────────────────
print(f"Connecting to {PI_IP}:{PI_PORT}...")
conn = socket.create_connection((PI_IP, PI_PORT))
print("Connected.\n")
threading.Thread(target=recv_thread, daemon=True).start()

# ─── Pygame init ──────────────────────────────────────
pygame.init()

CANVAS_W, CANVAS_H = 640, 480
PANEL_W             = 220
WIN_W               = CANVAS_W + PANEL_W
WIN_H               = CANVAS_H + 30

screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("UGV — Indian Road Sign Detection")

font_large = pygame.font.SysFont("consolas", 18, bold=True)
font_small = pygame.font.SysFont("consolas", 14)
font_tiny  = pygame.font.SysFont("consolas", 12)
clock      = pygame.time.Clock()

history    = []
PANEL_X    = CANVAS_W + 10

def draw_rounded_rect(surface, color, rect, radius=6, alpha=200):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha),
                     (0, 0, rect[2], rect[3]), border_radius=radius)
    surface.blit(s, (rect[0], rect[1]))

print("GUI running. Q to quit.")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    # ── Get latest data ────────────────────────────────
    with lock:
        dets      = latest["detections"].copy()
        fps       = latest["fps"]
        frame_num = latest["frame"]
        img_b64   = latest["img"]

    # ── Background ────────────────────────────────────
    screen.fill((15, 15, 15))

    # ── Decode and draw camera frame ──────────────────
    canvas = pygame.Surface((CANVAS_W, CANVAS_H))
    canvas.fill((30, 30, 30))

    if img_b64:
        try:
            jpg_bytes = base64.b64decode(img_b64)
            np_arr    = np.frombuffer(jpg_bytes, dtype=np.uint8)
            cv_img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_img    = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            cv_img    = cv2.resize(cv_img, (CANVAS_W, CANVAS_H))
            surf      = pygame.surfarray.make_surface(
                            np.transpose(cv_img, (1, 0, 2)))
            canvas.blit(surf, (0, 0))
        except Exception as e:
            print(f"Frame decode error: {e}")

    # ── Draw bounding boxes over camera frame ─────────
    for d in dets:
        label        = d["label"]
        conf         = d["conf"]
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        x1 = max(0, min(x1, CANVAS_W - 1))
        y1 = max(0, min(y1, CANVAS_H - 1))
        x2 = max(0, min(x2, CANVAS_W - 1))
        y2 = max(0, min(y2, CANVAS_H - 1))
        color = get_color(label)

        pygame.draw.rect(canvas, color,
                         (x1, y1, x2-x1, y2-y1), 2,
                         border_radius=4)

        txt     = f"{label}  {conf:.0%}"
        txt_s   = font_tiny.render(txt, True, (255, 255, 255))
        txt_w   = txt_s.get_width() + 8
        label_y = max(0, y1 - 18)
        draw_rounded_rect(canvas, color,
                          (x1, label_y, txt_w, 18),
                          radius=4, alpha=220)
        canvas.blit(txt_s, (x1 + 4, label_y + 2))

    screen.blit(canvas, (0, 0))

    # ── Right panel ────────────────────────────────────
    py = 10

    title = font_large.render("UGV DETECTION", True, (200, 200, 200))
    screen.blit(title, (PANEL_X, py))
    py += 30

    fps_color = (50,255,50) if fps>=6 else (255,165,0) if fps>=3 else (255,50,50)
    screen.blit(font_large.render(f"FPS:  {fps:.1f}", True, fps_color),
                (PANEL_X, py))
    py += 26

    screen.blit(font_small.render(f"Frame: {frame_num}", True, (100,100,100)),
                (PANEL_X, py))
    py += 28

    pygame.draw.line(screen, (60,60,60), (PANEL_X, py), (WIN_W-5, py))
    py += 10

    screen.blit(font_small.render("NOW DETECTING:", True, (150,150,150)),
                (PANEL_X, py))
    py += 20

    if dets:
        for d in dets:
            color = get_color(d["label"])
            pygame.draw.rect(screen, (50,50,50),
                             (PANEL_X, py, 200, 22), border_radius=4)
            bar_w = int(200 * d["conf"])
            pygame.draw.rect(screen, color,
                             (PANEL_X, py, bar_w, 22), border_radius=4)
            screen.blit(
                font_tiny.render(
                    f"{d['label'][:20]}  {d['conf']:.0%}",
                    True, (255,255,255)),
                (PANEL_X + 4, py + 4))
            py += 26
            if py > WIN_H - 120:
                break
    else:
        screen.blit(
            font_small.render("No signs detected", True, (80,80,80)),
            (PANEL_X, py))
        py += 24

    # Update history
    if dets:
        for d in dets:
            if d["label"] not in history:
                history.insert(0, d["label"])
        history[:] = history[:8]

    # History panel
    py = WIN_H - 115
    pygame.draw.line(screen, (60,60,60), (PANEL_X, py), (WIN_W-5, py))
    py += 10
    screen.blit(font_small.render("HISTORY:", True, (150,150,150)),
                (PANEL_X, py))
    py += 18
    for i, h in enumerate(history[:4]):
        a = max(60, 255 - i*55)
        screen.blit(font_tiny.render(f"• {h}", True, (a,a,a)),
                    (PANEL_X, py))
        py += 16

    # Status bar
    pygame.draw.rect(screen, (25,25,25), (0, WIN_H-24, WIN_W, 24))
    screen.blit(
        font_tiny.render(
            f"  Pi: {PI_IP}:{PI_PORT}  |  YOLO26n  |  Q to quit",
            True, (80,80,80)),
        (0, WIN_H - 18))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
conn.close()
print("Done.")
