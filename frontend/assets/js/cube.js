// Hero: a 3D concrete test cube (CSS 3D) that follows the pointer, orbited by drifting aggregate.
import { reduced, fine } from "./fx.js";

export function heroCube(stage) {
  const cube = stage.querySelector(".cube");
  let rx = -22, ry = 35, trx = -22, try_ = 35, spin = 0, scrollV = 0, lastY = scrollY;
  if (fine && !reduced) {
    stage.addEventListener("pointermove", (e) => {
      const r = stage.getBoundingClientRect();
      trx = -22 - ((e.clientY - r.top) / r.height - 0.5) * 30;
      try_ = 35 + ((e.clientX - r.left) / r.width - 0.5) * 50;
    });
    stage.addEventListener("pointerleave", () => { trx = -22; try_ = 35; });
  }
  addEventListener("scroll", () => { scrollV += (scrollY - lastY) * 0.05; lastY = scrollY; }, { passive: true });
  let visible = true;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(stage);
  const t0 = performance.now();
  const loop = (now) => {
    requestAnimationFrame(loop);
    if (!visible) return;
    const t = (now - t0) / 1000;
    scrollV *= 0.9;
    spin += reduced ? 0 : 0.12 + Math.abs(scrollV);
    rx += (trx - rx) * 0.06;
    ry += (try_ - ry) * 0.06;
    const bob = reduced ? 0 : Math.sin(t * 1.2) * 10;
    cube.style.transform = `translateY(${bob}px) rotateX(${rx}deg) rotateY(${ry + spin}deg)`;
  };
  requestAnimationFrame(loop);
  gravel(stage.querySelector("#gravel"));
}

// Drifting pieces of aggregate: irregular polygons with depth, parallax and slow tumble.
function gravel(canvas) {
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(devicePixelRatio || 1, 2);
  let W = 0, H = 0, mx = 0, my = 0;
  const resize = () => {
    const r = canvas.getBoundingClientRect();
    W = r.width; H = r.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  new ResizeObserver(resize).observe(canvas);
  resize();
  addEventListener("pointermove", (e) => { mx = e.clientX / innerWidth - 0.5; my = e.clientY / innerHeight - 0.5; }, { passive: true });
  let seed = 11;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const stones = Array.from({ length: 46 }, () => {
    const n = 5 + Math.floor(rnd() * 4);
    const pts = Array.from({ length: n }, (_, i) => {
      const a = (i / n) * Math.PI * 2 + rnd() * 0.5;
      const r = 0.6 + rnd() * 0.4;
      return [Math.cos(a) * r, Math.sin(a) * r];
    });
    const z = rnd();
    const orange = rnd() < 0.12;
    return { x: rnd(), y: rnd(), z, size: 3 + z * 11, rot: rnd() * 6.28, vr: (rnd() - 0.5) * 0.01, vy: 0.00008 + z * 0.00025, pts,
      shade: 70 + Math.floor(rnd() * 70), orange };
  });
  const loop = () => {
    requestAnimationFrame(loop);
    ctx.clearRect(0, 0, W, H);
    for (const s of stones) {
      if (!reduced) { s.y -= s.vy; s.rot += s.vr; if (s.y < -0.05) { s.y = 1.05; s.x = rnd(); } }
      const px = s.x * W + mx * 40 * s.z, py = s.y * H + my * 40 * s.z;
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate(s.rot);
      ctx.beginPath();
      s.pts.forEach(([a, b], i) => (i ? ctx.lineTo(a * s.size, b * s.size) : ctx.moveTo(a * s.size, b * s.size)));
      ctx.closePath();
      const alpha = 0.25 + s.z * 0.6;
      ctx.fillStyle = s.orange ? `rgba(255,90,31,${alpha})` : `rgba(${s.shade + 40},${s.shade + 34},${s.shade + 24},${alpha})`;
      ctx.fill();
      ctx.restore();
    }
  };
  loop();
}
