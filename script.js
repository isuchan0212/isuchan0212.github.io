'use strict';

// Everything on the site reads fine without JavaScript.
// This file adds: day and night, the light pouring in through the window, and the CV print button.

const printButton = document.querySelector('[data-print-cv]');
if (printButton) {
  printButton.hidden = false;
  printButton.addEventListener('click', () => window.print());
}

// Day and night. The theme is set before first paint (inline script in <head>);
// this keeps it in step with the toggle, the browser bar colour, and the system setting.
const root = document.documentElement;
const toggle = document.querySelector('.theme-toggle');
const themeColor = document.querySelector('meta[name="theme-color"]');
const applyTheme = (theme) => {
  root.dataset.theme = theme;
  if (themeColor) themeColor.content = theme === 'dark' ? '#1c1a18' : '#e8e0d2';
  if (toggle) {
    const label = theme === 'dark' ? 'Switch to day' : 'Switch to night';
    toggle.setAttribute('aria-label', label);
    toggle.title = label;
  }
};
applyTheme(root.dataset.theme === 'dark' ? 'dark' : 'light');
toggle?.addEventListener('click', () => {
  const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  try { localStorage.setItem('theme', next); } catch (e) {}
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
  let saved = null;
  try { saved = localStorage.getItem('theme'); } catch (err) {}
  if (!saved) applyTheme(e.matches ? 'dark' : 'light');
});

// Shift the concrete panels so the page's axis runs between two columns of tie holes,
// never right beside a seam.
const axis = document.querySelector('.sun, .cv-head > div');
const alignWall = () => {
  if (!axis) return;
  const r = axis.getBoundingClientRect();
  const x = axis.classList.contains('sun') ? r.left + r.width / 2 : r.left;
  document.body.style.backgroundPosition = `${Math.round(x - 120)}px 0`;
};
alignWall();
addEventListener('resize', alignWall);

const labels = [...document.querySelectorAll('.row-label')];

// Light pouring in through the window: the window's shape swept down and to the right.
// lean: how far the light slants to the right for each unit it falls.
let lean = .42;
const pourLight = () => {
  const hero = document.querySelector('.hero');
  const glass = document.querySelector('.view');
  const shaft = document.querySelector('.shaft');
  const dust = document.querySelector('.shaft-motes');
  const h = hero.getBoundingClientRect();
  const g = glass.getBoundingClientRect();
  const l = g.left - h.left, t = g.top - h.top, r = l + g.width, b = t + g.height;
  const fall = (h.height - t) * .95;
  const vx = fall * lean, vy = fall;
  const pt = (x, y) => `${x.toFixed(1)},${y.toFixed(1)}`;
  shaft.setAttribute('viewBox', `0 0 ${h.width} ${h.height}`);
  shaft.querySelector('.shaft-body').setAttribute('d',
    `M${pt(l, t)} L${pt(r, t)} L${pt(r + vx, t + vy)} L${pt(r + vx, b + vy)} L${pt(l + vx, b + vy)} L${pt(l, b)}Z`);
  // A few brighter strands within the light, starting from the bottom of the glass.
  const strands = [[.06, .14], [.3, .35], [.52, .63], [.8, .86]];
  shaft.querySelectorAll('.shaft-rays path').forEach((path, i) => {
    const [a, c] = strands[i];
    const x1 = l + g.width * a, x2 = l + g.width * c;
    path.setAttribute('d', `M${pt(x1, b)} L${pt(x2, b)} L${pt(x2 + vx * .8, b + vy * .8)} L${pt(x1 + vx * .8, b + vy * .8)}Z`);
  });
  const fade = shaft.querySelector('#shaft-fade');
  const cx = (l + r) / 2, cy = (t + b) / 2;
  fade.setAttribute('x1', cx); fade.setAttribute('y1', cy);
  fade.setAttribute('x2', cx + vx * .85); fade.setAttribute('y2', cy + vy * .85);
  const strand = shaft.querySelector('#shaft-strand');
  strand.setAttribute('x1', cx); strand.setAttribute('y1', b);
  strand.setAttribute('x2', cx + vx * .8); strand.setAttribute('y2', b + vy * .8);
  // Dust floating in the light.
  if (!dust.children.length) for (let i = 0; i < 10; i++) dust.append(document.createElement('i'));
  [...dust.children].forEach((mote, i) => {
    const u = ((i * 37) % 10) / 10, s2 = .08 + ((i * 53) % 10) / 16;
    mote.style.left = `${l + g.width * u + vx * s2}px`;
    mote.style.top = `${b + vy * s2}px`;
    mote.style.animationDelay = `${-i * 1.7}s`;
  });
};

if (document.querySelector('.shaft')) {
  // Section labels warm as they come up the page.
  const look = () => {
    labels.forEach((el) => el.classList.toggle('lit', el.getBoundingClientRect().top < innerHeight * .6));
  };
  pourLight();
  addEventListener('resize', pourLight);

  // The light leans a little with the pointer, and fades as the window scrolls away.
  const shaft = document.querySelector('.shaft');
  const hero = document.querySelector('.hero');
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let goal = lean, frame = 0;
  const settle = () => {
    lean += (goal - lean) * .08;
    pourLight();
    frame = Math.abs(goal - lean) > .001 ? requestAnimationFrame(settle) : 0;
  };
  if (!still && matchMedia('(hover: hover)').matches) {
    addEventListener('pointermove', (e) => {
      goal = .42 + (e.clientX / innerWidth - .5) * .22;
      if (!frame) frame = requestAnimationFrame(settle);
    }, { passive: true });
  }
  const fade = () => shaft.style.setProperty('--fade', Math.max(0, 1 - scrollY / (hero.offsetHeight * .9)).toFixed(3));
  addEventListener('scroll', fade, { passive: true });
  fade();
  addEventListener('load', pourLight);
  document.fonts?.ready.then(pourLight);
  addEventListener('scroll', look, { passive: true });
  look();
}
