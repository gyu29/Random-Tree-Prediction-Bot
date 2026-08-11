(function () {
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- scroll reveals ----
  var revealEls = document.querySelectorAll(".reveal, .ticker-row, .wf-seg");
  if (reduceMotion) {
    revealEls.forEach(function (el) { el.classList.add("in"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.18 });
    revealEls.forEach(function (el) { io.observe(el); });
  }

  // ---- hero canvas: signal line crossing a decision threshold ----
  var canvas = document.getElementById("heroCanvas");
  var ctx = canvas.getContext("2d");
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var width, height;

  function resize() {
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resize);
  resize();

  var styles = getComputedStyle(document.documentElement);
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  var t = 0;
  var points = [];
  var maxPoints = 260;
  var threshold = 0.62; // 0..1 of canvas height, inverted (drawn near top-third)
  var pulses = [];
  var lastVal = 0.3;

  function seed() {
    points = [];
    lastVal = 0.35;
    for (var i = 0; i < maxPoints; i++) {
      lastVal = nextVal(lastVal);
      points.push(lastVal);
    }
  }
  function nextVal(prev) {
    var drift = (Math.sin(t * 0.013 + prev * 4) * 0.5) * 0.06;
    var noise = (Math.random() - 0.5) * 0.09;
    var pulled = (0.4 - prev) * 0.02; // mean reversion so it doesn't wander off
    var v = prev + drift + noise + pulled;
    return Math.max(0.04, Math.min(0.97, v));
  }

  seed();

  function draw() {
    t++;
    var next = nextVal(points[points.length - 1]);
    var crossedUp = points[points.length - 1] < threshold && next >= threshold;
    points.push(next);
    if (points.length > maxPoints) points.shift();

    if (crossedUp) {
      pulses.push({ x: width, y: height * (1 - threshold), life: 1 });
    }

    ctx.clearRect(0, 0, width, height);

    // faint grid
    ctx.strokeStyle = cssVar("--canvas-grid");
    ctx.lineWidth = 1;
    var gridStep = 48;
    for (var gx = width % gridStep; gx < width; gx += gridStep) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, height); ctx.stroke();
    }
    for (var gy = 0; gy < height; gy += gridStep) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(width, gy); ctx.stroke();
    }

    // threshold dashed line
    var thresholdY = height * (1 - threshold);
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = cssVar("--ember");
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(0, thresholdY); ctx.lineTo(width, thresholdY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // signal line
    var stepX = width / (maxPoints - 1);
    ctx.beginPath();
    for (var i = 0; i < points.length; i++) {
      var x = i * stepX;
      var y = height * (1 - points[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = cssVar("--signal");
    ctx.lineWidth = 2.2;
    ctx.lineJoin = "round";
    ctx.shadowColor = cssVar("--signal");
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // pulses at threshold crossings, drifting left with the line and fading
    pulses.forEach(function (p) {
      p.x -= stepX;
      p.life -= 0.012;
    });
    pulses = pulses.filter(function (p) { return p.life > 0 && p.x > -20; });
    pulses.forEach(function (p) {
      var r = 4 + (1 - p.life) * 22;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.strokeStyle = cssVar("--ember");
      ctx.globalAlpha = p.life * 0.5;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = cssVar("--ember");
      ctx.globalAlpha = Math.min(1, p.life * 1.6);
      ctx.fill();
      ctx.globalAlpha = 1;
    });

    if (!reduceMotion) requestAnimationFrame(draw);
  }

  if (reduceMotion) {
    // one static, representative frame -- no animation loop
    for (var i = 0; i < 40; i++) t++, points.push(nextVal(points[points.length - 1])), points.shift();
    pulses.push({ x: width * 0.62, y: height * (1 - threshold), life: 0.6 });
    draw();
  } else {
    requestAnimationFrame(draw);
  }
})();

// ---- terminal tab switcher (Home / Analyze / Backtest) ----
(function () {
  var buttons = document.querySelectorAll("#terminal button.item[data-panel]");
  buttons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var target = btn.getAttribute("data-panel");
      buttons.forEach(function (b) { b.classList.toggle("active", b === btn); });
      document.querySelectorAll(".terminal-panel").forEach(function (panel) {
        panel.classList.toggle("active", panel.getAttribute("data-panel-content") === target);
      });
    });
  });
})();
