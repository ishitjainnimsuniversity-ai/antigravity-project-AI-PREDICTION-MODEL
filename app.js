/* ═══════════════════════════════════════════════════════════
   ADVANCED AI HEALTHCARE — APPLICATION ENGINE
   ═══════════════════════════════════════════════════════════ */

const API = '';  // Same origin
let videoStream = null;
let patientName = 'Guest';
let patientAge = 25;

// ─── PARTICLE BACKGROUND ───────────────────────────────────
(function initParticles() {
    const canvas = document.getElementById('particleCanvas');
    const ctx = canvas.getContext('2d');
    let particles = [];

    function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < 60; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            r: Math.random() * 1.5 + 0.5,
            o: Math.random() * 0.3 + 0.05
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 210, 255, ${p.o})`;
            ctx.fill();
        });
        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 210, 255, ${0.04 * (1 - dist / 150)})`;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
})();

// ─── SECTION NAVIGATION ────────────────────────────────────
function showSection(id) {
    document.querySelectorAll('main > section').forEach(s => {
        s.classList.add('hidden-section');
        s.classList.remove('active-section');
    });
    const el = document.getElementById(id);
    el.classList.remove('hidden-section');
    el.classList.add('active-section');
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── STEP 1: PATIENT REGISTRATION ──────────────────────────
function proceedToScan() {
    patientName = document.getElementById('patientName').value.trim() || 'Guest';
    patientAge = parseInt(document.getElementById('patientAge').value) || 25;
    showSection('scanSection');
}

// ─── STEP 2: CAMERA ────────────────────────────────────────
async function startCamera() {
    showSection('cameraSection');
    const video = document.getElementById('videoFeed');
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
        });
        video.srcObject = videoStream;
        document.getElementById('cameraStatus').textContent = 'Camera active — position your face in frame';

        // Start HUD overlay updates
        startHUDLoop();
    } catch (err) {
        document.getElementById('cameraStatus').textContent = 'Camera access denied. Please allow camera permissions.';
        console.error('Camera error:', err);
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(t => t.stop());
        videoStream = null;
    }
    showSection('scanSection');
}

let hudInterval = null;
function startHUDLoop() {
    let frame = 0;
    const predictions = ['Analyzing...', 'Scanning Dermal Layers...', 'Processing Neural Map...'];
    hudInterval = setInterval(() => {
        frame++;
        document.getElementById('hudPrediction').textContent = predictions[frame % predictions.length];
        if (frame % 4 === 0) {
            document.getElementById('hudDistance').textContent = 'IN POSITION';
            document.getElementById('hudDistance').classList.remove('hud-wait');
            document.getElementById('hudRetinal').textContent = 'RETINAL LOCK: ENABLED';
        }
    }, 1500);
}

// ─── CAPTURE FRAME ─────────────────────────────────────────
function captureFrame() {
    const video = document.getElementById('videoFeed');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

    // Stop camera
    if (videoStream) { videoStream.getTracks().forEach(t => t.stop()); videoStream = null; }
    if (hudInterval) { clearInterval(hudInterval); hudInterval = null; }

    sendForPrediction(dataUrl);
}

// ─── STEP 2B: UPLOAD ───────────────────────────────────────
function triggerUpload() {
    document.getElementById('fileInput').click();
}

function handleUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        sendForPrediction(e.target.result);
    };
    reader.readAsDataURL(file);
}

// ─── SEND TO BACKEND ───────────────────────────────────────
async function sendForPrediction(base64Image) {
    showSection('loadingSection');

    // Animate progress bar
    const fill = document.getElementById('progressFill');
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        fill.style.width = progress + '%';
    }, 300);

    const loadingTexts = [
        'Processing bio-signature through MobileNetV2 engine...',
        'Analyzing dermal texture patterns...',
        'Running ocular stability assessment...',
        'Computing 10-year bio-stability projection...',
        'Generating holistic prescription matrix...'
    ];
    let textIdx = 0;
    const textInterval = setInterval(() => {
        textIdx = (textIdx + 1) % loadingTexts.length;
        document.getElementById('loadingText').textContent = loadingTexts[textIdx];
    }, 1200);

    try {
        const response = await fetch(API + '/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                frame: base64Image,
                name: patientName,
                age: patientAge
            })
        });

        const data = await response.json();

        clearInterval(progressInterval);
        clearInterval(textInterval);
        fill.style.width = '100%';

        if (data.error) {
            alert('Prediction Error: ' + data.error);
            showSection('scanSection');
            return;
        }

        setTimeout(() => renderResults(data), 600);

    } catch (err) {
        clearInterval(progressInterval);
        clearInterval(textInterval);
        console.error('API Error:', err);
        alert('Server connection failed. Make sure server.py is running on port 5000.');
        showSection('scanSection');
    }
}

// ─── RENDER RESULTS ────────────────────────────────────────
function renderResults(data) {
    showSection('resultsSection');

    // Banner
    const info = data.condition_info || {};
    document.getElementById('resultIcon').textContent = info.icon || '🔬';
    document.getElementById('resultPrediction').textContent = data.prediction;
    document.getElementById('resultDescription').textContent = info.description || '';

    // Confidence ring
    const conf = data.confidence;
    const circumference = 327;
    const offset = circumference - (circumference * conf / 100);
    const circle = document.getElementById('confCircle');
    setTimeout(() => { circle.style.strokeDashoffset = offset; }, 100);
    circle.style.transition = 'stroke-dashoffset 1.5s ease';
    animateNumber('confPercent', 0, conf, 1500);

    // Patient info
    document.getElementById('infoName').textContent = data.patient.name.toUpperCase();
    document.getElementById('infoAge').textContent = data.patient.age + ' Years';
    document.getElementById('infoTime').textContent = data.timestamp;

    // Probability bars
    const probContainer = document.getElementById('probBars');
    probContainer.innerHTML = '';
    const probs = data.all_probabilities;
    const colors = { 'Acne': '#ff4757', 'Eczema': '#ff6348', 'Psoriasis': '#a55eea', 'Wrinkles': '#ffa502', 'Healthy Skin': '#2ed573' };
    
    Object.entries(probs).forEach(([cls, val]) => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <span class="prob-label">${cls}</span>
            <div class="prob-track">
                <div class="prob-fill" style="background: ${colors[cls] || 'var(--accent-gradient)'};" id="prob_${cls.replace(/\s/g, '')}"></div>
            </div>
            <span class="prob-val">${val.toFixed(1)}%</span>
        `;
        probContainer.appendChild(item);
        setTimeout(() => {
            document.getElementById('prob_' + cls.replace(/\s/g, '')).style.width = val + '%';
        }, 200);
    });

    // Prescriptions
    document.getElementById('rxTreatment').textContent = data.treatment;
    document.getElementById('rxDiet').textContent = data.diet;
    document.getElementById('rxAge').textContent = data.age_advice;

    // Eye Status
    document.getElementById('eyeStatusText').textContent = '👁️ ' + data.eye_status;
    document.getElementById('eyeCare').textContent = data.eye_prescription.CARE;
    document.getElementById('eyeFruits').textContent = data.eye_prescription.FRUITS;
    document.getElementById('eyeMed').textContent = data.eye_prescription.MED;

    // Risk factor
    document.getElementById('riskFactor').textContent = data.projection.risk_factor;

    // 10-Year Chart
    renderChart(data.projection);
}

// ─── CHART RENDERING (Pure Canvas) ─────────────────────────
function renderChart(projection) {
    const canvas = document.getElementById('projectionChart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const pad = { top: 30, right: 30, bottom: 40, left: 55 };
    const chartW = W - pad.left - pad.right;
    const chartH = H - pad.top - pad.bottom;

    const years = projection.years;
    const scores = projection.scores;
    const maxScore = 105;

    ctx.clearRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = pad.top + (chartH / 5) * i;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
        const val = Math.round(maxScore - (maxScore / 5) * i);
        const y = pad.top + (chartH / 5) * i;
        ctx.fillText(val + '%', pad.left - 10, y + 4);
    }

    // X-axis labels
    ctx.textAlign = 'center';
    years.forEach((yr, i) => {
        const x = pad.left + (chartW / (years.length - 1)) * i;
        ctx.fillText(yr, x, H - pad.bottom + 22);
    });

    // Data points
    const points = scores.map((s, i) => ({
        x: pad.left + (chartW / (years.length - 1)) * i,
        y: pad.top + chartH - (s / maxScore) * chartH
    }));

    // Gradient fill under curve
    const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    grad.addColorStop(0, 'rgba(0, 210, 255, 0.15)');
    grad.addColorStop(1, 'rgba(0, 210, 255, 0)');
    ctx.beginPath();
    ctx.moveTo(points[0].x, H - pad.bottom);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(points[points.length - 1].x, H - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.strokeStyle = '#00d2ff';
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Data dots
    points.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#00d2ff';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0, 210, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

// ─── UTILITIES ─────────────────────────────────────────────
function animateNumber(elementId, start, end, duration) {
    const el = document.getElementById(elementId);
    const range = end - start;
    const startTime = performance.now();
    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = Math.round(start + range * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function resetApp() {
    // Reset confidence circle
    document.getElementById('confCircle').style.strokeDashoffset = 327;
    document.getElementById('confCircle').style.transition = 'none';
    document.getElementById('confPercent').textContent = '0';
    document.getElementById('progressFill').style.width = '0%';
    showSection('registrationSection');
}

// ─── KEYBOARD SHORTCUT ─────────────────────────────────────
document.addEventListener('keydown', (e) => {
    if (e.key === 's' || e.key === 'S') {
        const camSection = document.getElementById('cameraSection');
        if (!camSection.classList.contains('hidden-section')) {
            captureFrame();
        }
    }
    if (e.key === 'q' || e.key === 'Q') {
        const camSection = document.getElementById('cameraSection');
        if (!camSection.classList.contains('hidden-section')) {
            stopCamera();
        }
    }
});
