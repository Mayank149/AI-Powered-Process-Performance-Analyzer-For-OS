document.addEventListener('DOMContentLoaded', () => {
    // State
    let processes = [];
    let logs = [];
    let filterUser = false;
    let filterSystem = false;
    let isPaused = false;

    // Elements
    const tableBody = document.querySelector('#process-table tbody');
    const processCount = document.getElementById('process-count');
    const logWindow = document.getElementById('log-window');
    const btnRefresh = document.getElementById('btn-refresh');
    const btnClear = document.getElementById('btn-clear');
    const btnDownload = document.getElementById('btn-download');
    const checkUser = document.getElementById('filter-user');
    const checkSystem = document.getElementById('filter-system');

    // Charts
    const cpuCanvas = document.getElementById('cpu-chart');
    const memCanvas = document.getElementById('mem-chart');

    // Event Listeners
    btnRefresh.addEventListener('click', fetchData);

    btnClear.addEventListener('click', () => {
        tableBody.innerHTML = '';
        processCount.textContent = '0 processes';
    });

    btnDownload.addEventListener('click', downloadLogs);

    checkUser.addEventListener('click', () => {
        filterUser = !filterUser;
        checkUser.classList.toggle('active');
        renderTable();
    });

    checkSystem.addEventListener('click', () => {
        filterSystem = !filterSystem;
        checkSystem.classList.toggle('active');
        renderTable();
    });

    // Initial Load
    // Clear logs on reload as requested
    fetch('/clear-logs', { method: 'POST' })
        .then(() => {
            fetchData();
            fetchLogs();
            fetchForecast();
        })
        .catch(err => console.error('Error clearing logs:', err));

    // Intervals
    setInterval(fetchData, 2000);
    setInterval(fetchLogs, 2000);
    setInterval(fetchForecast, 5000);

    // --- Data Fetching ---

    async function fetchData() {
        if (isPaused) return;

        // Visual feedback for manual refresh
        const originalText = btnRefresh.textContent;
        if (originalText !== 'Refreshing...') {
            btnRefresh.textContent = 'Refreshing...';
            btnRefresh.disabled = true;
        }

        try {
            const res = await fetch('/live-process-data');
            const data = await res.json();
            processes = data;
            renderTable();
        } catch (err) {
            console.error('Error fetching process data:', err);
        } finally {
            // Restore button state
            if (btnRefresh.textContent === 'Refreshing...') {
                btnRefresh.textContent = 'Refresh Now';
                btnRefresh.disabled = false;
            }
        }
    }

    async function fetchLogs() {
        try {
            const res = await fetch('/log-stream');
            const data = await res.json();
            // Only update if new logs
            if (JSON.stringify(data) !== JSON.stringify(logs)) {
                logs = data;
                renderLogs();
            }
        } catch (err) {
            console.error('Error fetching logs:', err);
        }
    }

    async function fetchForecast() {
        try {
            const res = await fetch('/forecast');
            const data = await res.json();
            drawChart(cpuCanvas, data.cpu, 'CPU %', '#3b82f6');
            drawChart(memCanvas, data.memory, 'Memory %', '#8b5cf6');
        } catch (err) {
            console.error('Error fetching forecast:', err);
        }
    }

    // --- Rendering ---

    function renderTable() {
        // Filter logic
        let filtered = processes;

        // If both unchecked, show all. If one checked, show that type.
        // Note: "User" vs "System" is a bit ambiguous in cross-platform psutil.
        // We'll use a heuristic: System usually has low PIDs or specific names, 
        // but for this demo, we might just filter by username if available, 
        // or just assume everything is "User" unless we have a flag.
        // Since the backend doesn't explicitly send "type", let's use a simple heuristic:
        // System: PID < 1000 (on Linux/Mac) or specific names.
        // For Windows, it's harder. Let's just assume all are shown unless filtered.

        // Actually, let's implement the requested logic:
        // "Checkbox: show only user processes"
        // "Checkbox: show only system processes"
        // "If both are unchecked, show all processes"

        if (filterUser || filterSystem) {
            filtered = processes.filter(p => {
                // Simple heuristic for demo purposes
                // In reality, you'd check p.username() from psutil
                const isSystem = p.pid < 1000 || ['System', 'Registry', 'smss.exe', 'csrss.exe', 'wininit.exe', 'services.exe', 'lsass.exe'].includes(p.name);

                if (filterUser && filterSystem) return true; // Both checked = show all? Or intersection? Usually union.
                if (filterUser && !isSystem) return true;
                if (filterSystem && isSystem) return true;
                return false;
            });
        }

        processCount.textContent = `${filtered.length} processes`;

        const html = filtered.map(p => {
            const isAnomaly = p.anomaly_label === -1;
            const rowClass = isAnomaly ? 'anomaly-row' : '';
            const status = isAnomaly ? 'ANOMALY' : 'Normal';

            return `
                <tr class="${rowClass}">
                    <td>${p.pid}</td>
                    <td>${p.name}</td>
                    <td>${p.cpu_percent.toFixed(1)}%</td>
                    <td>${p.memory_percent.toFixed(1)}%</td>
                    <td>${p.num_threads}</td>
                    <td>${formatBytes(p.read_speed)}</td>
                    <td>${formatBytes(p.write_speed)}</td>
                    <td>${status}</td>
                </tr>
            `;
        }).join('');

        tableBody.innerHTML = html;
    }

    function renderLogs() {
        const html = logs.map(log => {
            return `
                <div class="log-entry anomaly">
                    <span class="time">[${log.timestamp.split(' ')[1]}]</span>
                    <strong>${log.name}</strong> (PID: ${log.pid}) - Score: ${log.anomaly_score.toFixed(2)}
                </div>
            `;
        }).join('');

        logWindow.innerHTML = html;
    }

    // --- Charts (Canvas) ---

    function drawChart(canvas, data, label, color) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width = canvas.offsetWidth;
        const height = canvas.height = canvas.offsetHeight;

        // Clear
        ctx.clearRect(0, 0, width, height);

        // Config
        const padding = 20;
        const chartWidth = width - padding * 2;
        const chartHeight = height - padding * 2;
        const stepX = chartWidth / (data.length - 1);

        // Max value (Auto-scale for visibility, min 10%)
        const maxData = Math.max(...data);
        const maxY = Math.max(maxData * 1.2, 10);

        // Draw Line
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        data.forEach((val, i) => {
            const x = padding + i * stepX;
            const y = height - padding - (val / maxY * chartHeight);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.stroke();

        // Draw Fill
        ctx.lineTo(padding + (data.length - 1) * stepX, height - padding);
        ctx.lineTo(padding, height - padding);
        ctx.fillStyle = color + '20'; // Low opacity
        ctx.fill();

        // Draw Points
        ctx.fillStyle = color;
        data.forEach((val, i) => {
            const x = padding + i * stepX;
            const y = height - padding - (val / maxY * chartHeight);

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // --- Utilities ---

    function formatBytes(bytes, decimals = 2) {
        if (!+bytes) return '0 B';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }

    function downloadLogs() {
        if (logs.length === 0) {
            alert("No logs to download");
            return;
        }

        const headers = ["Timestamp", "PID", "Name", "Anomaly Score", "Details"];
        const csvContent = [
            headers.join(","),
            ...logs.map(log => [
                log.timestamp,
                log.pid,
                log.name,
                log.anomaly_score,
                `"${log.details}"`
            ].join(","))
        ].join("\n");

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.setAttribute('hidden', '');
        a.setAttribute('href', url);
        a.setAttribute('download', 'anomaly_logs.csv');
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
});
