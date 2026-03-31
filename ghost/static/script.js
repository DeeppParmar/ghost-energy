/**
 * VisionCore AI HUD - Command Center Logic
 * Orchestrates real-time telemetry, bento-grid animations, and neural events.
 */

class VisionHUD {
    constructor() {
        this.occChart = null;
        this.lastUpdate = Date.now();
        this._toastActiveKeys = new Set();
        this._lastPresenceCount = null;
        this._lastNoHumanToastAt = 0;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.startPolling();
        this.initCharts();
        this.loadSettings();
        this.loadReports();
    }

    setupEventListeners() {
        // Save Settings (on Settings Page)
        const saveBtn = document.getElementById('save-settings');
        if (saveBtn) {
            saveBtn.onclick = () => this.saveSettings();
        }

        // Test Email (on Settings Page)
        const testEmailBtn = document.getElementById('test-email-btn');
        if (testEmailBtn) {
            testEmailBtn.onclick = () => this.testEmail();
        }

        // Test Telegram (on Settings Page)
        const testTelegramBtn = document.getElementById('test-telegram-btn');
        if (testTelegramBtn) {
            testTelegramBtn.onclick = () => this.testTelegram();
        }

        // Report Generation (on Reports Page)
        const reportBtn = document.getElementById('generate-report-btn');
        if (reportBtn) {
            reportBtn.onclick = () => this.generateReport();
        }

        // CSV Export (History Page)
        const exportBtn = document.getElementById('export-csv-btn');
        if (exportBtn) {
            exportBtn.onclick = () => this.exportCSV();
        }

        // System Health Badge (Home Page)
        const healthBadge = document.getElementById('system-health-badge');
        if (healthBadge) {
            healthBadge.onclick = () => this.toggleSystemHealthPanel();
        }

        // Heatmap Refresh (History Page)
        const heatmapRefreshBtn = document.getElementById('heatmap-refresh-btn');
        if (heatmapRefreshBtn) {
            heatmapRefreshBtn.onclick = () => this.renderHeatmap();
        }

        const streamModeEl = document.getElementById('stream-mode');
        const videoFeedEl = document.getElementById('video-feed');
        if (streamModeEl && videoFeedEl) {
            streamModeEl.onchange = () => {
                const m = streamModeEl.value;
                if (m === 'thermal') videoFeedEl.src = '/video_feed_thermal';
                else if (m === 'trails') videoFeedEl.src = '/video_feed_trails';
                else videoFeedEl.src = '/video_feed';
            };
        }
    }

    pollSequence(fn, interval) {
        const execute = async () => {
            try {
                await fn();
            } catch (e) {
                // Ignore network interruption errors in polling
            } finally {
                setTimeout(execute, interval);
            }
        };
        execute();
    }

    startPolling() {
        // Main status update (Monitor Page)
        if (document.getElementById('person-count') || document.getElementById('light-status')) {
            this.pollSequence(() => this.updateStatus(), 1500);
        }

        // System health (CPU/RAM) (Universal)
        if (document.getElementById('cpu-bar')) {
            this.pollSequence(() => this.loadStats(), 4000);
        }

        // Energy Cost (Monitor Page)
        if (document.getElementById('money-wasted')) {
            this.pollSequence(() => this.updateEnergyCost(), 3000);
        }

        // Audit Log (History/Monitor Page)
        if (document.getElementById('audit-log')) {
            this.pollSequence(() => this.loadAuditLog(), 8000);
        }

        // Occupancy Chart (Monitor Page)
        if (document.getElementById('occupancyChart')) {
            this.pollSequence(() => this.updateOccupancyChart(), 7000);
        }

        // System Health (Home Page)
        if (document.getElementById('system-health-badge')) {
            this.pollSequence(() => this.loadSystemHealth(), 15000);
        }

        // Heatmap (History Page)
        if (document.getElementById('heatmap-container')) {
            this.renderHeatmap();
        }

        // Monthly Summary (Home Page)
        if (document.getElementById('monthly-total-alerts')) {
            this.pollSequence(() => this.loadMonthlySummary(), 60000);
            this.pollSequence(() => this.loadProjection(), 60000);
            this.pollSequence(() => this.loadLeaderboard(), 60000);
        }
    }

    toggleSystemHealthPanel() {
        const panel = document.getElementById('system-health-panel');
        if (!panel) return;
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }

    async loadSystemHealth() {
        try {
            const res = await fetch('/health');
            if (!res.ok) throw new Error('Health check failed');
            const d = await res.json();

            const dotEl = document.getElementById('system-health-dot');
            const textEl = document.getElementById('system-health-text');
            const jsonPre = document.getElementById('system-health-json');

            if (jsonPre) jsonPre.textContent = JSON.stringify(d, null, 2);

            if (!dotEl || !textEl) return;

            if (d.status === 'ok') {
                dotEl.style.background = 'var(--success)';
                dotEl.style.boxShadow = '0 0 14px rgba(16,185,129,0.45)';
                textEl.textContent = 'System OK';
                textEl.style.color = 'var(--success)';
            } else if (d.status === 'degraded') {
                dotEl.style.background = 'var(--warning)';
                dotEl.style.boxShadow = '0 0 14px rgba(245,158,11,0.45)';
                textEl.textContent = 'Degraded';
                textEl.style.color = 'var(--warning)';
            } else {
                dotEl.style.background = 'var(--danger)';
                dotEl.style.boxShadow = '0 0 14px rgba(248,113,113,0.45)';
                textEl.textContent = 'Error';
                textEl.style.color = 'var(--danger)';
            }
        } catch (e) {
            console.error('Health poll failed:', e);
        }
    }

    async updateStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            this.lastUpdate = Date.now();

            // Presence
            const countEl = document.getElementById('person-count');
            if (countEl) {
                const newText = data.person_count > 0 ? `${data.person_count} PERSON${data.person_count > 1 ? 'S' : ''}` : 'EMPTY';
                if (countEl.textContent !== newText) {
                    this.animateValue(countEl, newText);
                }
            }

            // Light
            const lightStatusEl = document.getElementById('light-status');
            const lightTypeEl = document.getElementById('light-type');
            if (lightStatusEl) lightStatusEl.textContent = data.light_status;
            if (lightTypeEl) lightTypeEl.textContent = data.light_type || 'Spectrum Analyzing...';

            // Luminance HUD overlay (monitor page)
            const lumValEl = document.getElementById('luminance-value');
            const lumModeEl = document.getElementById('luminance-mode');
            if (lumValEl) {
                const lum = typeof data.luminance === 'number' ? data.luminance : null;
                lumValEl.textContent = lum !== null ? Math.round(lum).toString().padStart(3, ' ') : '--';
            }
            if (lumModeEl) {
                lumModeEl.textContent = data.light_status || 'Analyzing…';
            }

            // AI FPS
            const fpsEl = document.getElementById('ai-fps');
            if (fpsEl && data.ai_fps !== undefined) {
                fpsEl.textContent = data.ai_fps.toFixed(1);
            }

            // Confidence Badge (Monitor Page)
            const confPill = document.getElementById('confidence-pill');
            if (confPill) {
                const confRaw = data.detection_confidence;
                const confNum = confRaw === null || confRaw === undefined ? null : Number(confRaw);
                const hasDet = confNum !== null && Number.isFinite(confNum);

                if (!hasDet) {
                    confPill.textContent = 'No detections';
                    confPill.style.backgroundColor = 'rgba(148,163,184,0.08)';
                    confPill.style.borderColor = 'rgba(148,163,184,0.25)';
                    confPill.style.color = 'var(--text-dim)';
                } else {
                    const pct = Math.max(0, Math.min(100, Math.round(confNum * 100)));
                    confPill.textContent = `${pct}%`;

                    if (confNum >= 0.75) {
                        confPill.style.backgroundColor = 'hsla(var(--h-success), var(--s-success), var(--l-success), 0.2)';
                        confPill.style.borderColor = 'hsla(var(--h-success), var(--s-success), var(--l-success), 0.35)';
                        confPill.style.color = 'var(--success)';
                    } else if (confNum >= 0.5) {
                        confPill.style.backgroundColor = 'hsla(var(--h-warning), var(--s-warning), var(--l-warning), 0.2)';
                        confPill.style.borderColor = 'hsla(var(--h-warning), var(--s-warning), var(--l-warning), 0.35)';
                        confPill.style.color = 'var(--warning)';
                    } else {
                        confPill.style.backgroundColor = 'hsla(var(--h-danger), var(--s-danger), var(--l-danger), 0.2)';
                        confPill.style.borderColor = 'hsla(var(--h-danger), var(--s-danger), var(--l-danger), 0.35)';
                        confPill.style.color = 'var(--danger)';
                    }
                }
            }

            // Verifier Badge (Monitor Page)
            const verifierPill = document.getElementById('verifier-pill');
            if (verifierPill) {
                const isOn = !!data.verifier_active;
                if (isOn) {
                    verifierPill.textContent = 'Verifier ON';
                    verifierPill.style.backgroundColor = 'hsla(var(--h-success), var(--s-success), var(--l-success), 0.18)';
                    verifierPill.style.borderColor = 'hsla(var(--h-success), var(--s-success), var(--l-success), 0.32)';
                    verifierPill.style.color = 'var(--success)';
                } else {
                    verifierPill.textContent = 'Verifier OFF';
                    verifierPill.style.backgroundColor = 'rgba(148,163,184,0.08)';
                    verifierPill.style.borderColor = 'rgba(148,163,184,0.25)';
                    verifierPill.style.color = 'var(--text-dim)';
                }
            }

            // Presence Stats
            const timeSinceEl = document.getElementById('time-since');
            if (timeSinceEl) {
                if (data.person_count > 0) {
                    timeSinceEl.textContent = 'Presence Verified ✓';
                } else {
                    const idle = data.time_since_presence;
                    const mins = Math.floor(idle / 60);
                    const secs = idle % 60;
                    timeSinceEl.textContent = mins > 0 ? `Empty for ${mins}m ${secs}s` : `Empty for ${secs}s`;
                }
            }

            const focusEl = document.getElementById('focus-timer');
            if (focusEl) {
                const sec = Number(data.focus_seconds || 0);
                const m = Math.floor(sec / 60);
                const s = sec % 60;
                focusEl.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
            }

            // Focus milestone badges
            if (data.focus_milestones) {
                const ms1 = document.getElementById('ms-1m');
                const ms5 = document.getElementById('ms-5m');
                const ms30 = document.getElementById('ms-30m');
                if (ms1) ms1.classList.toggle('unlocked', !!data.focus_milestones['1m']);
                if (ms5) ms5.classList.toggle('unlocked', !!data.focus_milestones['5m']);
                if (ms30) ms30.classList.toggle('unlocked', !!data.focus_milestones['30m']);
            }

            // Energy Alert & Auto-Off Banner
            const alertBanner = document.getElementById('energy-alert');
            if (alertBanner) {
                if (data.auto_off_active) {
                    alertBanner.style.display = 'block';
                    alertBanner.style.background = 'var(--danger)';
                    alertBanner.style.color = '#fff';
                    alertBanner.innerHTML = '⚠️ CRITICAL: SYSTEM AUTO-POWERED DOWN DUE TO INACTIVITY';
                } else if (data.is_energy_wasted && data.person_count === 0) {
                    alertBanner.style.display = 'block';
                    alertBanner.style.background = '';
                    alertBanner.style.color = '';
                    alertBanner.innerHTML = '⚠️ CRITICAL: ENERGY WASTE DETECTED IN VACANT SECTOR';
                } else {
                    alertBanner.style.display = 'none';
                }
            }

            // Toast: Only when we enter a waste state (not on every idle poll).
            // Conditions:
            //  - Room is considered wasting energy (backend logic)
            //  - Person count dropped to 0 from >0 OR we have just crossed 30s idle
            //  - Toast not shown in the last 2 minutes
            const now = Date.now();
            const wasOccupied = (this._lastPresenceCount ?? 0) > 0;
            const justBecameEmpty = wasOccupied && data.person_count === 0;
            const idleSeconds = data.time_since_presence || 0;
            const crossedIdleThreshold = idleSeconds >= 30;
            const longSinceLastToast = (now - this._lastNoHumanToastAt) > 120000;

            if (data.is_energy_wasted && (justBecameEmpty || crossedIdleThreshold) && longSinceLastToast) {
                this.showToastOnce(
                    'no-human-bright',
                    'No human in frame',
                    'AI sees lights but no presence in the sector.',
                    'warning'
                );
                this._lastNoHumanToastAt = now;
            }

            // Reset toast key when conditions are healthy again to allow future alerts.
            if (!data.is_energy_wasted || data.person_count > 0 || data.light_status === 'Dark') {
                this._toastActiveKeys.delete('no-human-bright');
            }

            this._lastPresenceCount = data.person_count;

        } catch (err) {
            console.error('Status fetch error:', err);
        }
    }

    animateValue(el, newValue) {
        el.style.transition = 'none';
        el.style.opacity = '0.4';
        el.style.transform = 'translateY(5px)';
        requestAnimationFrame(() => {
            el.textContent = newValue;
            el.style.transition = 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)';
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        });
    }

    async loadStats() {
        try {
            const res = await fetch('/stats');
            const data = await res.json();

            // System Health Bars
            if (data.system) {
                this.updateHealthBar('cpu', data.system.cpu_percent);
                this.updateHealthBar('ram', data.system.ram_percent);
            }
        } catch (e) { console.error(e); }
    }

    updateHealthBar(id, percent) {
        const bar = document.getElementById(`${id}-bar`);
        const val = document.getElementById(`${id}-val`);
        if (bar) bar.style.width = percent + '%';
        if (val) val.textContent = `${percent.toFixed(0)}%`;
    }

    async updateEnergyCost() {
        try {
            const res = await fetch('/api/energy_live');
            const d = await res.json();

            const moneyEl = document.getElementById('money-wasted');
            const trendEl = document.querySelector('.energy-trend');
            
            if (moneyEl) moneyEl.textContent = `₹${d.today.money_wasted.toFixed(2)}`;
            if (trendEl) trendEl.style.display = d.is_wasting_now ? 'inline-block' : 'none';
            
            const kwhEl = document.getElementById('kwh-wasted');
            if (kwhEl) kwhEl.textContent = `${d.today.kwh_wasted.toFixed(3)} kWh`;

            const mins = Math.floor(d.today.waste_minutes);
            const secs = Math.floor(d.today.waste_seconds % 60);
            const wasteTimeEl = document.getElementById('waste-time');
            if (wasteTimeEl) wasteTimeEl.textContent = `${mins}m ${secs}s`;

            const burnRateEl = document.getElementById('burn-rate');
            if (burnRateEl) burnRateEl.textContent = `₹${d.config.rate_per_hour.toFixed(2)}/hr`;

        } catch(e) { console.error('Energy cost error:', e); }
    }

    async loadAuditLog() {
        try {
            const res = await fetch('/api/history'); // Note: Updated to API path
            const data = await res.json();
            const logEl = document.getElementById('audit-log');
            if (!logEl) return;

            if (data.entries && data.entries.length > 0) {
                const recent = data.entries.reverse(); // Show all on history page
                logEl.innerHTML = recent.map(e => {
                    const durationNum = Number(e.Duration_Seconds);
                    const durationSafe = Number.isFinite(durationNum) ? durationNum : 0;

                    const moneyRaw = e.Money_Wasted;
                    const moneyNum = Number(moneyRaw);
                    const moneySafe = Number.isFinite(moneyNum) ? moneyNum : 0;

                    return `
                        <div style="padding: 12px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.01); border-radius: 8px; margin-bottom: 8px;">
                            <div>
                                <div style="font-weight: 800; color: var(--text);">${e.Room || 'Sector Unknown'}</div>
                                <div style="font-size: 0.7rem; color: var(--text-dim);">${e.Timestamp}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-weight: 800; color: var(--warning); font-size: 1.1rem;">⚡ ${durationSafe.toFixed(0)}s</div>
                                <small style="color: var(--cyan); font-size: 0.65rem; font-weight: 700;">₹${moneySafe.toFixed(2)} LOSS</small>
                            </div>
                        </div>
                    `;
                }).join('');
            } else {
                logEl.innerHTML = '<div style="opacity: 0.5; padding: 2.5rem; text-align: center;">No waste events recorded in database.</div>';
            }
        } catch (e) { console.error(e); }
    }

    exportCSV() {
        const fromEl = document.getElementById('export-from-date');
        const toEl = document.getElementById('export-to-date');
        const statusEl = document.getElementById('export-csv-status');
        const btn = document.getElementById('export-csv-btn');

        if (!fromEl || !toEl) return;

        const fromVal = fromEl.value;
        const toVal = toEl.value;

        const params = new URLSearchParams();
        if (fromVal) params.set('from', fromVal);
        if (toVal) params.set('to', toVal);

        const url = params.toString() ? `/api/export_csv?${params.toString()}` : '/api/export_csv';

        if (statusEl) statusEl.textContent = 'Exporting…';
        if (btn) btn.disabled = true;

        try {
            const a = document.createElement('a');
            a.href = url;
            a.download = 'visioncore_audit_export.csv';
            document.body.appendChild(a);
            a.click();
            a.remove();

            setTimeout(() => {
                if (statusEl) {
                    statusEl.textContent = 'Done ✓';
                }
                if (btn) btn.disabled = false;
            }, 900);
        } catch (e) {
            console.error('Export CSV error:', e);
            if (statusEl) statusEl.textContent = 'Export Failed';
            if (btn) btn.disabled = false;
        }
    }

    async initCharts() {
        if (document.getElementById('occupancyChart')) {
            this.updateOccupancyChart();
        }
    }

    async updateOccupancyChart() {
        try {
            const res = await fetch('/api/occupancy_history');
            const d = await res.json();
            const canvas = document.getElementById('occupancyChart');
            if (!canvas) return;

            const ctx = canvas.getContext('2d');
            const chartData = d.data.length ? d.data : [0, 0];
            const chartLabels = d.labels.length ? d.labels : ['', ''];

            if (!this.occChart) {
                const gradient = ctx.createLinearGradient(0, 0, 0, 150);
                gradient.addColorStop(0, 'hsla(188, 86%, 53%, 0.4)');
                gradient.addColorStop(1, 'transparent');

                this.occChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: chartLabels,
                        datasets: [{
                            label: 'Tracks',
                            data: chartData,
                            fill: true,
                            backgroundColor: gradient,
                            borderColor: 'hsl(188, 86%, 53%)',
                            borderWidth: 2,
                            tension: 0.4,
                            pointRadius: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: { duration: 1000 },
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { 
                                display: false, 
                                min: 0, 
                                max: Math.max(3, ...chartData) + 1 
                            },
                            x: { display: false }
                        }
                    }
                });
            } else {
                this.occChart.data.labels = chartLabels;
                this.occChart.data.datasets[0].data = chartData;
                this.occChart.update();
            }
        } catch(e) { console.error('Chart update error:', e); }
    }

    async renderHeatmap() {
        return this._renderHeatmapInternal();
    }

    async _renderHeatmapInternal(filterDate) {
        const container = document.getElementById('heatmap-container');
        if (!container) return;

        try {
            const url = filterDate ? `/api/heatmap_data?date=${encodeURIComponent(filterDate)}` : '/api/heatmap_data';
            const res = await fetch(url);
            const d = await res.json();

            container.innerHTML = '';

            const maxSeconds = Number(d.max_seconds) || 0;
            const cells = Array.isArray(d.cells) ? d.cells : [];

            if (!cells.length || maxSeconds <= 0) {
                container.innerHTML = `
                    <div style="opacity: 0.6; padding: 1.25rem; text-align:center; border:1px dashed var(--border); border-radius:12px;">
                        No waste heatmap data yet.
                    </div>
                `;
                return;
            }

            const cellMap = new Map();
            cells.forEach(c => {
                const day = Number(c.day);
                const hour = Number(c.hour);
                const sec = Number(c.seconds) || 0;
                if (Number.isFinite(day) && Number.isFinite(hour)) {
                    cellMap.set(`${day}-${hour}`, sec);
                }
            });

            const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

            const table = document.createElement('table');
            table.style.borderCollapse = 'separate';
            table.style.borderSpacing = '6px 6px';
            table.style.margin = '0 auto';
            table.style.userSelect = 'none';

            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');

            const corner = document.createElement('th');
            corner.style.padding = '0 6px';
            corner.style.color = 'var(--text-dim)';
            corner.style.fontSize = '0.65rem';
            corner.style.fontWeight = '800';
            corner.style.letterSpacing = '1px';
            headerRow.appendChild(corner);

            for (let h = 0; h < 24; h++) {
                const th = document.createElement('th');
                th.textContent = String(h).padStart(2, '0');
                th.style.color = 'var(--text-dim)';
                th.style.fontSize = '0.65rem';
                th.style.fontWeight = '800';
                th.style.letterSpacing = '0.5px';
                th.style.padding = '0 4px';
                headerRow.appendChild(th);
            }
            thead.appendChild(headerRow);

            const tbody = document.createElement('tbody');
            for (let day = 0; day < 7; day++) {
                const tr = document.createElement('tr');

                const dayTh = document.createElement('th');
                dayTh.textContent = days[day];
                dayTh.style.color = 'var(--text-dim)';
                dayTh.style.fontSize = '0.7rem';
                dayTh.style.fontWeight = '900';
                dayTh.style.padding = '0 10px 0 0';
                tr.appendChild(dayTh);

                for (let hour = 0; hour < 24; hour++) {
                    const td = document.createElement('td');
                    td.style.width = '22px';
                    td.style.height = '22px';
                    td.style.borderRadius = '6px';
                    td.style.border = '1px solid var(--border)';
                    td.style.backgroundColor = 'rgba(255,255,255,0.02)';
                    td.style.transition = 'background-color 0.3s ease';

                    const seconds = cellMap.get(`${day}-${hour}`) || 0;
                    const secondsInt = Math.round(seconds);
                    const m = Math.floor(secondsInt / 60);
                    const s = secondsInt % 60;

                    td.title = `${days[day]} ${String(hour).padStart(2, '0')}:00 — ${m}m ${s}s wasted`;

                    if (secondsInt > 0 && maxSeconds > 0) {
                        const alpha = Math.min(1, Math.max(0, seconds / maxSeconds));
                        td.style.backgroundColor = `hsla(var(--h-warning), var(--s-warning), var(--l-warning), ${alpha})`;
                        td.style.borderColor = 'rgba(245,158,11,0.35)';
                    }

                    tr.appendChild(td);
                }
                tbody.appendChild(tr);
            }

            table.appendChild(thead);
            table.appendChild(tbody);
            container.appendChild(table);
        } catch (e) {
            console.error('Heatmap render failed:', e);
            container.innerHTML = `<div style="opacity:0.7; padding:1.25rem;">Heatmap failed to load.</div>`;
        }
    }

    async renderHeatmapForDate(dateIso) {
        await this._renderHeatmapInternal(dateIso);
    }

    async loadMonthlySummary() {
        try {
            const res = await fetch('/api/monthly_summary');
            const d = await res.json();

            const monthLabelEl = document.getElementById('monthly-month-label');
            const alertsEl = document.getElementById('monthly-total-alerts');
            const wasteHoursEl = document.getElementById('monthly-total-waste-hours');
            const moneyEl = document.getElementById('monthly-total-money-wasted');
            const avgWasteEl = document.getElementById('monthly-avg-daily-waste');

            if (monthLabelEl) monthLabelEl.textContent = d.month_label || '--';
            if (alertsEl) alertsEl.textContent = d.total_alerts ?? '--';

            if (wasteHoursEl) {
                const hrs = Number(d.total_waste_hours);
                wasteHoursEl.innerHTML = Number.isFinite(hrs) ? `${hrs.toFixed(2)} <span class="stat-unit">hrs</span>` : '--';
            }

            if (moneyEl) {
                const money = Number(d.total_money_wasted_inr);
                moneyEl.textContent = Number.isFinite(money) ? `₹${money.toFixed(2)}` : '₹--';
            }

            if (avgWasteEl) {
                const avgMin = Number(d.avg_daily_waste_minutes);
                if (Number.isFinite(avgMin)) {
                    const avgSecondsTotal = Math.round(avgMin * 60);
                    const m = Math.floor(avgSecondsTotal / 60);
                    const s = avgSecondsTotal % 60;
                    avgWasteEl.textContent = `${m}m ${s}s`;
                } else {
                    avgWasteEl.textContent = '--';
                }
            }
        } catch (e) {
            console.error('Monthly summary load failed:', e);
        }
    }

    async loadProjection() {
        try {
            const res = await fetch('/api/projection');
            const d = await res.json();
            const h = document.getElementById('projection-hours');
            const m = document.getElementById('projection-money');
            if (h) h.textContent = `${Number(d.predicted_waste_hours || 0).toFixed(2)} hrs`;
            if (m) m.textContent = `₹${Number(d.predicted_money_wasted || 0).toFixed(2)} (tomorrow)`;
        } catch (e) {}
    }

    async loadLeaderboard() {
        try {
            const res = await fetch('/api/leaderboard');
            const d = await res.json();
            const r = document.getElementById('leader-rank');
            const desc = document.getElementById('leader-desc');
            if (r) r.textContent = `#${d.rank} of ${d.total}`;
            if (desc) desc.textContent = `Your eco ranking today`;

            // Render full leaderboard table
            const table = document.getElementById('leaderboard-table');
            if (table && Array.isArray(d.scores)) {
                table.innerHTML = d.scores.map((s, i) => {
                    const isYou = s.name === 'You';
                    const medals = ['🥇', '🥈', '🥉'];
                    const medal = i < 3 ? medals[i] : `#${i + 1}`;
                    const barW = Math.max(5, Math.min(100, s.eco_score));
                    const barColor = isYou ? 'rgba(99,102,241,0.6)' : 'rgba(255,255,255,0.08)';
                    const nameStyle = isYou ? 'color:#fff;font-weight:900;' : 'color:var(--text-dim);font-weight:600;';
                    return `
                        <div style="display:flex;align-items:center;gap:10px;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.04);${ isYou ? 'background:rgba(99,102,241,0.06);margin:0 -0.5rem;padding:0.5rem;border-radius:8px;' : '' }">
                            <span style="min-width:28px;text-align:center;font-size:0.9rem;">${medal}</span>
                            <span style="flex:1;font-size:0.8rem;${nameStyle}">${s.name}</span>
                            <div style="width:80px;height:6px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;">
                                <div style="width:${barW}%;height:100%;background:${barColor};border-radius:3px;transition:width 0.5s;"></div>
                            </div>
                            <span style="min-width:36px;text-align:right;font-size:0.75rem;font-weight:800;color:${s.eco_score >= 80 ? 'var(--success)' : s.eco_score >= 50 ? 'var(--warning)' : 'var(--danger)'}">${s.eco_score}</span>
                        </div>
                    `;
                }).join('');
            }
        } catch (e) { console.error('Leaderboard error:', e); }
    }

    async loadSettings() {
        if (!document.getElementById('receiver-email')) return;
        try {
            const res = await fetch('/api/settings');
            const data = await res.json();
            ['receiver-email', 'sender-email', 'sender-password', 'room-name', 'alert-delay', 'camera-source'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = data[id.replace('-', '_')] || '';
            });
            // Telegram settings
            const tgEnabled = document.getElementById('telegram-enabled');
            const tgToken = document.getElementById('telegram-bot-token');
            const tgChatId = document.getElementById('telegram-chat-id');
            if (tgEnabled) tgEnabled.checked = !!data.telegram_enabled;
            if (tgToken) tgToken.value = data.telegram_bot_token || '';
            if (tgChatId) tgChatId.value = data.telegram_chat_id || '';
            
            // Smart Automation
            const autoOffDelay = document.getElementById('auto-off-delay');
            const logFreq = document.getElementById('log-frequency');
            if (autoOffDelay) autoOffDelay.value = data.auto_off_delay_minutes || 0;
            if (logFreq) logFreq.value = data.log_frequency_minutes || 0;
        } catch (e) {}
    }

    async saveSettings() {
        const msgEl = document.getElementById('settings-msg');
        const payload = {
            receiver_email: document.getElementById('receiver-email').value,
            room_name: document.getElementById('room-name').value,
            alert_delay: document.getElementById('alert-delay').value,
            camera_source: document.getElementById('camera-source').value
        };

            // Include Telegram settings if the fields exist on the page
        const tgEnabled = document.getElementById('telegram-enabled');
        const tgToken = document.getElementById('telegram-bot-token');
        const tgChatId = document.getElementById('telegram-chat-id');
        if (tgEnabled) payload.telegram_enabled = tgEnabled.checked;
        if (tgToken) payload.telegram_bot_token = tgToken.value;
        if (tgChatId) payload.telegram_chat_id = tgChatId.value;

        // General Email Credentials
        const senEl = document.getElementById('sender-email');
        const senPassEl = document.getElementById('sender-password');
        if (senEl) payload.sender_email = senEl.value;
        if (senPassEl) payload.sender_password = senPassEl.value;

        // Smart Automation
        const autoOffDelay = document.getElementById('auto-off-delay');
        const logFreq = document.getElementById('log-frequency');
        if (autoOffDelay) payload.auto_off_delay_minutes = parseInt(autoOffDelay.value) || 0;
        if (logFreq) payload.log_frequency_minutes = parseInt(logFreq.value) || 0;

        try {
            const res = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success && data.camera_restarted) {
                this.showStatusMsg(msgEl, '✅ Saved. Camera source switched.');
            } else {
                this.showStatusMsg(msgEl, data.success ? '✅ Committed to AI Core' : '❌ Committal Failed');
            }
        } catch (e) { this.showStatusMsg(msgEl, '❌ Local Sync Error'); }
    }

    async testEmail() {
        const msgEl = document.getElementById('settings-msg');
        this.showStatusMsg(msgEl, '📧 Dispatching Test Relay...', 'var(--accent)');
        try {
            const recEl = document.getElementById('receiver-email');
            const senEl = document.getElementById('sender-email');
            const senPassEl = document.getElementById('sender-password');
            const body = {};
            if (recEl) body.receiver_email = recEl.value;
            if (senEl) body.sender_email = senEl.value;
            if (senPassEl) body.sender_password = senPassEl.value;

            const res = await fetch('/api/test_email', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const d = await res.json();
            this.showStatusMsg(msgEl, d.success ? '✅ Protocol Confirmed' : '❌ Relay Nullified. Check App Password.', d.success ? 'var(--success)' : 'var(--danger)');
        } catch (e) { this.showStatusMsg(msgEl, '❌ Network Partition', 'var(--danger)'); }
    }

    async testTelegram() {
        const msgEl = document.getElementById('telegram-msg');
        if (msgEl) {
            msgEl.style.color = 'var(--text-dim)';
            msgEl.textContent = '📲 Sending test message...';
        }
        try {
            // Send current form values — backend will auto-save them
            const tokenEl = document.getElementById('telegram-bot-token');
            const chatEl = document.getElementById('telegram-chat-id');
            const enabledEl = document.getElementById('telegram-enabled');
            const body = {};
            if (tokenEl) body.telegram_bot_token = tokenEl.value;
            if (chatEl) body.telegram_chat_id = chatEl.value;
            if (enabledEl) body.telegram_enabled = enabledEl.checked;

            const res = await fetch('/api/test_telegram', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const d = await res.json();
            if (msgEl) {
                msgEl.style.color = d.success ? 'var(--success)' : 'var(--danger)';
                msgEl.textContent = d.message;
            }
        } catch (e) {
            if (msgEl) {
                msgEl.style.color = 'var(--danger)';
                msgEl.textContent = '❌ Connection failed';
            }
        }
    }

    async generateReport() {
        const btn = document.getElementById('generate-report-btn');
        const status = document.getElementById('report-status');
        btn.disabled = true;
        btn.textContent = '⏳ Compiling...';
        
        try {
            const res = await fetch('/api/generate_report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const d = await res.json();
            if (d.success) {
                status.innerHTML = `<span style="color:var(--success)">✅ Report Optimized</span> 
                    <a href="${d.download_url}" style="color:var(--cyan);text-decoration:none;margin-left:10px;" download>⬇ Download</a>`;
                this.loadReports();
            }
        } catch(e) { console.error('Reporting error:', e); }
        btn.disabled = false;
        btn.textContent = 'Run System Audit →';
    }

    async loadReports() {
        const list = document.getElementById('report-list');
        if (!list) return;
        try {
            const res = await fetch('/api/reports');
            const data = await res.json();
            list.innerHTML = data.reports.length ? 
                data.reports.map(r => `
                    <a class="report-link" href="${r.download_url}" download style="display: flex; justify-content: space-between; padding: 12px; background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 8px; text-decoration: none; font-size: 0.85rem; transition: all 0.2s;">
                        <span style="color: var(--text); font-weight: 700;">📄 ${r.filename}</span>
                        <div style="text-align: right;">
                            <div style="color: var(--text-dim); font-size: 0.75rem;">${r.created}</div>
                            <small style="color: var(--cyan); font-size: 0.65rem;">${r.size_kb} KB</small>
                        </div>
                    </a>
                `).join('') : '<p style="font-size: 0.8rem; color: var(--text-dim); opacity: 0.5;">No reports archived.</p>';
        } catch(e) {}
    }

    showStatusMsg(el, text, color = null) {
        if (!el) return;
        el.textContent = text;
        if (color) el.style.color = color;
        setTimeout(() => el.textContent = '', 4000);
    }

    // ─── Toast helpers ───
    showToastOnce(key, title, message, variant = 'default', ttlMs = 4000) {
        if (this._toastActiveKeys.has(key)) return;
        this._toastActiveKeys.add(key);
        this.showToast(title, message, variant, ttlMs, () => {
            this._toastActiveKeys.delete(key);
        });
    }

    showToast(title, message, variant = 'default', ttlMs = 4000, onClose = null) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = 'toast' + (variant === 'warning' ? ' toast-warning' : variant === 'success' ? ' toast-success' : '');

        const body = document.createElement('div');
        body.className = 'toast-body';

        const titleEl = document.createElement('div');
        titleEl.className = 'toast-title';
        titleEl.textContent = title;

        const msgEl = document.createElement('div');
        msgEl.className = 'toast-message';
        msgEl.textContent = message;

        body.appendChild(titleEl);
        body.appendChild(msgEl);

        const closeBtn = document.createElement('button');
        closeBtn.className = 'toast-close';
        closeBtn.type = 'button';
        closeBtn.textContent = '✕';
        closeBtn.onclick = () => {
            toast.style.animation = 'toastOut 0.25s ease forwards';
            setTimeout(() => {
                toast.remove();
                if (onClose) onClose();
            }, 260);
        };

        toast.appendChild(body);
        toast.appendChild(closeBtn);
        container.appendChild(toast);

        setTimeout(() => {
            if (!document.body.contains(toast)) return;
            closeBtn.click();
        }, ttlMs);
    }
}

// Initial Launch
document.addEventListener('DOMContentLoaded', () => {
    window.hud = new VisionHUD();
});
