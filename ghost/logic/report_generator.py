"""
report_generator.py — Beautiful Daily PDF Audit Reports
Generates professional energy-audit PDFs using fpdf2.
"""

import csv
import os
from datetime import datetime, timedelta
from fpdf import FPDF

# ── Constants ──
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
AUDIT_CSV = os.path.join(REPORTS_DIR, 'energy_audit.csv')

# Approximate cost per kWh in INR (Indian average ≈ ₹8)
COST_PER_KWH = 8.0
# Average room lighting power in kW
LIGHT_POWER_KW = 0.2


class AuditPDF(FPDF):
    """Custom PDF with branded header and footer."""

    def __init__(self, report_date, room_name):
        super().__init__()
        self.report_date = report_date
        self.room_name = room_name
        self.set_auto_page_break(auto=True, margin=25)

    # ── Header ──
    def header(self):
        # Top gradient bar (simulated with rectangles)
        self.set_fill_color(99, 102, 241)  # Indigo primary
        self.rect(0, 0, 210, 12, 'F')
        self.set_fill_color(79, 70, 229)   # Darker indigo
        self.rect(0, 12, 210, 2, 'F')

        # Brand name
        self.set_y(18)
        self.set_font('Helvetica', 'B', 22)
        self.set_text_color(40, 40, 60)
        self.cell(0, 10, 'VisionCore', new_x='LEFT', new_y='NEXT')

        # Subtitle
        self.set_font('Helvetica', '', 10)
        self.set_text_color(120, 120, 140)
        self.cell(0, 5, 'AI-Powered Energy Audit Report', new_x='LEFT', new_y='NEXT')

        # Date & Room
        self.set_font('Helvetica', '', 9)
        self.set_text_color(150, 150, 165)
        self.cell(0, 5,
                  f'Report Date: {self.report_date}   |   Room: {self.room_name}',
                  new_x='LEFT', new_y='NEXT')

        # Divider line
        self.set_draw_color(220, 220, 230)
        self.set_line_width(0.3)
        self.line(10, self.get_y() + 3, 200, self.get_y() + 3)
        self.ln(8)

    # ── Footer ──
    def footer(self):
        self.set_y(-20)
        # Thin line
        self.set_draw_color(220, 220, 230)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_font('Helvetica', '', 7)
        self.set_text_color(160, 160, 175)
        self.cell(0, 5,
                  f'VisionCore Lab Monitor  |  Auto-generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}  |  Page {self.page_no()}/{{nb}}',
                  align='C')


def _read_audit_entries(target_date=None):
    """Read audit CSV and filter to a specific date (default: today)."""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')

    entries = []
    if not os.path.exists(AUDIT_CSV):
        return entries

    with open(AUDIT_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get('Timestamp', '')
            if ts.startswith(target_date):
                entries.append(row)
    return entries


def _compute_stats(entries):
    """Compute summary statistics from audit entries."""
    total_waste_seconds = sum(float(e.get('Duration_Seconds', 0)) for e in entries)
    total_waste_hours = total_waste_seconds / 3600
    total_alerts = len(entries)

    # Energy saved = waste hours × lighting power
    energy_saved_kwh = total_waste_hours * LIGHT_POWER_KW
    money_saved = energy_saved_kwh * COST_PER_KWH

    # Operating hours (rough: span from first to last alert)
    presence_hours = 0.0
    if len(entries) >= 2:
        try:
            first = datetime.strptime(entries[0]['Timestamp'], '%Y-%m-%d %H:%M:%S')
            last = datetime.strptime(entries[-1]['Timestamp'], '%Y-%m-%d %H:%M:%S')
            span = (last - first).total_seconds() / 3600
            presence_hours = max(0, span - total_waste_hours)
        except Exception:
            pass

    return {
        'total_alerts': total_alerts,
        'waste_seconds': total_waste_seconds,
        'waste_hours': round(total_waste_hours, 2),
        'presence_hours': round(presence_hours, 2),
        'energy_saved_kwh': round(energy_saved_kwh, 3),
        'money_saved': round(money_saved, 2),
    }


def _draw_summary_card(pdf, x, y, w, h, label, value, unit, r, g, b):
    """Draw a rounded summary card with a colored accent."""
    # Card background
    pdf.set_fill_color(248, 249, 252)
    pdf.rect(x, y, w, h, 'F')

    # Left accent bar
    pdf.set_fill_color(r, g, b)
    pdf.rect(x, y, 3, h, 'F')

    # Label
    pdf.set_xy(x + 8, y + 4)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(130, 130, 145)
    pdf.cell(w - 10, 5, label.upper())

    # Value
    pdf.set_xy(x + 8, y + 12)
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(w - 10, 10, str(value))

    # Unit
    pdf.set_xy(x + 8, y + 25)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(150, 150, 165)
    pdf.cell(w - 10, 5, unit)


def generate_daily_report(target_date=None, room_name=None):
    """
    Generate a daily PDF audit report.

    Args:
        target_date: Date string 'YYYY-MM-DD'. Defaults to today.
        room_name: Room name to display. Defaults to config value.

    Returns:
        str: Path to the generated PDF file.
    """
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        import config as cfg
    except ImportError:
        cfg = None

    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    if room_name is None:
        room_name = getattr(cfg, 'ROOM_NAME', 'Lab Room') if cfg else 'Lab Room'

    entries = _read_audit_entries(target_date)
    stats = _compute_stats(entries)

    # ── Create PDF ──
    pdf = AuditPDF(report_date=target_date, room_name=room_name)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Time formatting helper
    def fmt_time(hours_val):
        h = int(hours_val)
        m = int((hours_val - h) * 60)
        if h > 0:
            return f"{h}h {m}m"
        return f"{m}m"

    waste_str = fmt_time(stats['waste_hours'])
    presence_str = fmt_time(stats.get('presence_hours', 0))

    # ── Section: Executive Summary ──
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 8, 'Executive Summary', new_x='LEFT', new_y='NEXT')
    pdf.ln(2)

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(100, 100, 115)
    pdf.multi_cell(0, 5,
        f'This report summarizes the energy monitoring activity for {room_name} on {target_date}. '
        f'The AI system detected {stats["total_alerts"]} energy waste events and tracked '
        f'{waste_str} of unnecessary lighting.')
    pdf.ln(6)

    # ── Summary Cards (3-column row) ──
    card_y = pdf.get_y()
    card_w = 60
    card_h = 34
    gap = 5

    _draw_summary_card(pdf, 10, card_y, card_w, card_h,
                       'Total Alerts', stats['total_alerts'], 'waste events detected',
                       239, 68, 68)      # Red

    _draw_summary_card(pdf, 10 + card_w + gap, card_y, card_w, card_h,
                       'Lights Wasted', waste_str, 'time of unused lighting',
                       245, 158, 11)     # Amber

    _draw_summary_card(pdf, 10 + 2 * (card_w + gap), card_y, card_w, card_h,
                       'Money Saved', f"Rs.{stats['money_saved']}", f'{stats["energy_saved_kwh"]} kWh saved',
                       16, 185, 129)     # Green

    pdf.set_y(card_y + card_h + 10)

    # ── Second Row of Cards ──
    card_y2 = pdf.get_y()

    _draw_summary_card(pdf, 10, card_y2, card_w, card_h,
                       'Presence Hours', presence_str, 'humans detected in room',
                       99, 102, 241)     # Indigo

    _draw_summary_card(pdf, 10 + card_w + gap, card_y2, card_w, card_h,
                       'Energy Saved', f"{stats['energy_saved_kwh']}", 'kilowatt-hours',
                       34, 211, 238)     # Cyan

    _draw_summary_card(pdf, 10 + 2 * (card_w + gap), card_y2, card_w, card_h,
                       'Avg Waste/Event',
                       f"{round(stats['waste_seconds'] / max(stats['total_alerts'], 1), 1)}s",
                       'seconds per alert',
                       139, 92, 246)     # Purple

    pdf.set_y(card_y2 + card_h + 12)

    # ── Section: Detailed Alert Log ──
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 8, 'Detailed Alert Log', new_x='LEFT', new_y='NEXT')
    pdf.ln(2)

    if entries:
        # Table header
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(99, 102, 241)
        pdf.set_text_color(255, 255, 255)
        col_widths = [12, 50, 55, 38, 35]
        headers = ['#', 'Timestamp', 'Room', 'Duration', 'Status']
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 8, h, border=0, fill=True, align='C')
        pdf.ln()

        # Table rows
        pdf.set_font('Helvetica', '', 8)
        for idx, entry in enumerate(entries):
            # Alternate row colors
            if idx % 2 == 0:
                pdf.set_fill_color(248, 249, 252)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(60, 60, 75)

            dur_s = float(entry.get('Duration_Seconds', 0))
            dur_str = f"{dur_s:.1f}s" if dur_s < 60 else f"{dur_s / 60:.1f}m"

            pdf.cell(col_widths[0], 7, str(idx + 1), border=0, fill=True, align='C')
            pdf.cell(col_widths[1], 7, entry.get('Timestamp', ''), border=0, fill=True, align='C')
            pdf.cell(col_widths[2], 7, entry.get('Room', ''), border=0, fill=True, align='C')
            pdf.cell(col_widths[3], 7, dur_str, border=0, fill=True, align='C')

            # Status badge color
            status = entry.get('Status', '')
            if status == 'ALERT_SENT':
                pdf.set_text_color(239, 68, 68)
            else:
                pdf.set_text_color(16, 185, 129)
            pdf.cell(col_widths[4], 7, status, border=0, fill=True, align='C')
            pdf.set_text_color(60, 60, 75)
            pdf.ln()
    else:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(150, 150, 165)
        pdf.cell(0, 10, 'No energy waste events were recorded on this date.', align='C',
                 new_x='LEFT', new_y='NEXT')

    pdf.ln(8)

    # ── Section: Recommendations ──
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(0, 8, 'AI Recommendations', new_x='LEFT', new_y='NEXT')
    pdf.ln(2)

    recommendations = []
    if stats['total_alerts'] > 10:
        recommendations.append(
            'High alert frequency detected. Consider installing motion-activated lighting '
            'or timer-based controls to reduce manual dependency.')
    if stats['waste_hours'] > 2:
        recommendations.append(
            f'Over {stats["waste_hours"]} hours of light wasted. '
            'Review room scheduling and occupancy patterns.')
    if stats['total_alerts'] == 0:
        recommendations.append(
            'Excellent! No energy waste detected today. The room is being used efficiently.')
    if not recommendations:
        recommendations.append(
            'Energy usage is within acceptable limits. Continue monitoring for deviations.')

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(80, 80, 95)
    for i, rec in enumerate(recommendations, 1):
        pdf.set_fill_color(248, 249, 252)
        x0 = pdf.get_x()
        y0 = pdf.get_y()
        # Bullet background
        pdf.rect(10, y0, 190, 12, 'F')
        pdf.set_fill_color(99, 102, 241)
        pdf.rect(10, y0, 3, 12, 'F')
        pdf.set_xy(16, y0 + 2)
        pdf.multi_cell(180, 4, f'{i}. {rec}')
        pdf.ln(2)

    # ── Save PDF ──
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = f'energy_report_{target_date}.pdf'
    filepath = os.path.join(REPORTS_DIR, filename)
    pdf.output(filepath)
    print(f"[REPORT] PDF generated: {filepath}")
    return filepath


if __name__ == '__main__':
    # Quick test: generate today's report
    path = generate_daily_report()
    print(f"Report saved to: {path}")
