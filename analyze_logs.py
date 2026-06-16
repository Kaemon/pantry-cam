"""
Analyze pantry and coffee logs.
Usage: python analyze_logs.py [date]   e.g. python analyze_logs.py 2026-06-11
       python analyze_logs.py          (uses today)
"""
import sys
import csv
import glob
from datetime import datetime, date
from collections import Counter, defaultdict

SESSION_GAP_SEC = 120   # visits < 2 min apart → same session (occlusion/track split)


def load_csv(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def fmt_dur(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def merge_sessions(rows, name_col, entry_col, exit_col, dur_col):
    """
    Merge consecutive visits by the same person where the gap is < SESSION_GAP_SEC.
    Returns a list of merged sessions as dicts with the same keys.
    """
    FMT = "%Y-%m-%d %H:%M:%S"
    by_name = defaultdict(list)
    for r in rows:
        by_name[r[name_col]].append(r)

    merged = []
    for name, visits in by_name.items():
        visits = sorted(visits, key=lambda r: r[entry_col])
        session_entry = datetime.strptime(visits[0][entry_col], FMT)
        session_exit  = datetime.strptime(visits[0][exit_col],  FMT)
        session_dur   = float(visits[0][dur_col])

        for v in visits[1:]:
            v_entry = datetime.strptime(v[entry_col], FMT)
            v_exit  = datetime.strptime(v[exit_col],  FMT)
            gap = (v_entry - session_exit).total_seconds()
            if gap < SESSION_GAP_SEC:
                session_exit = max(session_exit, v_exit)
                session_dur += float(v[dur_col])
            else:
                merged.append({name_col: name, entry_col: session_entry,
                                exit_col: session_exit, dur_col: session_dur})
                session_entry = v_entry
                session_exit  = v_exit
                session_dur   = float(v[dur_col])

        merged.append({name_col: name, entry_col: session_entry,
                       exit_col: session_exit, dur_col: session_dur})
    return merged


# ── Load ──────────────────────────────────────────────────────────────────────

date_filter = sys.argv[1] if len(sys.argv) > 1 else str(date.today())
pantry_rows = load_csv(f"pantry_log_{date_filter}*.csv")
coffee_rows = load_csv(f"coffee_log_{date_filter}*.csv")

if not pantry_rows and not coffee_rows:
    print(f"No logs found for {date_filter}")
    sys.exit(0)

# ── Pantry analysis ───────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"  PANTRY  ({date_filter})")
print(f"{'='*50}")

if not pantry_rows:
    print("  No data.")
else:
    sessions = merge_sessions(pantry_rows, "track_id", "entry_time", "exit_time", "duration_sec")
    named    = [s for s in sessions if not s["track_id"].startswith("#")]

    entry_hours = [s["entry_time"].hour for s in named]

    # Per-person totals (used throughout)
    person_totals = defaultdict(float)
    person_counts = defaultdict(int)
    for s in named:
        person_totals[s["track_id"]] += s["duration_sec"]
        person_counts[s["track_id"]] += 1

    avg_per_person = sum(person_totals.values()) / len(person_totals) if person_totals else 0

    print(f"\nRaw log entries : {len(pantry_rows)}")
    print(f"Sessions (merged): {len(named)}  (gap < {SESSION_GAP_SEC}s merged)")
    print(f"People seen     : {len(person_totals)}")
    print(f"Avg time/person : {fmt_dur(avg_per_person)}")
    print(f"Longest (total) : {fmt_dur(max(person_totals.values()))}")

    # Peak hours — unique people present during each hour
    hour_people = defaultdict(set)
    for s in named:
        h_start = s["entry_time"].hour
        h_end   = s["exit_time"].hour
        for h in range(h_start, h_end + 1):
            hour_people[h].add(s["track_id"])
    print(f"\nPeople by hour  :")
    for h in sorted(hour_people):
        n = len(hour_people[h])
        bar = "█" * n
        print(f"  {h:02d}:00  {bar}  ({n})")

    # Who spent the most total time
    print(f"\nTop visitors    :")
    for name, total in sorted(person_totals.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:<16} {fmt_dur(total):>10}   ({person_counts[name]} sessions)")

    # Avg duration per hour
    hour_durs = defaultdict(list)
    for s in named:
        hour_durs[s["entry_time"].hour].append(s["duration_sec"])
    print(f"\nAvg duration by hour:")
    for h in sorted(hour_durs):
        avg = sum(hour_durs[h]) / len(hour_durs[h])
        print(f"  {h:02d}:00  {fmt_dur(avg)}")

# ── Coffee analysis ───────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"  COFFEE MACHINE  ({date_filter})")
print(f"{'='*50}")

if not coffee_rows:
    print("  No data.")
else:
    c_durs  = [float(r["duration_sec"]) for r in coffee_rows]
    c_hours = [datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M:%S").hour
               for r in coffee_rows]

    print(f"\nTotal uses     : {len(coffee_rows)}")
    print(f"Avg duration   : {fmt_dur(sum(c_durs)/len(c_durs))}")
    print(f"Longest use    : {fmt_dur(max(c_durs))}")

    print(f"\nUses by hour   :")
    for h in sorted(Counter(c_hours)):
        bar = "█" * Counter(c_hours)[h]
        print(f"  {h:02d}:00  {bar}  ({Counter(c_hours)[h]})")

print()

# ── Timeline chart (interactive) ─────────────────────────────────────────────

try:
    import plotly.graph_objects as go
    import webbrowser, os

    unnamed = [s for s in sessions if s["track_id"].startswith("#")]
    unnamed_ids = sorted(set(s["track_id"] for s in unnamed),
                         key=lambda tid: min(s["entry_time"] for s in unnamed
                                             if s["track_id"] == tid))

    people = sorted(person_totals, key=lambda n: min(
        s["entry_time"] for s in named if s["track_id"] == n))

    TAB20 = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
        "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
    ]
    color_map = {n: TAB20[i % len(TAB20)] for i, n in enumerate(people)}

    # Scatter: one dot per session, x=midpoint time, y=person, size=duration
    traces = []

    all_sessions_for_chart = (
        [(s, color_map[s["track_id"]], s["track_id"]) for s in named] +
        [(s, "#aaaaaa", s["track_id"]) for s in unnamed]
    )

    for s, color, label in all_sessions_for_chart:
        mid   = s["entry_time"] + (s["exit_time"] - s["entry_time"]) / 2
        dur_s = s["duration_sec"]
        dur_str = fmt_dur(dur_s)
        size  = max(8, min(60, dur_s / 10))   # 10s→8px, 600s→60px
        total_str = fmt_dur(person_totals.get(label, dur_s))
        traces.append(go.Scatter(
            x=[mid],
            y=[f"{label}  (total {total_str})" if not label.startswith("#")
               else label],
            mode="markers",
            marker=dict(size=size, color=color, opacity=0.8,
                        line=dict(width=1, color="white")),
            hovertemplate=(f"<b>{label}</b><br>"
                           f"{s['entry_time'].strftime('%H:%M')} → "
                           f"{s['exit_time'].strftime('%H:%M')}<br>"
                           f"Duration: {dur_str}<extra></extra>"),
            showlegend=False,
        ))

    # Coffee dots
    if coffee_rows:
        for r in coffee_rows:
            t_start = datetime.strptime(r["start_time"], "%Y-%m-%d %H:%M:%S")
            t_end   = datetime.strptime(r["end_time"],   "%Y-%m-%d %H:%M:%S")
            dur_s   = float(r["duration_sec"])
            mid     = t_start + (t_end - t_start) / 2
            size    = max(8, min(60, dur_s / 10))
            traces.append(go.Scatter(
                x=[mid], y=["☕ coffee"],
                mode="markers",
                marker=dict(size=size, color="#6B3A2A", opacity=0.85,
                            line=dict(width=1, color="white")),
                hovertemplate=(f"<b>Coffee</b><br>"
                               f"{t_start.strftime('%H:%M')} → "
                               f"{t_end.strftime('%H:%M')}<br>"
                               f"Duration: {fmt_dur(dur_s)}<extra></extra>"),
                showlegend=False,
            ))

    n_rows = len(people) + len(unnamed_ids) + (1 if coffee_rows else 0)
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Pantry & Coffee — {date_filter}  (dot size = duration)",
        height=max(400, n_rows * 40 + 120),
        xaxis=dict(type="date", tickformat="%H:%M", title="Time of day"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200, r=20, t=50, b=50),
        plot_bgcolor="#f9f9f9",
    )

    out_path = os.path.abspath(f"pantry_timeline_{date_filter}.html")
    fig.write_html(out_path)
    webbrowser.open(f"file://{out_path}")
    print(f"Chart saved → {out_path}")

except ImportError:
    print("(install plotly to see interactive timeline:  pip install plotly)")
