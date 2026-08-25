from datetime import datetime


def generate_report_html(state):
    s = state

    def fmt_pct(x):
        return f"{x * 100:.2f}%" if x is not None else "—"

    attack_rows = ""
    for name, r in s.attack_results.items():
        attack_rows += f"""
        <tr>
          <td>{name}</td>
          <td>{fmt_pct(r.get('accuracy'))}</td>
          <td>{fmt_pct(r.get('ci_low'))} – {fmt_pct(r.get('ci_high'))}</td>
          <td>{r.get('n_samples', '—')}</td>
        </tr>"""

    plot_imgs = "".join(
        f'<h3>{name}</h3><img src="data:image/png;base64,{b64}" style="max-width:100%;margin-bottom:24px;">'
        for name, b64 in s.plots.items()
    )

    gradcam_imgs = "".join(
        f'<img src="data:image/png;base64,{g["image"]}" style="width:150px;margin:4px;" '
        f'title="true={g.get("true")} pred={g.get("pred")}">'
        for g in s.gradcam_results
    )

    metrics_html = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in s.metrics.items()) if s.metrics else ""

    stats_html = ""
    for k, v in (s.stats or {}).items():
        stats_html += f"<li><b>{k}</b>: statistic={v.get('statistic'):.4f}, p-value={v.get('p_value'):.4g}</li>"

    smoothing_html = ""
    if s.smoothing_result:
        sm = s.smoothing_result
        smoothing_html = (
            f"<li><b>Smoothed accuracy</b> (sigma={sm['sigma']}): {fmt_pct(sm['smoothed_accuracy'])}</li>"
            f"<li><b>Mean certified L2 radius</b>: {sm['mean_certified_radius']:.4f}</li>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Adversarial Robustness Report</title>
<style>
body{{font-family:Georgia,serif;max-width:900px;margin:40px auto;color:#1b2333;line-height:1.6;padding:0 20px;}}
h1{{border-bottom:3px solid #C9A227;padding-bottom:10px;}}
h2{{color:#0B1F3D;margin-top:2.2em;border-bottom:1px solid #ddd;padding-bottom:6px;}}
table{{border-collapse:collapse;width:100%;margin:1em 0;}}
th,td{{border:1px solid #ccc;padding:8px 10px;text-align:left;font-size:0.92em;}}
th{{background:#0B1F3D;color:#fff;}}
</style></head><body>
<h1>CNN Adversarial Robustness — Research Report</h1>
<p><i>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</i></p>

<h2>1. Setup</h2>
<ul>
  <li><b>Dataset:</b> {s.dataset_name} ({s.num_classes} classes, {s.img_size}x{s.img_size}, {s.in_channels}ch)</li>
  <li><b>Backbone:</b> {s.backbone_name}</li>
  <li><b>Architectural enhancement:</b> {s.enhancement}</li>
  <li><b>Device:</b> {s.device}</li>
  <li><b>Adversarial / robust training methods run:</b> {', '.join(s.adv_train_history.keys()) or '—'}</li>
</ul>

<h2>2. Model Complexity</h2>
<ul>{metrics_html or '<li>Not computed.</li>'}</ul>

<h2>3. Clean Accuracy</h2>
<p>{fmt_pct(s.clean_eval.get('accuracy')) if s.clean_eval else '—'}
 (95% CI: {fmt_pct(s.clean_eval.get('ci_low')) if s.clean_eval else '—'} – {fmt_pct(s.clean_eval.get('ci_high')) if s.clean_eval else '—'})</p>

<h2>4. Adversarial Accuracy by Attack</h2>
<table>
<tr><th>Attack</th><th>Accuracy</th><th>95% CI</th><th>N samples</th></tr>
{attack_rows or '<tr><td colspan="4">No attacks run yet.</td></tr>'}
</table>

<h2>5. Certified Robustness (Randomized Smoothing)</h2>
<ul>{smoothing_html or '<li>Not computed.</li>'}</ul>

<h2>6. Statistical Significance (McNemar, vs. clean)</h2>
<ul>{stats_html or '<li>Not computed.</li>'}</ul>

<h2>7. Plots</h2>
{plot_imgs or '<p>No plots generated yet.</p>'}

<h2>8. Grad-CAM Examples</h2>
<div>{gradcam_imgs or '<p>No Grad-CAM run yet.</p>'}</div>

</body></html>"""
    return html
