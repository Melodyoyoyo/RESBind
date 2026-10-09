from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


COLORS = {"metal": "#4C78A8", "nucleic_acid": "#7A5195", "small_molecule": "#E45756"}
LABELS = {"metal": "Metal", "nucleic_acid": "Nucleic acid", "small_molecule": "Small molecule"}


def prediction_figure(predictions: pd.DataFrame):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tasks = list(COLORS)
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for row, task in enumerate(tasks, 1):
        figure.add_trace(
            go.Scatter(
                x=predictions["position"], y=predictions[f"prob_{task}"], mode="lines",
                name=LABELS[task], line={"color": COLORS[task], "width": 1.8},
            ),
            row=row, col=1,
        )
        threshold = float(predictions[f"threshold_{task}"].iloc[0])
        figure.add_hline(y=threshold, line_dash="dot", line_color=COLORS[task], opacity=0.7, row=row, col=1)
        figure.update_yaxes(title_text="Probability", range=[0, 1.02], row=row, col=1)
    figure.update_xaxes(title_text="Residue position", row=3, col=1)
    figure.update_layout(height=620, template="simple_white", margin={"l": 60, "r": 20, "t": 25, "b": 50})
    return figure


def structure_html(pdb_path: Path, predictions: pd.DataFrame) -> str:
    pdb_text = Path(pdb_path).read_text(encoding="utf-8", errors="ignore")
    selections = {}
    for task, color in COLORS.items():
        positions = predictions.loc[predictions[f"pred_{task}"].astype(bool), "position"].astype(int).tolist()
        selections[task] = {"positions": positions, "color": color, "label": LABELS[task]}
    payload = json.dumps({"pdb": pdb_text, "selections": selections}).replace("</", "<\\/")
    legend = "".join(
        f'<span style="margin-right:16px"><i style="display:inline-block;width:10px;height:10px;background:{color};border-radius:50%;margin-right:5px"></i>{html.escape(LABELS[task])}</span>'
        for task, color in COLORS.items()
    )
    return f"""
<div style="font-family:Arial,sans-serif">
  <div id="resbind-viewer" style="height:560px;width:100%;position:relative"></div>
  <div style="padding:8px 2px;color:#403b38">{legend}</div>
</div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
(function() {{
  const data = {payload};
  const viewer = $3Dmol.createViewer(document.getElementById('resbind-viewer'), {{backgroundColor: 'white'}});
  viewer.addModel(data.pdb, 'pdb');
  viewer.setStyle({{}}, {{cartoon: {{color: '#D7D3D0'}}}});
  Object.values(data.selections).forEach(function(item) {{
    if (item.positions.length) {{
      viewer.addStyle({{resi: item.positions}}, {{stick: {{color: item.color, radius: 0.18}}, sphere: {{color: item.color, radius: 0.42}}}});
    }}
  }});
  viewer.zoomTo(); viewer.render();
}})();
</script>
"""

