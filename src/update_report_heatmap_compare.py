from pathlib import Path

R = Path('/Users/William/.openclaw/workspace/projects/badminton-ai/reports')
index = R / 'index.html'
html = index.read_text()

compare_block = """
<h2>Shuttle Heatmap</h2>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>
  <div>
    <h3>Original</h3>
    <img src='shuttle_heatmap.png' alt='shuttle heatmap original'/>
  </div>
  <div>
    <h3>Denoised</h3>
    <img src='shuttle_heatmap_denoised.png' alt='shuttle heatmap denoised'/>
  </div>
</div>
"""

# replace first occurrence of old single heatmap section
old = """
<h2>Shuttle Heatmap</h2>
<img src='shuttle_heatmap.png' alt='shuttle heatmap'/>
"""
if old in html:
    html = html.replace(old, compare_block, 1)
else:
    # append if not found
    html += compare_block

index.write_text(html)
print('saved', index)
