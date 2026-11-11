# pvbatch script: render a colour field of an OpenFOAM case into PNG frames (one per written time).
# run through tools/film.sh (it supplies xvfb and ffmpeg); direct use:
#   xvfb-run -a pvbatch tools/film.py <case_dir> <field> <outdir> [vmin vmax [component [width height [colormap]]]]
# field: e.g. alpha.water, U, p ; component: Magnitude (default for vectors), X, Y, Z ; 2D cases are viewed along z
import sys, os
from paraview.simple import *

case, field, outdir = sys.argv[1:4]
vmin = float(sys.argv[4]) if len(sys.argv) > 4 else None
vmax = float(sys.argv[5]) if len(sys.argv) > 5 else None
comp = sys.argv[6] if len(sys.argv) > 6 else "Magnitude"
W = int(sys.argv[7]) if len(sys.argv) > 7 else 960
H = int(sys.argv[8]) if len(sys.argv) > 8 else 540
cmap = sys.argv[9] if len(sys.argv) > 9 else "Viridis (matplotlib)"
os.makedirs(outdir, exist_ok=True)
open(os.path.join(case, "case.foam"), "a").close()
r = OpenFOAMReader(FileName=os.path.join(case, "case.foam"))
r.MeshRegions = ["internalMesh"]
r.CellArrays = [field]
r.UpdatePipelineInformation()
times = [t for t in r.TimestepValues if t > 0] or list(r.TimestepValues)
view = CreateRenderView(); view.ViewSize = [W, H]; view.OrientationAxesVisibility = 0
view.Background = [1, 1, 1]
try: view.UseColorPaletteForBackground = 0
except Exception: pass
d = Show(r, view)
ColorBy(d, ("CELLS", field, comp) if comp != "Magnitude" and comp else ("CELLS", field, "Magnitude") if field == "U" else ("CELLS", field))
lut = GetColorTransferFunction(field)
lut.ApplyPreset(cmap, True)
if vmin is not None: lut.RescaleTransferFunction(vmin, vmax)
d.SetScalarBarVisibility(view, True)
bar = GetScalarBar(lut, view)
bar.Title = field; bar.ComponentTitle = "" if comp == "Magnitude" else comp
bar.TitleColor = [0, 0, 0]; bar.LabelColor = [0, 0, 0]          # black on the white background
bar.WindowLocation = "Any Location"; bar.Orientation = "Horizontal"; bar.Position = [0.30, 0.03]; bar.ScalarBarLength = 0.40
view.CameraParallelProjection = 1
Render(view)    # the first Render resets the camera on its own; set ours only after it
for i, t in enumerate(times):
    view.ViewTime = t
    r.UpdatePipeline(t)
    if i == 0:
        view.ResetCamera(); view.CameraPosition = [view.CameraFocalPoint[0], view.CameraFocalPoint[1], view.CameraFocalPoint[2] + 10]
        view.CameraViewUp = [0, 1, 0]
        b = r.GetDataInformation().GetBounds()                  # fit the domain to the frame (parallel projection)
        cx, cy, cz = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])
        view.CameraFocalPoint = [cx, cy, cz]; view.CameraPosition = [cx, cy, cz + 10]
        wd, ht = max(b[1] - b[0], 1e-12), max(b[3] - b[2], 1e-12)
        view.CameraParallelScale = 0.5 * max(ht * 1.25, wd * 1.08 * H / W)   # the extra height on top leaves room for the bar
    Render(view)
    SaveScreenshot(os.path.join(outdir, "frame_%04d.png" % i), view, ImageResolution=[W, H])
print("frames:", len(times))
