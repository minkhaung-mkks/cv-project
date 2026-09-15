# Fish Model Viewer

Double-click `start-viewer.command` to open the viewer. It runs on your Mac and does not upload the models.

Controls:

- W, A, S, D or arrow keys rotate the fish.
- Q tilts left and E tilts right.
- Z zooms out and X zooms in.
- Space pauses or starts auto-rotation.
- R resets the view.
- Drag any Gaussian-splat `.ply` or `.splat` file onto the viewer to open it.

The viewer reads the models in `../models` through the `public/models` link. Add an `<option>` in `index.html` if you add another model.
