# Fish Model Viewer

Double-click `start-viewer.command` to open the viewer. It runs on your Mac and does not upload the models.

Controls:

- W, A, S, D or arrow keys rotate the fish.
- Q tilts left and E tilts right.
- Z zooms out and X zooms in.
- Space pauses or starts auto-rotation.
- R resets the view.
- Drag any Gaussian-splat `.ply` or `.splat` file onto the viewer to open it.

The viewer reads the models in `../models` through the `public/models` link. Its menu includes the three September models, four representative earlier models, and four converted models from the first handheld attempts. The hull files are coloured point clouds rather than Gaussian-splat models, so they remain with the outputs but are not listed in this viewer.

W, A, S, and D orbit around the measured center of the loaded model using the current screen directions. W always moves toward the top of the screen and A always moves left, including after the view has been rolled. Q and E roll the view left or right without moving the fish away from the center.
