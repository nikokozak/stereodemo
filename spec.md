# Project Specification: Stereo Demo with Syphon Input

## Goal

Integrate a Syphon input source into the existing stereo demo Python application. This will allow the application to process real-time stereo image pairs streamed from a separate OpenCV program via Syphon.

## Current State

The application (`stereodemo`) currently supports input from:
- Image files/directories (`FileListSource`)
- OAK-D camera (`OakdSource`)
- Syphon Source (`SyphonSource`)

Processing involves selecting a stereo depth estimation method and visualizing the results.

## Integration Plan

1.  **Create `SyphonSource`:**
    *   Define a new class `SyphonSource` in `stereodemo/syphon_source.py`.
    *   This class will implement the `visualizer.Source` interface.
    *   It will connect to the Syphon server(s) specified by the user.
    *   It will retrieve left and right images from the Syphon stream(s).
    *   It will handle calibration data (likely loaded from a file specified via CLI).
2.  **Modify `main.py`:**
    *   Add command-line arguments:
        *   `--syphon`: Flag to enable Syphon input.
        *   `--syphon-left-server`: Name of the Syphon server for the left camera.
        *   `--syphon-right-server`: Name of the Syphon server for the right camera. (Alternatively, handle single-server input if applicable).
    *   Update the source selection logic to instantiate `SyphonSource` when `--syphon` is used.
3.  **Add Dependencies:**
    *   Add `Syphon-python` to the project dependencies (`pyproject.toml` or `requirements.txt`).
4.  **Documentation & Testing:**
    *   Update README.
    *   Add usage examples.
    *   Test with the OpenCV streaming application.

## Open Questions

- How are left/right images streamed via Syphon (1 server vs. 2)?
- What are the Syphon server names?

## Milestones

- [x] Create `spec.md`
- [x] Define `visualizer.Source` interface requirements (by reading `visualizer.py`)
- [x] Implement basic `SyphonSource` class structure.
- [x] Add Syphon client logic to `SyphonSource`.
- [x] Add CLI arguments to `main.py`.
- [x] Integrate `SyphonSource` instantiation in `main.py`.
- [x] Add dependency management (`setup.cfg`).
- [x] Add simple retry mechanism for Syphon connection in `SyphonSource`.
- [x] Handle calibration loading for `SyphonSource`.
- [x] Add default automatic calibration for when no calibration file is provided.
- [x] Fix point cloud persistence issue (keep old cloud visible during processing).
- [x] Debug and fix segmentation fault related to rendering updates and intrinsic/image dimension mismatch.
- [ ] Test Syphon connection and frame retrieval thoroughly.
- [ ] Refine error handling.
- [ ] Update documentation.

## Recent Work: Rendering Fixes (Apr 7)

*   **Goal:** Prevent the point cloud display from clearing while the next frame is processing.
*   **Initial Problem:** The point cloud disappeared between frames because `_clear_outputs()` was called too early in `_process_input`.
*   **Attempt 1:** Modified `_update_rendering` to use `update_geometry` instead of `remove_geometry`/`add_geometry`. This caused `AttributeError`s related to camera clipping planes.
*   **Attempt 2:** Corrected camera API calls (`get_near`/`get_far`). Resulted in no point cloud rendering at all.
*   **Attempt 3:** Realized the point cloud *creation* logic (disparity -> depth -> RGBD -> PointCloud) was accidentally removed during refactoring. Re-added this logic into `_check_run_complete`.
*   **Attempt 4:** Restored UI buttons accidentally removed by the previous edit.
*   **Attempt 5 (Segfault Debug):** Added detailed logging to `_check_run_complete`. Logs revealed the crash occurred *after* point cloud creation but during/before `_update_rendering`.
*   **Root Cause:** Mismatch between image dimensions (e.g., HxW) and the dimensions used to initialize `o3d.camera.PinholeCameraIntrinsic` (e.g., WxH). 
*   **Fix:** Modified `_process_input` to create the `PinholeCameraIntrinsic` using width/height derived directly from the processed `input.left_image.shape`. 