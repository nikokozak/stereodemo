# Project Specification: Stereo Demo with Syphon Input

## Goal

Integrate a Syphon input source into the existing stereo demo Python application. This will allow the application to process real-time stereo image pairs streamed from a separate OpenCV program via Syphon.

## Current State

The application (`stereodemo`) currently supports input from:
- Image files/directories (`FileListSource`)
- OAK-D camera (`OakdSource`)

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
- [ ] Test Syphon connection and frame retrieval.
- [x] Handle calibration loading for `SyphonSource`.
- [x] Add default automatic calibration for when no calibration file is provided.
- [ ] Refine error handling.
- [ ] Update documentation. 