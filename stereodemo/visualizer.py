import copy
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .methods import IntParameter, EnumParameter, StereoOutput, StereoMethod, Calibration, InputPair
   
disparity_window = None

class ImageWindow:
    def __init__(self, name: str, size: Tuple[int, int]):
        self.name = name
        self.window = gui.Application.instance.create_window(name, size[0], size[1])
        self.image_widget = gui.ImageWidget()
        self.window.add_child(self.image_widget)

    def update_image(self, image: np.ndarray):
        image_geom = o3d.geometry.Image(image)
        self.image_widget.update_image(image_geom)
        self.window.post_redraw()

class ImageWindowsManager:
    def __init__(self):
        self.windows_by_name = {}

    def imshow(self, name: str, image: np.ndarray, window_title: Optional[str], max_size: int):
        if name not in self.windows_by_name:
            rows, cols, _ = image.shape
            if cols > rows:
                initial_size = max_size, int(max_size * rows / cols)            
            else:
                initial_size = int(max_size * cols / rows), max_size
            self.windows_by_name[name] = ImageWindow(name, initial_size)
        self.windows_by_name[name].update_image(image)
        if window_title is not None:
            self.windows_by_name[name].title = window_title

image_windows_manager = ImageWindowsManager()

def imshow (name: str, image: np.ndarray, window_title=None, max_size=640):
    global image_windows_manager
    if image_windows_manager is None:
        image_windows_manager = ImageWindowsManager()
    image_windows_manager.imshow(name, image, window_title, max_size)

def color_disparity (disparity_map: np.ndarray, calibration: Calibration):
    min_disp = (calibration.fx * calibration.baseline_meters) / calibration.depth_range[1]
    # disparity_pixels = (calibration.fx * calibration.baseline_meters) / depth_meters
    max_disp = (calibration.fx * calibration.baseline_meters) / calibration.depth_range[0]
    norm_disparity_map = 255*((disparity_map-min_disp) / (max_disp-min_disp))
    disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_VIRIDIS)
    return disparity_color

def show_color_disparity (name: str, color_disparity: np.ndarray):
    imshow ("StereoDemo - Disparity", color_disparity, name)

class Settings:
    def __init__(self):
        self.show_axes = False

class Source:
    def __init__(self):
        pass

    @abstractmethod
    def is_live(self) -> bool:
        """Whether the source is capture live images or not"""
        return False

    def selected_index (self) -> int:
        return 0

    @abstractmethod
    def get_next_pair(self) -> InputPair:
        return InputPair(None, None, None, None)

    def get_pair_at_index(self, idx: int) -> InputPair:
        return InputPair(None, None, None, None)

    def get_pair_list(self) -> List[str]:
        return []

class Visualizer:
    def __init__(self, stereo_methods: Dict[str, StereoMethod], source: Source):
        gui.Application.instance.initialize()

        self.vis = gui.Application.instance
        self.source = source

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor_future = None
        self.is_live_processing_active = False

        self.stereo_methods = stereo_methods
        self.stereo_methods_output = {}
        self.input = InputPair (None, None, None, None)
        self._downsample_factor = 0

        self.window = gui.Application.instance.create_window("StereoDemo", 1280, 1024)
        w = self.window  # to make the code more concise

        self.settings = Settings()

        # 3D widget
        self._scene = gui.SceneWidget()        
        self._scene.scene = rendering.Open3DScene(w.renderer)
        # self._scene.scene.show_ground_plane(True, rendering.Scene.GroundPlane.XZ)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self._scene.set_on_key(self._on_key_pressed)

        self._clear_outputs ()

        for name, o in self.stereo_methods_output.items():
            if o.point_cloud is not None:
                self._scene.scene.add_geometry(name, o.point_cloud, rendering.MaterialRecord())

        self._reset_camera()

        em = w.theme.font_size
        self.separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        controls_horiz = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
        self._next_image_button = gui.Button("Next Image")
        self._next_image_button.set_on_clicked(self._next_image_clicked)
        controls_horiz.add_child(self._next_image_button)

        controls_horiz.add_stretch()

        self._live_button = gui.Button("Start Live")
        self._live_button.set_on_clicked(self._toggle_live_mode)
        controls_horiz.add_child(self._live_button)
        # Add the horizontal layout containing buttons TO the settings panel
        self._settings_panel.add_child(controls_horiz)

        if self.source.is_live():
            self._next_image_button.enabled = True
            self._live_button.visible = True
        else:
            self._next_image_button.enabled = True
            self._live_button.visible = False

        if not self.source.is_live():
            self._settings_panel.add_fixed(self.separation_height)
            self.images_combo = gui.Combobox()
            input_pairs = self.source.get_pair_list()
            for pair_name in input_pairs:
                self.images_combo.add_item(pair_name)
            self.images_combo.selected_index = 0
            self.images_combo.set_on_selection_changed(self._image_selected)
            self._settings_panel.add_child(self.images_combo)
            self._settings_panel.add_fixed(self.separation_height)
        else:
            self.images_combo = None
        
        horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        label = gui.Label("Input downsampling")
        label.tooltip = "Number of /2 downsampling steps to apply on the input"
        horiz.add_child(label)
        downsampling_slider = gui.Slider(gui.Slider.INT)
        downsampling_slider.set_limits(0, 4)
        downsampling_slider.int_value = self._downsample_factor
        downsampling_slider.set_on_value_changed(self._downsampling_changed)
        horiz.add_child(downsampling_slider)
        self._settings_panel.add_child(horiz)

        self._settings_panel.add_fixed(self.separation_height)

        self.algo_list = gui.ListView()
        self.algo_list.set_items(list(stereo_methods.keys()))
        self.algo_list.selected_index = 0
        self.algo_list.set_max_visible_items(8)
        self.algo_list.set_on_selection_changed(self._on_algo_list_selected)
        self._settings_panel.add_child(self.algo_list)

        self.method_params_proxy = gui.WidgetProxy()
        self._settings_panel.add_child (self.method_params_proxy)

        self.last_runtime = gui.Label("")
        self._settings_panel.add_child (self.last_runtime)

        self.input_status = gui.Label("No input.")
        self._settings_panel.add_child (self.input_status)

        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em, gui.Margins(em, 0, 0, 0))
        reset_cam_button = gui.Button("Reset Camera")
        reset_cam_button.set_on_clicked(self._reset_camera)
        view_ctrls.add_child(reset_cam_button)
        
        horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        label = gui.Label("Max depth (m)")
        label.tooltip = "Max depth to render in meters"
        horiz.add_child(label)
        self.depth_range_slider = gui.Slider(gui.Slider.DOUBLE)
        self.depth_range_slider.set_limits(0.5, 1000)
        self.depth_range_slider.double_value = 100
        self.depth_range_slider.set_on_value_changed(self._depth_range_slider_changed)
        horiz.add_child(self.depth_range_slider)
        view_ctrls.add_child(horiz)

        self._depth_range_manually_changed = False
        
        self._settings_panel.add_fixed(self.separation_height)
        self._settings_panel.add_child(view_ctrls)

        self._settings_panel.add_stretch()
        self._processing_status_label = gui.Label("Idle")
        self._settings_panel.add_child(self._processing_status_label)

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        
        self._on_algo_list_selected(self.algo_list.selected_value, False)
        self._apply_settings()

        if not self.source.is_live():
            self._image_selected(None, None)

    def _on_key_pressed (self, keyEvent):
        if keyEvent.key == gui.KeyName.Q:
            self.vis.quit()
            return gui.SceneWidget.EventCallbackResult.HANDLED
        return gui.SceneWidget.EventCallbackResult.IGNORED
    
    def _downsampling_changed(self, v):
        self._downsample_factor = int(v)
        self._process_input (self.full_res_input)

    def _downsample_input (self, input: InputPair):
        for i in range(0, self._downsample_factor):
            if np.max(input.left_image.shape[:2]) < 250:
                break
            input.left_image = cv2.pyrDown(input.left_image)
            input.right_image = cv2.pyrDown(input.right_image)
            if input.input_disparity is not None:
                input.input_disparity = cv2.pyrDown(input.input_disparity)
            input.calibration.downsample(input.left_image.shape[1], input.left_image.shape[0])

    def read_next_pair (self):
        if self.executor_future is not None and not self.executor_future.done():
             print("DEBUG: read_next_pair called while processing, skipping fetch.")
             return # Don't fetch a new pair if the previous one isn't done

        print("DEBUG: Attempting to read next pair from source...")
        self._processing_status_label.text = "Fetching frame..." # Update status
        try:
            input_pair = self.source.get_next_pair()
            self._update_pair_index() # Update dropdown if needed
            self._processing_status_label.text = "Processing frame..." # Update status
            self._process_input(input_pair)
        except Exception as e:
             print(f"ERROR: Failed to get or process next pair: {e}")
             import traceback
             traceback.print_exc()
             self._processing_status_label.text = "Error fetching frame!"
             # Optionally stop live mode on error
             if self.is_live_processing_active:
                self._toggle_live_mode()

    def _process_input (self, input):
        print(f"DEBUG Visualizer: Processing input with shape left={input.left_image.shape if input.left_image is not None else None}, right={input.right_image.shape if input.right_image is not None else None}")
        
        if self._downsample_factor > 0:
            self.full_res_input = input
            input = copy.deepcopy(input)
            self._downsample_input (input)
        else:
            self.full_res_input = input

        if not self._depth_range_manually_changed:
            self.depth_range_slider.double_value = input.calibration.depth_range[1]

        if input.left_image is None or input.right_image is None:
            print("DEBUG Visualizer: Input images are None!")
            return

        if not input.has_data():
            print("DEBUG Visualizer: Input reports no data!")
            return

        # Create Open3D camera intrinsics from Calibration
        # *** Use the actual image dimensions AFTER downsampling ***
        img_height = input.left_image.shape[0]
        img_width = input.left_image.shape[1]
        self.o3dCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(img_width),       # Use actual width
            int(img_height),      # Use actual height
            input.calibration.fx, # Adjust fx, fy, cx, cy based on downsampling? 
            input.calibration.fy, # Calibration object should handle this via .downsample()
            input.calibration.cx0,
            input.calibration.cy
        )
        print(f"DEBUG Visualizer: Updated o3dCameraIntrinsic using image shape: w={self.o3dCameraIntrinsic.width}, h={self.o3dCameraIntrinsic.height}, fx={self.o3dCameraIntrinsic.intrinsic_matrix[0,0]}")

        # Create a copy of the images to avoid modifying the originals
        left_display = input.left_image.copy()
        right_display = input.right_image.copy()
        
        # Convert to RGB for Open3D display
        if len(left_display.shape) == 2:  # Grayscale
            left_display = cv2.cvtColor(left_display, cv2.COLOR_GRAY2RGB)
            right_display = cv2.cvtColor(right_display, cv2.COLOR_GRAY2RGB)
        elif len(left_display.shape) == 3 and left_display.shape[2] == 3:  # BGR input
            left_display = cv2.cvtColor(left_display, cv2.COLOR_BGR2RGB)
            right_display = cv2.cvtColor(right_display, cv2.COLOR_BGR2RGB)

        print(f"DEBUG Visualizer: Displaying combined image with shape {left_display.shape}")
        combined_image = np.hstack([left_display, right_display])
        imshow("StereoDemo - Input image", combined_image)  # Open3D GUI expects RGB
        
        self.input = input
        self.input_status.text = f"Input: {input.left_image.shape[1]}x{input.left_image.shape[0]} " + input.status

        if self.input.has_data():
            print("DEBUG Visualizer: Input has data, proceeding with processing")
            assert self.input.left_image.shape[1] == self.input.calibration.width and self.input.left_image.shape[0] == self.input.calibration.height
            self._run_current_method()

    def update_once (self):
        if self.executor_future is not None:
            self._check_run_complete()
        return gui.Application.instance.run_one_tick()

    def _clear_outputs (self):
        for name in self.stereo_methods.keys():
            self.stereo_methods_output[name] = StereoOutput(
                disparity_pixels=None,
                color_image_bgr=None,
                computation_time=np.nan)
            if self._scene.scene.has_geometry(name):
                self._scene.scene.remove_geometry(name)

    def _reset_camera (self):
        # bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-10, 0,-10]), np.array([0,3,0]))
        bbox = self._scene.scene.bounding_box
        min_bound, max_bound = bbox.min_bound.copy(), bbox.max_bound.copy()
        min_bound[0] = min(min_bound[0], -5)
        min_bound[2] = min(min_bound[2], -5)
        max_bound[0] = max(max_bound[0],  5)
        max_bound[1] = max(max_bound[1],  2)
        max_bound[2] = 0
        bbox.min_bound, bbox.max_bound = min_bound, max_bound

        self._scene.setup_camera(60.0, bbox, np.array([0,0,0]))
        eye = np.array([0, 0.5,  1.0])
        lookat = np.array([0, 0, -1.0])
        up = np.array([0, 1.0, 0])
        self._scene.look_at(lookat, eye, up)

        if self.input.has_data():
            self._depth_range_manually_changed = False
            self.depth_range_slider.double_value = self.input.calibration.depth_range[1]
            self._update_rendering ()

    def _build_stereo_method_widgets(self, name):
        em = self.window.theme.font_size
        method = self.stereo_methods[name]
        container = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        label = gui.Label(method.description)
        label.text_color = gui.Color(1.0, 0.5, 0.0)
        container.add_child(label)
        self._reload_settings_functions = []
        for name, param in method.parameters.items():
            if isinstance(param, IntParameter):
                horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
                label = gui.Label(name)
                label.tooltip = param.description
                horiz.add_child(label)
                slider = gui.Slider(gui.Slider.INT)
                slider.set_limits(param.min, param.max)
                slider.int_value = param.value
                def set_value_from_method(slider=slider, method=method, name=name):
                    slider.int_value = method.parameters[name].value
                self._reload_settings_functions.append(set_value_from_method)
                # workaround late binding
                # https://docs.python-guide.org/writing/gotchas/#:~:text=Python's%20closures%20are%20late%20binding,surrounding%20scope%20at%20call%20time.
                def callback(value, method=method, name=name, slider=slider):
                    p = method.parameters[name]
                    p.set_value(int(value))
                    slider.int_value = p.value
                slider.set_on_value_changed(callback)
                horiz.add_child(slider)
                container.add_child(horiz)
            elif isinstance(param, EnumParameter):
                horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
                label = gui.Label(name)
                label.tooltip = param.description
                horiz.add_child(label)
                combo = gui.Combobox()
                for value in param.values:
                    combo.add_item(value)
                combo.selected_index = param.index
                def callback(combo_idx, combo_val, method=method, name=name, combo=combo):
                    method.parameters[name].set_index(combo.selected_index)
                combo.set_on_selection_changed(callback)
                def set_value_from_method(combo=combo, method=method, name=name):
                    combo.selected_index = method.parameters[name].index
                self._reload_settings_functions.append(set_value_from_method)
                horiz.add_child(combo)
                container.add_child(horiz)
            
        horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))        
        apply_button = gui.Button("Apply")
        apply_button.horizontal_padding_em = 3
        apply_button.set_on_clicked(self._run_current_method)
        horiz.add_child(apply_button)
        horiz.add_fixed(self.separation_height)
        reset_default = gui.Button("Reset defaults")            
        reset_default.set_on_clicked(self._reset_method_defaults)
        horiz.add_child(reset_default)
        container.add_child(horiz)
        return container

    def _on_algo_list_selected(self, name: str, is_dbl_click: bool):
        print(f"DEBUG: Method selected: {name}")
        self.method_params_proxy.set_widget(self._build_stereo_method_widgets(name))
        self._update_runtime()
        
        print("DEBUG: Updating geometry visibility")
        for other_name in self.stereo_methods_output.keys():
            print(f"DEBUG: Setting {other_name} visibility to False")
            self._scene.scene.show_geometry(other_name, False)
        
        print(f"DEBUG: Setting {name} visibility to True")
        self._scene.scene.show_geometry(name, True)
        
        self._apply_settings()
        
        if self.stereo_methods_output[name].disparity_pixels is None:
            print(f"DEBUG: No disparity map for {name}, running method")
            self._run_current_method()
        
        if self.stereo_methods_output[name].disparity_color is not None:
            print(f"DEBUG: Showing color disparity for {name}")
            show_color_disparity(name, self.stereo_methods_output[name].disparity_color)

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _next_image_clicked(self):
        if not self.is_live_processing_active: # Only allow if not in live mode
             print("DEBUG: Next image button clicked.")
             self.read_next_pair()
        else:
             print("DEBUG: Next image button ignored (live mode active).")

    def _image_selected(self, combo_idx, combo_val):
        idx = self.images_combo.selected_index
        input = self.source.get_pair_at_index (idx)
        self._process_input (input)

    def _update_pair_index (self):
        if self.images_combo is not None:
            self.images_combo.selected_index = self.source.selected_index()

    def _apply_settings(self):
        self._scene.scene.show_axes(self.settings.show_axes)

    def _reset_method_defaults(self):
        name = self.algo_list.selected_value
        method = self.stereo_methods[name]
        method.reset_defaults()
        for m in self._reload_settings_functions:
            m()

    def _check_run_complete(self):
        if self.executor_future is None:
             return # Nothing running

        if not self.executor_future.done():
            # Optionally update status label here if you want a "Processing..." indicator
            # self._processing_status_label.text = "Processing..." # Can be verbose
            return

        # --- Processing is done ---
        print("DEBUG: Retrieving stereo computation result")
        try:
            stereo_output = self.executor_future.result()
        except Exception as e:
             print(f"ERROR: Stereo computation failed: {e}")
             import traceback
             traceback.print_exc()
             self._processing_status_label.text = "Error!"
             self.executor_future = None # Reset future on error
             # Stop live mode on error? Maybe desirable.
             if self.is_live_processing_active:
                 self._toggle_live_mode() # Turn off live mode
             return

        self.executor_future = None # Clear the future

        name = self.algo_list.selected_value
        
        # --- Point Cloud Creation --- 
        # Create the Open3D PointCloud object here from the computed disparity
        if stereo_output is not None and stereo_output.disparity_pixels is not None:
            print("DEBUG [PCD Create]: Start - Have disparity map")
            try:
                # Calculate depth from disparity
                print("DEBUG [PCD Create]: Calculating depth from disparity...")
                depth_meters = StereoMethod.depth_meters_from_disparity(
                    stereo_output.disparity_pixels, self.input.calibration
                )
                print(f"DEBUG [PCD Create]: Depth calculated. Shape: {depth_meters.shape}, dtype: {depth_meters.dtype}")

                # Clip depth based on slider
                print("DEBUG [PCD Create]: Clipping depth...")
                max_depth = self.depth_range_slider.double_value
                depth_meters[depth_meters > max_depth] = 0 # Clip far points
                depth_meters[depth_meters < 0] = 0      # Clip negative/invalid points
                depth_meters[np.isnan(depth_meters)] = 0
                depth_meters[np.isinf(depth_meters)] = 0
                print(f"DEBUG [PCD Create]: Depth clipped. min: {np.min(depth_meters)}, max: {np.max(depth_meters)}")

                # Get color image (use input left if method didn't provide one)
                print("DEBUG [PCD Create]: Getting color image...")
                if stereo_output.color_image_bgr is None:
                    color_image = self.input.left_image.copy()
                    print("DEBUG [PCD Create]: Using input left image as color.")
                else:
                    color_image = stereo_output.color_image_bgr
                    print("DEBUG [PCD Create]: Using method's output color image.")
                print(f"DEBUG [PCD Create]: Color image shape: {color_image.shape}, dtype: {color_image.dtype}")
                
                # Ensure color is RGB for Open3D
                print("DEBUG [PCD Create]: Converting color to RGB...")
                if len(color_image.shape) == 2: # Grayscale to RGB
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
                else: # BGR to RGB
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                o3d_color_np = color_image
                print(f"DEBUG [PCD Create]: Creating o3d.geometry.Image for color (shape: {o3d_color_np.shape}, dtype: {o3d_color_np.dtype})...")
                o3d_color = o3d.geometry.Image(o3d_color_np)
                print("DEBUG [PCD Create]: o3d color image created.")
                
                print("DEBUG [PCD Create]: Creating o3d.geometry.Image for depth...")
                o3d_depth_np = depth_meters.astype(np.float32)
                print(f"DEBUG [PCD Create]: Creating o3d.geometry.Image for depth (shape: {o3d_depth_np.shape}, dtype: {o3d_depth_np.dtype})...")
                o3d_depth = o3d.geometry.Image(o3d_depth_np)
                print("DEBUG [PCD Create]: o3d depth image created.")

                # Create RGBD image
                print("DEBUG [PCD Create]: Creating RGBD image...")
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color,
                    o3d_depth,
                    depth_scale=1.0, # Depth is already in meters
                    depth_trunc=max_depth, # Use slider value for truncation
                    convert_rgb_to_intensity=False
                )
                print("DEBUG [PCD Create]: RGBD image created.")

                # Create PointCloud from RGBD
                print("DEBUG [PCD Create]: Checking o3dCameraIntrinsic...")
                if self.o3dCameraIntrinsic is None:
                    print("ERROR [PCD Create]: o3dCameraIntrinsic not set, cannot create point cloud!")
                    stereo_output.point_cloud = None
                else:
                    print(f"DEBUG [PCD Create]: o3dCameraIntrinsic found: w={self.o3dCameraIntrinsic.width} h={self.o3dCameraIntrinsic.height} fx={self.o3dCameraIntrinsic.intrinsic_matrix[0,0]}")
                    print("DEBUG [PCD Create]: Creating PointCloud from RGBD...")
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd, self.o3dCameraIntrinsic
                    )
                    print(f"DEBUG [PCD Create]: PointCloud created initially with {len(pcd.points)} points.")
                    
                    # Apply standard transform
                    print("DEBUG [PCD Create]: Applying transform...")
                    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                    print("DEBUG [PCD Create]: Transform applied.")
                    
                    stereo_output.point_cloud = pcd # Assign to the output object
                    print(f"DEBUG [PCD Create]: Point cloud assigned to output. Final points: {len(pcd.points)}.")

            except Exception as e:
                print(f"ERROR [PCD Create]: Failed during point cloud creation: {e}")
                import traceback
                traceback.print_exc()
                stereo_output.point_cloud = None # Ensure it's None on error
        else:
            print("DEBUG [PCD Create]: Skip - No valid stereo output or disparity map.")
            if stereo_output:
                 stereo_output.point_cloud = None # Ensure it's None if disparity was missing
        # --- End Point Cloud Creation ---

        self.stereo_methods_output[name] = stereo_output
        self._update_runtime() # Update time label

        print(f"DEBUG: Computation for {name} finished. Updating rendering.")
        self._update_rendering([name]) # Update only the completed method's geometry

        # Show disparity for the completed method
        if self.stereo_methods_output[name].disparity_color is not None:
            show_color_disparity(name, self.stereo_methods_output[name].disparity_color)

        # --- Check if we need to loop in Live mode ---
        if self.is_live_processing_active:
            print("DEBUG: Live mode active, reading next pair")
            self._processing_status_label.text = "Live: Fetching next..."
            self.read_next_pair() # <<<--- Automatically start next frame
        else:
            self._processing_status_label.text = "Idle" # Update status

    def _depth_range_slider_changed(self, v: float):
        self._depth_range_manually_changed = True
        self._update_rendering()

    def _update_rendering(self, names_to_update=None):
        if names_to_update is None:
            names_to_update = list(self.stereo_methods_output.keys())

        cam = self._scene.scene.camera
        width, height = self.window.size.width, self.window.size.height
        if width > 0 and height > 0:
            # Use get_near() and get_far() methods
            cam.set_projection(cam.get_field_of_view(), width / height, \
                cam.get_near(), cam.get_far(), cam.get_field_of_view_type())

        updated = False
        for name in names_to_update:
            if name not in self.stereo_methods_output:
                continue
            o = self.stereo_methods_output[name]
            if o.point_cloud is not None:
                # Check if points are empty to avoid Open3D errors
                if not o.point_cloud.has_points():
                    if self._scene.scene.has_geometry(name):
                        self._scene.scene.remove_geometry(name)
                    continue # Skip empty point clouds

                material = rendering.MaterialRecord()
                # material.shader = "unlitGradient"
                material.shader = "defaultLit" # Try this shader instead
                material.base_color = [0.7, 0.7, 0.7, 1.0]  # Give it a default color
                # material.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
                material.point_size = 2 * self.window.scaling
                
                # Remove the old geometry if it exists
                if self._scene.scene.has_geometry(name):
                    self._scene.scene.remove_geometry(name)
                
                # Add the new geometry
                self._scene.scene.add_geometry(name, o.point_cloud, material)
                
                updated = True
            else:
                # If the output point cloud is None, remove the geometry if it exists
                if self._scene.scene.has_geometry(name):
                    self._scene.scene.remove_geometry(name)
                    updated = True
        
        # # Always show axes?
        # if self.settings.show_axes:
        #     if not self._scene.scene.has_geometry("__axes__"):
        #         axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
        #         self._scene.scene.add_geometry("__axes__", axes, rendering.MaterialRecord())
        # else:
        #     if self._scene.scene.has_geometry("__axes__"):
        #         self._scene.scene.remove_geometry("__axes__")

        # This is essential for updating the scene after geometry changes
        if updated:
            self._scene.force_redraw()

    def _run_current_method(self):
        if self.executor_future is not None and not self.executor_future.done():
            print("DEBUG: Computation already in progress.")
            # Maybe update status briefly?
            # self._processing_status_label.text = "Busy..."
            # time.sleep(0.1)
            # if not self.is_live_processing_active: # Revert if not live
            #     self._processing_status_label.text = "Idle"
            return # Don't start a new one if one is running

        if not self.input.has_data():
            print("DEBUG: Cannot run method - input has no data")
            self._processing_status_label.text = "No Input Data"
            return

        name = self.algo_list.selected_value
        print(f"DEBUG: Running stereo method: {name}")
        self._processing_status_label.text = f"Processing: {name}..." # <-- Update status

        def do_beefy_work():
            print("DEBUG: Starting stereo computation in background thread")
            start_time = time.time()
            try:
                # Make a deep copy of the input to avoid race conditions if live mode updates it
                compute_input = copy.deepcopy(self.input)
                stereo_output = self.stereo_methods[name].compute_disparity(compute_input)
                # Ensure computation_time is set
                if stereo_output and np.isnan(stereo_output.computation_time):
                     stereo_output.computation_time = time.time() - start_time
                print(f"DEBUG: Stereo computation complete. Output shape: {stereo_output.disparity_pixels.shape if stereo_output and stereo_output.disparity_pixels is not None else None}")
                return stereo_output
            except Exception as e:
                 print(f"ERROR during stereo computation thread for {name}: {e}")
                 import traceback
                 traceback.print_exc()
                 # Return None or raise exception to signal failure in _check_run_complete
                 raise e # Re-raise the exception

        self.executor_future = self.executor.submit(do_beefy_work)

    def _toggle_live_mode(self):
        self.is_live_processing_active = not self.is_live_processing_active
        if self.is_live_processing_active:
            self._live_button.text = "Stop Live"
            self._next_image_button.enabled = False # Disable Next while live
            self._processing_status_label.text = "Live: Starting..."
            # If not already processing, start the first frame
            if self.executor_future is None or self.executor_future.done():
                print("DEBUG: Starting live mode, reading first pair")
                self.read_next_pair()
            else:
                 print("DEBUG: Live mode activated, but processing already running.")
        else:
            self._live_button.text = "Start Live"
            self._next_image_button.enabled = True # Re-enable Next
            self._processing_status_label.text = "Idle"
            print("DEBUG: Stopping live mode.")
            # Don't clear outputs, just stop requesting new frames

    def _update_runtime (self):
        name = self.algo_list.selected_value
        output = self.stereo_methods_output[name]
        if np.isnan(output.computation_time):
            self.last_runtime.text = "No output yet."
        else:
            self.last_runtime.text = f"Computation time: {output.computation_time*1e3:.1f} ms"

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        settings_width = 17 * layout_context.theme.font_size
        r = self.window.content_rect
        self._scene.frame = gui.Rect(0, r.y, r.get_right() - settings_width, r.height)
        # height = min(
        #     r.height,
        #     self._settings_panel.calc_preferred_size(
        #         layout_context, gui.Widget.Constraints()).height)
        height = r.height
        self._settings_panel.frame = gui.Rect(r.get_right() - settings_width, r.y, settings_width, height)
