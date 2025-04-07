import numpy as np
from pathlib import Path
import sys
import cv2 # Ensure cv2 is imported if not already
import time  # Import time for sleep function
import ctypes

try:
    from syphon.client import SyphonMetalClient
    from syphon.server_directory import SyphonServerDirectory
    import Metal
    from syphon.utils.numpy import copy_mtl_texture_to_image  # Import the official utility function
except ImportError:
    print("\nError: syphon-python library not found or not working.", file=sys.stderr)
    print("Please try reinstalling:", file=sys.stderr)
    print("  1. Uninstall: pip uninstall Syphon-python", file=sys.stderr)
    print("  2. Install again: pip install Syphon-python", file=sys.stderr)
    print("\nIf the issue persists, try:", file=sys.stderr)
    print("  1. Check Python version (3.8+ required)", file=sys.stderr)
    print("  2. Make sure you're on macOS", file=sys.stderr)
    print("  3. Ensure Metal framework is available", file=sys.stderr)
    sys.exit(1)

from .visualizer import Source, Calibration # Assuming Calibration is in visualizer or methods
from .methods import InputPair # Assuming InputPair is in methods

# Default calibration values for common resolutions
DEFAULT_CALIBRATIONS = {
    # 640x480 - Similar to OAK-D camera
    "640x480": {
        "width": 640,
        "height": 480,
        "baseline_meters": 0.075,
        "fx": 451.0,
        "fy": 451.0,
        "cx0": 320.0,
        "cx1": 320.0,
        "cy": 240.0,
        "depth_range": [0.5, 20.0],
        "left_image_rect_normalized": [0.0, 0.0, 1.0, 1.0]
    },
    # 1280x720 (720p)
    "1280x720": {
        "width": 1280,
        "height": 720,
        "baseline_meters": 0.075, 
        "fx": 900.0,
        "fy": 900.0,
        "cx0": 640.0,
        "cx1": 640.0,
        "cy": 360.0,
        "depth_range": [0.5, 20.0],
        "left_image_rect_normalized": [0.0, 0.0, 1.0, 1.0]
    },
    # 1920x1080 (1080p)
    "1920x1080": {
        "width": 1920,
        "height": 1080,
        "baseline_meters": 0.075,
        "fx": 1400.0,
        "fy": 1400.0,
        "cx0": 960.0,
        "cx1": 960.0,
        "cy": 540.0,
        "depth_range": [0.5, 20.0],
        "left_image_rect_normalized": [0.0, 0.0, 1.0, 1.0]
    }
}

class SyphonSource(Source):
    """
    An input source that reads a side-by-side stereo image pair 
    from a single Syphon server and splits it.
    """
    def __init__(self, server_name: str, calibration_path: Path = None):
        """
        Initializes the Syphon client and loads calibration.

        Args:
            server_name: The name of the Syphon server streaming the side-by-side image.
            calibration_path: Path to the JSON calibration file. 
                              IMPORTANT: Calibration must match the dimensions and principal points
                              of the *individual* (split) left/right images.
                              If None, a default calibration will be used based on frame size.
        """
        super().__init__()
        self.server_name = server_name
        self.calibration_path = calibration_path
        self.calibration = None
        self.client = None # Single client now
        self.using_default_calibration = False
        
        # Initialize Syphon client first
        self._connect_to_syphon()
        
        if not self.client:
             print("Error: Failed to connect to Syphon server after multiple attempts. Exiting.", file=sys.stderr)
             sys.exit(1) # Exit if connection failed
        
        # Load calibration if path is provided
        if self.calibration_path:
            try:
                if not self.calibration_path.exists():
                    raise FileNotFoundError(f"Calibration file not found: {self.calibration_path}")
                with open(self.calibration_path, 'r') as f:
                    self.calibration = Calibration.from_json(f.read()) 
                    print(f"INFO: Loaded calibration from {self.calibration_path}")
            except Exception as e:
                print(f"Error loading calibration file {self.calibration_path}: {e}", file=sys.stderr)
                print("Will attempt to use default calibration based on frame dimensions.", file=sys.stderr)
                self.calibration = None
        else:
            print("No calibration file specified. Will use default calibration based on frame dimensions.", file=sys.stderr)
            
        # If no calibration is loaded yet, we'll create a default one after receiving the first frame
        # This happens in get_next_pair()
        if self.calibration:
            print(f"INFO: Calibration expects resolution (per side): {self.calibration.width}x{self.calibration.height}")

    def _connect_to_syphon(self, max_retries=5, retry_interval=1.0):
        """
        Attempts to connect to the Syphon server with retries.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_interval: Time in seconds between retries
        """
        retries = 0
        while retries < max_retries:
            try:
                # Get server description from the server directory
                server_directory = SyphonServerDirectory()
                server_descriptions = server_directory.servers
                
                # Print available servers for debugging
                print(f"Available Syphon servers: {len(server_descriptions)}")
                for i, desc in enumerate(server_descriptions):
                    print(f"  {i+1}. {desc.name} - {desc.app_name}")
                
                # Find our server
                target_server = None
                for desc in server_descriptions:
                    if desc.name == self.server_name:
                        target_server = desc
                        break
                
                if target_server:
                    print(f"Found server: {target_server.name} from app: {target_server.app_name}")
                    self.client = SyphonMetalClient(target_server)
                    print(f"INFO: Connected to Syphon server: {self.server_name}")
                    return True
                else:
                    raise RuntimeError(f"Syphon server '{self.server_name}' not found")
                    
            except Exception as e:
                retries += 1
                if retries < max_retries:
                    print(f"Warning: Failed to connect to Syphon server '{self.server_name}': {e}", file=sys.stderr)
                    print(f"Retrying in {retry_interval} seconds... (Attempt {retries}/{max_retries})", file=sys.stderr)
                    time.sleep(retry_interval)
                else:
                    print(f"Error: Failed to connect to Syphon server '{self.server_name}' after {max_retries} attempts: {e}", file=sys.stderr)
                    self.client = None
                    return False

    def _create_default_calibration(self, width: int, height: int) -> Calibration:
        """
        Creates a default calibration based on frame dimensions.
        
        Args:
            width: Width of a single camera view (half of the side-by-side image)
            height: Height of the image
            
        Returns:
            A default Calibration object
        """
        resolution_key = f"{width}x{height}"
        
        if resolution_key in DEFAULT_CALIBRATIONS:
            # Use predefined calibration for common resolutions
            calib_params = DEFAULT_CALIBRATIONS[resolution_key]
            print(f"INFO: Using default calibration for {resolution_key} resolution", file=sys.stderr)
        else:
            # Create a reasonable default for any resolution
            print(f"INFO: Creating default calibration for {resolution_key} resolution", file=sys.stderr)
            calib_params = {
                "width": width,
                "height": height,
                "baseline_meters": 0.075,  # Reasonable default
                "fx": width * 0.8,         # Reasonable default focal length
                "fy": height * 0.8,        # Base fy on height
                "cx0": width/2.0,          # Principal point at center
                "cx1": width/2.0,
                "cy": height/2.0,
                "depth_range": [0.5, 20.0],
                "left_image_rect_normalized": [0.0, 0.0, 1.0, 1.0]
            }
            
        self.using_default_calibration = True
        return Calibration(**calib_params)

    def is_live(self) -> bool:
        """This source provides live images."""
        return True

    def _metal_texture_to_numpy(self, metal_texture):
        """
        Convert a Metal texture to a numpy array
        
        Args:
            metal_texture: The Metal texture to convert
            
        Returns:
            A numpy array containing the image data (RGB format)
        """
        if metal_texture is None:
            raise RuntimeError("Received None texture from Syphon")
        
        # Get texture dimensions
        texture_width = metal_texture.width
        texture_height = metal_texture.height
        
        # Create a Metal buffer to hold the texture data
        buffer_size = texture_width * texture_height * 4  # RGBA
        device = Metal.MTLCreateSystemDefaultDevice()
        buffer = device.newBufferWithLength_options_(buffer_size, Metal.MTLResourceStorageModeShared)
        
        # Create a command buffer to copy texture to buffer
        command_queue = device.newCommandQueue()
        command_buffer = command_queue.commandBuffer()
        
        # Create a blit encoder to copy texture to buffer
        blit_encoder = command_buffer.blitCommandEncoder()
        blit_encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_(
            metal_texture, 0, 0, Metal.MTLOriginMake(0, 0, 0),
            Metal.MTLSizeMake(texture_width, texture_height, 1),
            buffer, 0, texture_width * 4, 0
        )
        blit_encoder.endEncoding()
        
        # Execute command
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # Get data from buffer
        data_pointer = buffer.contents()
        buffer_pointer = ctypes.cast(data_pointer, ctypes.POINTER(ctypes.c_ubyte * buffer_size))
        buffer_array = np.frombuffer(buffer_pointer.contents, dtype=np.uint8)
        
        # Reshape to image (RGB format)
        return buffer_array.reshape(texture_height, texture_width, 4)[:,:,:3]  # Remove alpha channel

    def get_next_pair(self) -> InputPair:
        """
        Fetches the latest frame from the Syphon server and splits it.

        Returns:
            An InputPair containing the left and right images and calibration,
            or raises an exception if frames cannot be retrieved or split.
        """
        if not self.client:
            raise RuntimeError("Syphon client is not connected.")

        print("DEBUG: Entering get_next_pair")

        try:
            # Get the combined side-by-side frame from the Syphon client
            try:
                # Check if there's a new frame
                print("DEBUG: Checking for new Syphon frame...")
                frame_wait_count = 0
                while not self.client.has_new_frame:
                    print(f"Waiting for new frame... ({frame_wait_count})", end="\\r")
                    frame_wait_count += 1
                    time.sleep(0.01)  # Short sleep to not hog CPU

                print("\\nDEBUG: New frame detected (has_new_frame is True)")

                # Get texture and convert to numpy array using the official method
                metal_texture = self.client.new_frame_image
                print(f"DEBUG: Obtained metal_texture: {type(metal_texture)}")
                sbs_frame_rgb = copy_mtl_texture_to_image(metal_texture)
                print(f"DEBUG: Converted texture to numpy array, shape: {sbs_frame_rgb.shape}")

            except Exception as e:
                print(f"Error getting frame from Syphon: {e}. Attempting to reconnect...", file=sys.stderr)
                if self._connect_to_syphon(max_retries=3):
                    time.sleep(0.5)
                    if hasattr(self.client, 'has_new_frame'):
                        return self.get_next_pair()
                    else:
                        raise RuntimeError("Reconnected client is not properly initialized")
                else:
                    raise RuntimeError(f"Failed to reconnect to Syphon server")

            # Split the frame horizontally
            height, total_width, _ = sbs_frame_rgb.shape
            print(f"DEBUG: Full SBS frame shape: H={height}, W={total_width}")
            mid_point = total_width // 2
            print(f"DEBUG: Calculated mid_point: {mid_point}")
            single_width = mid_point

            # If we don't have a calibration yet, create a default one
            if not self.calibration:
                self.calibration = self._create_default_calibration(single_width, height)

            # Check if total width is roughly double the calibration width
            if abs(total_width - 2 * self.calibration.width) > 2:  # Allow minor tolerance
                if not self.using_default_calibration:
                    print(f"Warning: Combined Syphon frame width ({total_width}) is not approx twice the "
                          f"calibration width ({self.calibration.width}). Check calibration file or stream format.", file=sys.stderr)
                else:
                    # If using default calibration and it doesn't match, recreate it
                    self.calibration = self._create_default_calibration(single_width, height)

            # Check if frame height matches calibration height
            if height != self.calibration.height and not self.using_default_calibration:
                print(f"Warning: Syphon frame height ({height}) does not match calibration height "
                      f"({self.calibration.height}). Check calibration file or stream format.", file=sys.stderr)

            # Slice the frame into left and right images
            left_frame_rgb = sbs_frame_rgb[:, :mid_point]
            right_frame_rgb = sbs_frame_rgb[:, mid_point:]
            print(f"DEBUG: Left frame shape: {left_frame_rgb.shape}")
            print(f"DEBUG: Right frame shape: {right_frame_rgb.shape}")
            print(f"DEBUG: Left frame min/max values: {left_frame_rgb.min()}/{left_frame_rgb.max()}")
            print(f"DEBUG: Right frame min/max values: {right_frame_rgb.min()}/{right_frame_rgb.max()}")

            # Ensure frames are in the format expected by the stereo methods (BGR)
            left_image_bgr = cv2.cvtColor(left_frame_rgb, cv2.COLOR_RGB2BGR)
            right_image_bgr = cv2.cvtColor(right_frame_rgb, cv2.COLOR_RGB2BGR)

            # Verify converted images
            print(f"DEBUG: BGR Left frame min/max values: {left_image_bgr.min()}/{left_image_bgr.max()}")
            print(f"DEBUG: BGR Right frame min/max values: {right_image_bgr.min()}/{right_image_bgr.max()}")

            # Validate images before creating InputPair
            if left_image_bgr.size == 0 or right_image_bgr.size == 0:
                raise RuntimeError("Empty image detected after conversion")
            
            if not np.any(left_image_bgr) or not np.any(right_image_bgr):
                raise RuntimeError("Images contain only zeros")

            # Create debug visualization
            debug_vis = np.hstack((left_image_bgr, right_image_bgr))
            cv2.putText(debug_vis, "Calibrated Input", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(debug_vis, f"Resolution: {single_width}x{height}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Calibrated Input", debug_vis)
            cv2.waitKey(1)

            # Save debug images occasionally
            if np.random.random() < 0.01:  # 1% chance to save debug images
                cv2.imwrite("debug_left.png", left_image_bgr)
                cv2.imwrite("debug_right.png", right_image_bgr)
                print("DEBUG: Saved debug images")

            # Set status based on calibration type
            status_prefix = "Using default calibration" if self.using_default_calibration else "Using loaded calibration"
            status = f"{status_prefix}: {self.server_name}"

            print("DEBUG: Frame successfully split and converted. Returning InputPair.")
            input_pair = InputPair(left_image_bgr,
                                 right_image_bgr,
                                 self.calibration,
                                 status)
            
            # Verify InputPair before returning
            if not input_pair.has_data():
                raise RuntimeError("Created InputPair reports no data")
                
            return input_pair

        except RuntimeError as e:
            print(f"DEBUG: Runtime Error in get_next_pair: {e}", file=sys.stderr)
            raise e
        except Exception as e:
            print(f"DEBUG: Unexpected Error in get_next_pair: {e}", file=sys.stderr)
            raise e

    def selected_index (self) -> int:
        """Not applicable for a live source."""
        return 0 # Or -1

    def get_pair_at_index(self, idx: int) -> InputPair:
        """Not applicable for a live source."""
        # Maybe return the *current* pair instead of None?
        # return self.get_next_pair() # Careful, this blocks.
        print("Warning: get_pair_at_index called on live Syphon source.", file=sys.stderr)
        return InputPair(None, None, self.calibration, "N/A (Live Source)")

    def get_pair_list(self) -> list[str]:
        """Not applicable for a live source."""
        return []

    def __del__(self):
        """Ensure Syphon client is released."""
        if self.client:
            self.client.stop()
            print(f"INFO: Stopped Syphon client for {self.server_name}")