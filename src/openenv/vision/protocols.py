"""Vision observation protocols and conventions.

Provides optional protocols for vision-first environments.
These are NOT mandatory - environments can use any observation format.
"""

from typing import Protocol, runtime_checkable, Any, Dict, Optional, Union


@runtime_checkable
class VisionObservation(Protocol):
    """Protocol for observations that include visual frames.
    
    This is an OPTIONAL protocol that vision-first environments
    can implement to provide a standard interface for accessing
    image data in observations.
    
    Environments are NOT required to implement this - it's a
    convenience for environments that want to expose visual data
    in a standard way.
    
    Example:
        @dataclass
        class RobotCityObs:
            frame: np.ndarray  # RGB image (H, W, 3)
            reward: float
            done: bool
            info: dict
            
            def get_frame(self) -> Any:
                return self.frame
            
            def get_frame_info(self) -> dict:
                return {
                    "width": self.frame.shape[1],
                    "height": self.frame.shape[0],
                    "channels": self.frame.shape[2],
                }
    """
    
    def get_frame(self) -> Any:
        """Get the primary visual frame from this observation.
        
        Returns:
            Image frame (numpy array, PIL Image, or encoded string)
        """
        ...
    
    def get_frame_info(self) -> Dict[str, Any]:
        """Get metadata about the frame.
        
        Returns:
            Dictionary with at least 'width', 'height' keys
        """
        ...


def is_vision_observation(obs: Any) -> bool:
    """Check if an observation implements VisionObservation protocol.
    
    Args:
        obs: Observation to check
        
    Returns:
        True if observation has get_frame() and get_frame_info() methods
    """
    return isinstance(obs, VisionObservation)


def extract_frame(obs: Any) -> Optional[Any]:
    """Extract frame from observation if possible.
    
    Tries multiple strategies:
    1. VisionObservation.get_frame()
    2. obs.frame attribute
    3. obs["frame"] key
    4. obs["image"] key
    5. obs["rgb"] key
    
    Args:
        obs: Observation that may contain a frame
        
    Returns:
        Extracted frame or None if not found
    """
    # Try VisionObservation protocol
    if is_vision_observation(obs):
        return obs.get_frame()
    
    # Try common attribute names
    for attr in ("frame", "image", "rgb", "pixels", "observation"):
        if hasattr(obs, attr):
            val = getattr(obs, attr)
            if val is not None:
                return val
    
    # Try dictionary keys
    if isinstance(obs, dict):
        for key in ("frame", "image", "rgb", "pixels", "observation"):
            if key in obs:
                return obs[key]
    
    return None


def make_vision_observation(
    frame: Any,
    reward: float = 0.0,
    done: bool = False,
    info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standard vision observation dictionary.
    
    This is a helper for environments that want to return
    observations in a standard format without implementing
    a full class.
    
    Args:
        frame: Image frame
        reward: Reward value
        done: Done flag
        info: Additional info dictionary
        
    Returns:
        Dictionary with standard vision observation fields
    """
    frame_info = {}
    
    # Try to get frame dimensions
    if hasattr(frame, "shape"):
        if len(frame.shape) == 2:
            frame_info = {
                "height": frame.shape[0],
                "width": frame.shape[1],
                "channels": 1,
            }
        elif len(frame.shape) == 3:
            frame_info = {
                "height": frame.shape[0],
                "width": frame.shape[1],
                "channels": frame.shape[2],
            }
    elif hasattr(frame, "width") and hasattr(frame, "height"):
        # PIL Image
        frame_info = {
            "width": frame.width,
            "height": frame.height,
            "channels": len(frame.mode),  # Approximate
        }
    
    return {
        "frame": frame,
        "frame_info": frame_info,
        "reward": reward,
        "done": done,
        "info": info or {},
    }
