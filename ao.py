import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Dict, Optional
import dataclasses

class PupilMask(eqx.Module):
    resolution: int
    diameter: float
    central_obstruction: float = eqx.field(default=0.0)
    
    def __call__(self) -> jnp.ndarray:
        x = jnp.linspace(-self.resolution/2, self.resolution/2, self.resolution)
        xx, yy = jnp.meshgrid(x, x)
        circle = xx**2 + yy**2
        outer_mask = circle < (self.resolution/2)**2
        
        if self.central_obstruction > 0:
            inner_mask = circle >= (self.central_obstruction * self.resolution/2)**2
            return outer_mask * inner_mask
        return outer_mask

@dataclasses.dataclass
class WFSGeometry:
    """Geometry parameters for Shack-Hartmann WFS"""
    diameter: float
    resolution: int
    n_subapertures: int
    
    def __post_init__(self):
        self.d_subap = self.diameter / self.n_subapertures
        self.pixels_per_subap = self.resolution // self.n_subapertures
        self.n_pixels = self.n_subapertures * self.pixels_per_subap
        
        # Compute subaperture centers
        x = jnp.linspace(
            -self.diameter/2 + self.d_subap/2, 
            self.diameter/2 - self.d_subap/2, 
            self.n_subapertures
        )
        self.subap_centers = jnp.meshgrid(x, x)
        
        # Reference centroids
        x_det = jnp.linspace(0, self.n_pixels, self.n_subapertures+1)
        self.reference_centroids = (x_det[1:] + x_det[:-1])/2
        
        # Pixel coordinates for PSF calculation
        xx, yy = jnp.meshgrid(
            jnp.linspace(-self.d_subap/2, self.d_subap/2, self.pixels_per_subap),
            jnp.linspace(-self.d_subap/2, self.d_subap/2, self.pixels_per_subap)
        )
        self.pixel_coords = (xx, yy)

class AtmosphericTurbulence(eqx.Module):
    """Atmospheric turbulence generator using FFT method"""
    resolution: int
    r0: float
    L0: float
    diameter: float
    key: jax.random.PRNGKey
    
    def generate_phase_screen(self) -> jnp.ndarray:
        dx = self.diameter / self.resolution
        del_f = 1.0 / (self.resolution * dx)
        
        fx = jnp.fft.fftfreq(self.resolution, dx)
        fx, fy = jnp.meshgrid(fx, fx)
        f = jnp.sqrt(fx**2 + fy**2)
        
        # von Karman spectrum parameters
        fm = 5.92/(2*jnp.pi*0.01)  # inner scale
        f0 = 1./self.L0
        
        # Power spectral density
        PSD_phi = (0.023*self.r0**(-5/3) * 
                  jnp.exp(-(f/fm)**2) * 
                  (f**2 + f0**2)**(-11/6))
        PSD_phi = PSD_phi.at[0,0].set(0)
        
        # Generate random complexes
        key1, key2 = jax.random.split(self.key)
        cn = (jax.random.normal(key1, (self.resolution, self.resolution)) + 
              1j * jax.random.normal(key2, (self.resolution, self.resolution))) / jnp.sqrt(2.)
        
        return jnp.real(jnp.fft.ifft2(cn * jnp.sqrt(PSD_phi))) * self.resolution**2 * del_f
    
class ShackHartmannWFS(eqx.Module):
    geometry: WFSGeometry
    wavelength: float
    
    def calculate_centroid(self, psf: jnp.ndarray) -> Tuple[float, float]:
        """Calculate centroid of PSF using JAX operations"""
        total_intensity = jnp.sum(psf)
        y_size, x_size = psf.shape
        
        y_coords = jnp.arange(y_size)
        x_coords = jnp.arange(x_size)
        
        y_centroid = jnp.sum(jnp.sum(psf, axis=1) * y_coords) / total_intensity
        x_centroid = jnp.sum(jnp.sum(psf, axis=0) * x_coords) / total_intensity
        
        return y_centroid, x_centroid
    
    def create_valid_subaps_mask(self, pupil_mask: jnp.ndarray) -> jnp.ndarray:
        """Create mask for valid subapertures"""
        n_subapertures = self.geometry.n_subapertures
        pixels_per_subap = self.geometry.pixels_per_subap
        valid_subaps = jnp.zeros((n_subapertures, n_subapertures))
        
        for i in range(n_subapertures):
            for j in range(n_subapertures):
                y_start = i * pixels_per_subap
                x_start = j * pixels_per_subap
                
                subap_mask = jax.lax.dynamic_slice(
                    pupil_mask,
                    (y_start, x_start),
                    (pixels_per_subap, pixels_per_subap)
                )
                
                illumination = jnp.sum(subap_mask) / (pixels_per_subap**2)
                valid_subaps = valid_subaps.at[i, j].set(illumination > 0.5)
        
        return valid_subaps
    
    def compute_psf(self, slopes_x: float, slopes_y: float) -> jnp.ndarray:
        """Compute PSF for a subaperture given local wavefront slopes"""
        xx, yy = self.geometry.pixel_coords
        phase = 2*jnp.pi/self.wavelength * (slopes_x * xx + slopes_y * yy)
        pupil = jnp.exp(1j * phase)
        psf = jnp.abs(jnp.fft.fft2(pupil))**2
        return psf / psf.sum()
    
    def measure_slopes(
        self,
        phase_screen: jnp.ndarray,
        valid_subaps: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Measure wavefront slopes from phase screen using vectorized operations."""
        n_subapertures = self.geometry.n_subapertures
        pixels_per_subap = self.geometry.pixels_per_subap
        subap_centers_x, subap_centers_y = self.geometry.subap_centers
        
        # Calculate base coordinates for all subapertures
        x_starts = jnp.array(
            (subap_centers_x + self.geometry.d_subap*n_subapertures/2) /
            (self.geometry.diameter/self.geometry.resolution)
        ).astype(jnp.int32)
        
        y_starts = jnp.array(
            (subap_centers_y + self.geometry.d_subap*n_subapertures/2) /
            (self.geometry.diameter/self.geometry.resolution)
        ).astype(jnp.int32)
        
        def compute_subaperture_slopes(y_start, x_start, i, j):
            """Compute slopes for a single subaperture"""
            # Extract subaperture phase
            subap_phase = jax.lax.dynamic_slice(
                phase_screen,
                (y_start, x_start),
                (pixels_per_subap, pixels_per_subap)
            )
            
            # Compute gradients
            dx = self.geometry.d_subap / self.geometry.pixel_coords[0].shape[0]
            grad_y, grad_x = jnp.gradient(subap_phase, dx)
            slope_x = jnp.mean(grad_x)
            slope_y = jnp.mean(grad_y)
            
            # Compute PSF and centroids
            psf = self.compute_psf(slope_x, slope_y)
            cy, cx = self.calculate_centroid(psf)
            
            # Add reference centroids correctly
            centroid_x = cx + self.geometry.reference_centroids[j]
            centroid_y = cy + self.geometry.reference_centroids[i]
            
            return slope_x, slope_y, centroid_x, centroid_y
        
        # Vectorized map over all subapertures
        def process_row(i):
            def process_col(j):
                y_start, x_start = y_starts[i, j], x_starts[i, j]
                slopes = compute_subaperture_slopes(y_start, x_start, i, j)
                
                # Apply valid_subaps mask
                masked_slopes = jax.tree_map(
                    lambda x: jnp.where(valid_subaps[i, j], x, 0.0),
                    slopes
                )
                return masked_slopes
                
            return jax.vmap(process_col)(jnp.arange(n_subapertures))
        
        # Map over all rows
        results = jax.vmap(process_row)(jnp.arange(n_subapertures))
        
        # Reshape results
        slopes_x = results[0]
        slopes_y = results[1]
        centroids_x = results[2]  # Reference centroids already added in compute_subaperture_slopes
        centroids_y = results[3]  # Reference centroids already added in compute_subaperture_slopes
        
        return slopes_x, slopes_y, centroids_x, centroids_y
    
class WavefrontReconstructor(eqx.Module):
    """Southwell reconstructor implemented as an Equinox Module"""
    d_subap: float
    
    def smooth_phase(self, phase: jnp.ndarray, sigma: float = 0.5) -> jnp.ndarray:
        """Apply Gaussian smoothing using JAX-compatible operations"""
        x = jnp.array([-1, 0, 1])
        xx, yy = jnp.meshgrid(x, x)
        kernel = jnp.exp(-(xx**2 + yy**2)/(2*sigma**2))
        kernel = kernel / jnp.sum(kernel)
        
        return jax.scipy.signal.convolve2d(
            phase, 
            kernel,
            mode='same',
            boundary='fill'
        )
    
    def __call__(
        self, 
        slopes_x: jnp.ndarray, 
        slopes_y: jnp.ndarray, 
        valid_subaps: jnp.ndarray
    ) -> jnp.ndarray:
        """Reconstruct wavefront from slopes using Southwell method"""
        # Ensure all inputs have the same shape
        assert slopes_x.shape == slopes_y.shape == valid_subaps.shape, \
            "Input shapes must match"
            
        # Convert valid_subaps to float for calculations
        valid_subaps = valid_subaps.astype(jnp.float32)
        
        # X-direction reconstruction
        phase_x = jnp.zeros_like(slopes_x)
        phase_x = phase_x.at[:, 1:].set(
            jnp.cumsum(slopes_x[:, :-1], axis=1) * self.d_subap
        )
        
        # Y-direction reconstruction
        phase_y = jnp.zeros_like(slopes_y)
        phase_y = phase_y.at[1:, :].set(
            jnp.cumsum(slopes_y[:-1, :], axis=0) * self.d_subap
        )
        
        # Average both reconstructions
        phase = (phase_x + phase_y) / 2.0
        
        # Remove piston from valid regions
        valid_sum = jnp.sum(valid_subaps)
        valid_mean = jnp.sum(phase * valid_subaps) / (valid_sum + 1e-10)
        phase = jnp.where(valid_subaps > 0, phase - valid_mean, 0.0)
        
        # Apply smoothing and mask
        smoothed_phase = self.smooth_phase(phase)
        return smoothed_phase * valid_subaps