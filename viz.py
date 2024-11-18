from ao import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def visualize_wfs_results(
    phase_screen: jnp.ndarray,
    pupil_mask: jnp.ndarray,
    valid_subaps: jnp.ndarray,
    slopes_x: jnp.ndarray,
    slopes_y: jnp.ndarray,
    centroids_x: jnp.ndarray,
    centroids_y: jnp.ndarray,
    reconstructed: jnp.ndarray,
    geometry: WFSGeometry,
    wfs: 'ShackHartmannWFS'
) -> plt.Figure:
    """
    Comprehensive visualization of WFS measurements and reconstruction
    """
    # Convert JAX arrays to NumPy for visualization
    phase_screen = np.array(phase_screen)
    pupil_mask = np.array(pupil_mask)
    valid_subaps = np.array(valid_subaps)
    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)
    centroids_x = np.array(centroids_x)
    centroids_y = np.array(centroids_y)
    reconstructed = np.array(reconstructed)
    subap_centers_x = np.array(geometry.subap_centers[0])
    subap_centers_y = np.array(geometry.subap_centers[1])
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1], figure=fig)
    gs.update(wspace=0.3)
    
    # 1. Original phase screen with pupil mask
    ax1 = fig.add_subplot(gs[0])
    masked_phase = phase_screen * pupil_mask
    im1 = ax1.imshow(
        masked_phase, 
        extent=[-geometry.diameter/2, geometry.diameter/2, 
                -geometry.diameter/2, geometry.diameter/2]
    )
    plt.colorbar(im1, ax=ax1, label='Phase [rad]')
    
    # Add subaperture grid
    for i in range(geometry.n_subapertures + 1):
        x = -geometry.diameter/2 + i * geometry.d_subap
        ax1.axvline(x=x, color='r', alpha=0.3)
        ax1.axhline(y=x, color='r', alpha=0.3)
    ax1.set_title('Phase Screen & Pupil')
    ax1.set_aspect('equal')
    
    # 2. Spot pattern detector image
    ax2 = fig.add_subplot(gs[1])
    detector_image = np.zeros((geometry.n_pixels, geometry.n_pixels))
    
    for i in range(geometry.n_subapertures):
        for j in range(geometry.n_subapertures):
            if not valid_subaps[i,j]:
                continue
                
            psf = np.array(wfs.compute_psf(slopes_x[i,j], slopes_y[i,j]))
            y_start = i * geometry.pixels_per_subap
            x_start = j * geometry.pixels_per_subap
            detector_image[
                y_start:y_start+geometry.pixels_per_subap,
                x_start:x_start+geometry.pixels_per_subap
            ] = psf
            
            # Plot reference and measured centroids
            ax2.plot(geometry.reference_centroids[j], 
                    geometry.reference_centroids[i], 
                    'r+', markersize=5)
            ax2.plot(centroids_x[i,j], centroids_y[i,j], 
                    'g+', markersize=5)
    
    im2 = ax2.imshow(detector_image, extent=[0, geometry.n_pixels, geometry.n_pixels, 0])
    plt.colorbar(im2, ax=ax2, label='Intensity')
    
    # Add subaperture grid
    for i in range(geometry.n_subapertures + 1):
        ax2.axvline(x=i*geometry.pixels_per_subap, color='w', alpha=0.3)
        ax2.axhline(y=i*geometry.pixels_per_subap, color='w', alpha=0.3)
    ax2.set_title('Detector Image\nred: reference, green: measured')
    ax2.set_aspect('equal')
    
    # 3. Slope vectors
    ax3 = fig.add_subplot(gs[2])
    max_slope = max(np.max(np.abs(slopes_x)), np.max(np.abs(slopes_y)))
    if max_slope > 0:
        slopes_x_norm = slopes_x / max_slope
        slopes_y_norm = slopes_y / max_slope
    else:
        slopes_x_norm = slopes_x
        slopes_y_norm = slopes_y
    
    # Plot slope vectors only for valid subapertures
    for i in range(geometry.n_subapertures):
        for j in range(geometry.n_subapertures):
            if valid_subaps[i,j]:
                ax3.arrow(
                    subap_centers_x[i,j], 
                    subap_centers_y[i,j],
                    slopes_x_norm[i,j] * geometry.d_subap * 0.4,
                    slopes_y_norm[i,j] * geometry.d_subap * 0.4,
                    head_width=geometry.d_subap*0.1, 
                    color='r'
                )
    
    # Create explicit grid lines at subaperture boundaries
    grid_positions = np.linspace(
        -geometry.diameter/2, 
        geometry.diameter/2, 
        geometry.n_subapertures + 1
    )
    for pos in grid_positions:
        ax3.axvline(x=pos, color='gray', alpha=0.3, linestyle='--')
        ax3.axhline(y=pos, color='gray', alpha=0.3, linestyle='--')
    
    ax3.set_xlim([-geometry.diameter/2, geometry.diameter/2])
    ax3.set_ylim([-geometry.diameter/2, geometry.diameter/2])
    ax3.set_aspect('equal')
    ax3.set_title('Measured Slopes')
    
    # 4. Reconstructed phase
    ax4 = fig.add_subplot(gs[3])
    recon_masked = reconstructed * valid_subaps
    valid_indices = valid_subaps > 0
    if np.any(valid_indices):
        recon_masked -= np.mean(recon_masked[valid_indices])
    
    im4 = ax4.imshow(
        recon_masked,
        extent=[-geometry.diameter/2, geometry.diameter/2, 
                -geometry.diameter/2, geometry.diameter/2]
    )
    plt.colorbar(im4, ax=ax4, label='Reconstructed Phase [rad]')
    ax4.set_title('Reconstructed Phase')
    ax4.set_aspect('equal')
    
    return fig

def main():
    # Parameters
    resolution = 128
    diameter = 8.0
    n_subapertures = 10
    wavelength = 500e-9
    
    # Initialize modules
    pupil = PupilMask(resolution, diameter, central_obstruction=0.15)
    geometry = WFSGeometry(diameter, resolution, n_subapertures)
    wfs = ShackHartmannWFS(geometry, wavelength)
    reconstructor = WavefrontReconstructor(geometry.d_subap)
    turbulence = AtmosphericTurbulence(
        resolution, r0=0.15, L0=100, diameter=diameter,
        key=jax.random.PRNGKey(0)
    )
    
   # Generate phase screen and measure
    pupil_mask = pupil()
    valid_subaps = wfs.create_valid_subaps_mask(pupil_mask)
    phase_screen = turbulence.generate_phase_screen()
    slopes_x, slopes_y, centroids_x, centroids_y = wfs.measure_slopes(
        phase_screen, valid_subaps
    )
    
    # Convert valid_subaps to float32 for calculations
    valid_subaps = valid_subaps.astype(jnp.float32)
    
    # Reconstruct wavefront
    reconstructed = reconstructor(slopes_x, slopes_y, valid_subaps)
    
    # Visualize results
    fig = visualize_wfs_results(
        phase_screen,
        pupil_mask,
        valid_subaps,
        slopes_x,
        slopes_y,
        centroids_x,
        centroids_y,
        reconstructed,
        geometry,
        wfs
    )
    plt.show()
    
    return phase_screen, reconstructed, (slopes_x, slopes_y, centroids_x, centroids_y)

if __name__ == "__main__":
    main()