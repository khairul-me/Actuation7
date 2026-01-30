import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_raw_cube(path, lines=512, bands=204, samples=512, dtype=np.uint16):
    """
    Load hyperspectral raw cube data.
    
    Parameters:
    -----------
    path : str
        Path to the .raw file
    lines : int
        Number of lines (height) in the image
    bands : int
        Number of spectral bands
    samples : int
        Number of samples (width) in the image
    dtype : numpy dtype
        Data type of the raw file
        
    Returns:
    --------
    cube : numpy array
        Hyperspectral data cube with shape (lines, samples, bands)
    """
    raw = np.fromfile(path, dtype=dtype)
    cube = raw.reshape((lines, bands, samples)).transpose(0, 2, 1)  # (lines, samples, bands)
    return cube


def visualize_band(cube, band_index, title=None, cmap='gray'):
    """
    Visualize a specific spectral band.
    
    Parameters:
    -----------
    cube : numpy array
        Hyperspectral data cube with shape (lines, samples, bands)
    band_index : int
        Index of the band to visualize
    title : str, optional
        Title for the plot
    cmap : str
        Colormap for visualization
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cube[:, :, band_index], cmap=cmap)
    plt.colorbar(label='Intensity')
    if title:
        plt.title(title)
    else:
        plt.title(f'Band {band_index}')
    plt.xlabel('Samples')
    plt.ylabel('Lines')
    plt.tight_layout()
    plt.show()


def visualize_rgb(cube, red_band=70, green_band=53, blue_band=19):
    """
    Create an RGB visualization from the hyperspectral cube.
    
    Parameters:
    -----------
    cube : numpy array
        Hyperspectral data cube with shape (lines, samples, bands)
    red_band : int
        Band index for red channel
    green_band : int
        Band index for green channel
    blue_band : int
        Band index for blue channel
    """
    # Extract RGB bands
    r = cube[:, :, red_band]
    g = cube[:, :, green_band]
    b = cube[:, :, blue_band]
    
    # Normalize each channel to 0-1 range
    r_norm = (r - r.min()) / (r.max() - r.min())
    g_norm = (g - g.min()) / (g.max() - g.min())
    b_norm = (b - b.min()) / (b.max() - b.min())
    
    # Stack to create RGB image
    rgb = np.dstack([r_norm, g_norm, b_norm])
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title(f'RGB Composite (R:{red_band}, G:{green_band}, B:{blue_band})')
    plt.xlabel('Samples')
    plt.ylabel('Lines')
    plt.tight_layout()
    plt.show()


def main():
    # Define the data directory
    data_dir = Path(r'K:\Khairul_Weed\T2_2026-01-09_002\capture')
    
    # Main data file
    file_name = 'T2_2026-01-09_002.raw'
    file_path = data_dir / file_name
    
    print(f"Loading hyperspectral data from: {file_path}")
    
    # Load the hyperspectral cube
    cube = load_raw_cube(file_path, lines=512, bands=204, samples=512, dtype=np.uint16)
    
    print(f"Data cube shape: {cube.shape}")
    print(f"Data type: {cube.dtype}")
    print(f"Min value: {cube.min()}")
    print(f"Max value: {cube.max()}")
    print(f"Mean value: {cube.mean():.2f}")
    
    # Visualize a specific band (band 100 as an example)
    band_to_visualize = 100
    print(f"\nVisualizing band {band_to_visualize}...")
    visualize_band(cube, band_to_visualize, title=f'Band {band_to_visualize}')
    
    # Create RGB composite using default bands from header file
    print("\nCreating RGB composite...")
    visualize_rgb(cube, red_band=70, green_band=53, blue_band=19)
    
    # Optional: Visualize multiple bands in a grid
    print("\nCreating multi-band visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    bands_to_show = [0, 50, 100, 150, 180, 203]  # First, middle, and last bands
    
    for idx, (ax, band) in enumerate(zip(axes.flat, bands_to_show)):
        ax.imshow(cube[:, :, band], cmap='gray')
        ax.set_title(f'Band {band}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Return the cube for further analysis
    return cube


if __name__ == "__main__":
    cube = main()
    
    # Additional analysis can be done here
    # For example, you can examine spectral signatures at specific pixels
    # pixel_spectrum = cube[256, 256, :]  # Center pixel
    # plt.figure()
    # plt.plot(pixel_spectrum)
    # plt.xlabel('Band Index')
    # plt.ylabel('Intensity')
    # plt.title('Spectral Signature at Pixel (256, 256)')
    # plt.show()
