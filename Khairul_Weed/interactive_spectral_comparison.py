import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class SpectralSelector:
    """Interactive tool for selecting and comparing spectral signatures."""
    
    def __init__(self, cube, wavelengths=None):
        """
        Initialize the spectral selector.
        
        Parameters:
        -----------
        cube : numpy array
            Hyperspectral data cube with shape (lines, samples, bands)
        wavelengths : array-like, optional
            Wavelength values for each band (in nm)
        """
        self.cube = cube
        self.wavelengths = wavelengths if wavelengths is not None else np.arange(cube.shape[2])
        self.selected_points = []
        self.point_labels = ['Weed', 'Crop']
        self.fig = None
        self.ax_image = None
        self.ax_spectra = None
        
    def on_click(self, event):
        """Handle mouse click events to select pixels."""
        if event.inaxes != self.ax_image:
            return
        
        if len(self.selected_points) >= 2:
            print("Two points already selected. Close the window to finish or restart the script.")
            return
        
        # Get the clicked coordinates
        x, y = int(event.xdata), int(event.ydata)
        
        # Store the point
        self.selected_points.append((y, x))  # Store as (row, col) for numpy indexing
        
        # Get the label for this point
        label = self.point_labels[len(self.selected_points) - 1]
        print(f"{label} selected at pixel ({y}, {x})")
        
        # Plot a marker on the image
        color = 'red' if label == 'Weed' else 'green'
        self.ax_image.plot(x, y, marker='o', markersize=10, color=color, 
                          markeredgecolor='white', markeredgewidth=2, label=label)
        self.ax_image.legend(loc='upper right')
        
        # Extract and plot the spectral signature
        spectrum = self.cube[y, x, :]
        self.ax_spectra.plot(self.wavelengths, spectrum, marker='o', markersize=3,
                            linestyle='-', linewidth=2, color=color, label=label)
        self.ax_spectra.legend(loc='best')
        self.ax_spectra.set_xlabel('Wavelength (nm)', fontsize=12)
        self.ax_spectra.set_ylabel('Reflectance Intensity', fontsize=12)
        self.ax_spectra.set_title('Spectral Signatures Comparison', fontsize=14, fontweight='bold')
        self.ax_spectra.grid(True, alpha=0.3)
        
        # Update the display
        self.fig.canvas.draw()
        
        if len(self.selected_points) == 2:
            print("\nBoth points selected! Analysis complete.")
            print("Close the window when done reviewing the results.")
            self.print_statistics()
    
    def print_statistics(self):
        """Print statistics about the selected spectral signatures."""
        weed_spectrum = self.cube[self.selected_points[0][0], self.selected_points[0][1], :]
        crop_spectrum = self.cube[self.selected_points[1][0], self.selected_points[1][1], :]
        
        print("\n" + "="*60)
        print("SPECTRAL SIGNATURE STATISTICS")
        print("="*60)
        print(f"Weed pixel at {self.selected_points[0]}:")
        print(f"  Mean: {weed_spectrum.mean():.2f}")
        print(f"  Min: {weed_spectrum.min()}")
        print(f"  Max: {weed_spectrum.max()}")
        print(f"  Std: {weed_spectrum.std():.2f}")
        
        print(f"\nCrop pixel at {self.selected_points[1]}:")
        print(f"  Mean: {crop_spectrum.mean():.2f}")
        print(f"  Min: {crop_spectrum.min()}")
        print(f"  Max: {crop_spectrum.max()}")
        print(f"  Std: {crop_spectrum.std():.2f}")
        
        # Calculate spectral angle
        dot_product = np.dot(weed_spectrum, crop_spectrum)
        norm_weed = np.linalg.norm(weed_spectrum)
        norm_crop = np.linalg.norm(crop_spectrum)
        spectral_angle = np.arccos(dot_product / (norm_weed * norm_crop))
        spectral_angle_deg = np.degrees(spectral_angle)
        
        print(f"\nSpectral Angle between signatures: {spectral_angle_deg:.2f} degrees")
        print("="*60)
    
    def select_pixels(self, reference_band=100):
        """
        Display the reference band and allow user to select pixels.
        
        Parameters:
        -----------
        reference_band : int
            Band index to display as reference image
        """
        # Create figure with two subplots
        self.fig = plt.figure(figsize=(16, 6))
        
        # Left subplot: Reference band image
        self.ax_image = plt.subplot(1, 2, 1)
        img = self.cube[:, :, reference_band]
        im = self.ax_image.imshow(img, cmap='gray')
        self.ax_image.set_title(f'Reference Band {reference_band}\nClick to select: 1st=Weed (red), 2nd=Crop (green)', 
                               fontsize=12, fontweight='bold')
        self.ax_image.set_xlabel('Sample (X)')
        self.ax_image.set_ylabel('Line (Y)')
        plt.colorbar(im, ax=self.ax_image, label='Intensity')
        
        # Right subplot: Spectral signatures
        self.ax_spectra = plt.subplot(1, 2, 2)
        self.ax_spectra.set_xlabel('Wavelength (nm)', fontsize=12)
        self.ax_spectra.set_ylabel('Reflectance Intensity', fontsize=12)
        self.ax_spectra.set_title('Spectral Signatures (click image to add)', fontsize=14, fontweight='bold')
        self.ax_spectra.grid(True, alpha=0.3)
        
        # Connect the click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        print("\n" + "="*60)
        print("INTERACTIVE SPECTRAL SIGNATURE SELECTOR")
        print("="*60)
        print("Instructions:")
        print("1. Click on a WEED pixel in the image (will appear RED)")
        print("2. Click on a CROP pixel in the image (will appear GREEN)")
        print("3. Spectral signatures will be plotted automatically")
        print("4. Close the window when done")
        print("="*60 + "\n")
        
        plt.show()
        
        return self.selected_points


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


def main():
    # Define wavelengths from the header file (397.32 to 1003.58 nm)
    wavelengths = np.array([
        397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29,
        426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25,
        455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32,
        484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48,
        513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75,
        542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12,
        572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60,
        601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18,
        631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87,
        660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65,
        690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54,
        720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54,
        750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64,
        780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84,
        810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14,
        841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55,
        871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06,
        902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
        932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40,
        963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22,
        994.31, 997.40, 1000.49, 1003.58
    ])
    
    # Define the data directory and file
    data_dir = Path(r'K:\Khairul_Weed\T2_2026-01-09_002\capture')
    file_name = 'T2_2026-01-09_002.raw'
    file_path = data_dir / file_name
    
    print(f"Loading hyperspectral data from: {file_path}")
    
    # Load the hyperspectral cube
    cube = load_raw_cube(file_path, lines=512, bands=204, samples=512, dtype=np.uint16)
    
    print(f"Data cube shape: {cube.shape}")
    print(f"Number of spectral bands: {len(wavelengths)}")
    print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm\n")
    
    # Create the interactive selector
    selector = SpectralSelector(cube, wavelengths)
    
    # Start the interactive selection (using band 100 as reference)
    selected_points = selector.select_pixels(reference_band=100)
    
    print("\nProgram completed.")


if __name__ == "__main__":
    main()
