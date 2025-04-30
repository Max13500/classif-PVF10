"""
PVModule class definition
"""
from typing import Optional, ClassVar
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, iqr


@dataclass
class PVModule:
    """Classe contenant les données et méthodes relatives à un module"""

    image: Path
    format: str  # The image format, as prepared by the dataset authors
    original_split: str  # The original dataset split where the image has been affected
    color_array: np.array  # The original 3-channel (BGR) array from the image file
    array: np.array  # The grayscale pixel values array (the temperature matrix)
    status: Optional[str] = None  # Defect label or healthy
    stats: dict = field(init=False, default_factory=dict)  # Statistical indicators
    histogram: np.array = field(init=False, default=None)  # Histogram (counts, edges)

    # Class variables
    # min_max is used to set the min & max pixel values on all modules
    min_max: ClassVar[tuple] = (0, 255)
        
    def plot(self, cmap: str = "inferno", display_colorbar: bool = True):
        """Affiche le thermogramme (la matrice des températures) du module"""

        plt.figure(figsize=(2,2))
        im = plt.imshow(self.array, cmap=cmap)
        if display_colorbar:
            values = np.arange(*self.min_max)
            ticks = [v for v in values if v % 50 == 0]  # Tick label interval: 50
            plt.colorbar(im, values=values, ticks=ticks, label="Pixel value (0-255)")    
        plt.title(self.format)
        plt.show()

    def plot_histogram(self, display_labels=True):
        """Affiche l'histogramme du module"""

        counts, edges = self.histogram
        
        plt.figure(figsize=(6,3))
        plt.stairs(counts, edges, fill=True)
        plt.tick_params(axis='both', which='major', labelsize=9)
        major_ticks = [t for t in edges if t % 10 == 0]
        plt.xticks(major_ticks, rotation=45, ha="right")
        plt.xticks(edges, minor=True)
        plt.title(f"{self.image.stem}\n{self.format}")
        if display_labels:
            plt.xlabel("Pixel values")
            plt.ylabel("Pixel count")
        plt.show()

    def extract_stats(self):
        """Extrait des indicateurs statistiques de la matrice de températures"""

        # Standard statistical indicators
        self.stats["mean"] = np.nanmean(self.array)
        self.stats["median"] = np.nanmedian(self.array)
        self.stats["max"] = np.nanmax(self.array)
        self.stats["std"] = np.nanstd(self.array)
        self.stats["min"] = np.nanmin(self.array)
        self.stats["ptp"] = self.stats["max"] - self.stats["min"]

        # Additional statistical indicators
        self.stats["skewness"] = skew(self.array, axis=None)
        self.stats["kurtosis"] = kurtosis(self.array, axis=None)
        self.stats["iqr_25_75"] = iqr(self.array)

        # Percentiles
        for p in np.arange(0.05, 1., 0.05):
            label = f"p_{p:.2f}"
            self.stats[label] = np.nanquantile(self.array, p)

        # Array size
        self.stats["size"] = np.sum(~np.isnan(self.array))

    def extract_histogram(self):
        """Extrait l'histogramme de la matrice de températures"""

        self.histogram = np.histogram(self.array, bins=np.arange(self.min_max[0], self.min_max[1] + 2))

    def __str__(self):
        """Permet d'afficher les infos principales de l'objet"""

        content = []
        content.append(f"Image: {self.image.stem}")
        content.append(f"Format: {self.format}")
        content.append(f"Split d'origine: {self.original_split}")
        content.append(f"Statut: {self.status}")
        content.append(f"Taille: {self.stats['size']}")
        content.append(f"Min/max: {self.stats['min']}/{self.stats['max']}")

        return "\n".join(content)
    
        