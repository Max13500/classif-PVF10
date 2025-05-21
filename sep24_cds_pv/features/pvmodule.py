"""
PVModule class definition
FeaturesExtractor class definition
"""
from typing import Optional, ClassVar, Self
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, iqr
from skimage.feature import graycomatrix, graycoprops
import cv2


@dataclass
class PVModule:
    """Classe contenant les données et méthodes relatives à un module"""

    image_path: Path
    format: str  # The image format, as prepared by the dataset authors
    original_split: str  # The original dataset split where the image has been affected
    color_array: np.array  # The original 3-channel (BGR) array from the image file
    array: np.array  # The grayscale pixel values array (the temperature matrix)
    status: Optional[str] = None  # Defect label or healthy
    stats: dict = field(init=False, default_factory=dict)  # Statistical indicators
    histogram: np.array = field(init=False, default=None)  # Histogram (counts, edges)
    histogram_dict: dict = field(init=False, default_factory=dict)  # Histogram dict {label: count}
    glcm_vector: dict = field(init=False, default_factory=dict)  # GLCM dict {glcm_feature: value}
    hot_spots: dict = field(init=False, default_factory=dict)  # Hot spots

    # Class variables
    # ---------------
    
    # min_max is used to set the min & max pixel values on all modules
    min_max: ClassVar[tuple] = (0, 255)

    # GLCM constants    
    GLCM_DISTANCES = [8]
    GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    # the "vault" is used to record all loaded modules, referenced by the path of the image file
    _vault: ClassVar[dict[str, Self]] = {}
        
    def __post_init__(self):
        """Enregistre l'objet créé dans le 'coffre-fort' :-)"""

        self._vault[self.image_path.as_posix()] = self

    @classmethod
    def get_module(cls, image_path: Path | str) -> Self:
        """Renvoie l'objet PVModule correspondant à l'image passée en paramètre"""

        image_path = Path(image_path)

        if (module := cls._vault.get(image_path.as_posix())) is not None:
            return module
        else:
            img = cv2.imread(image_path) # Récupération de l'image en couleur
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            status = str(image_path.parent.name)[2:]
            split = str(image_path.parents[1].name)
            format = str(image_path.parents[2].name).removeprefix("PVF_10_")

            module = cls(
                image_path=image_path, 
                format=format,
                original_split=split,
                color_array=img,
                array=img_gray, 
                status=status,
            )

            return module

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
        plt.title(f"{self.image_path.stem}\n{self.format}")
        if display_labels:
            plt.xlabel("Pixel values")
            plt.ylabel("Pixel count")
        plt.show()

    def extract_stats(self):
        """Extrait des indicateurs statistiques de la matrice de températures"""

        if not self.stats:

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

        if not self.histogram:
            self.histogram = np.histogram(self.array, bins=np.arange(self.min_max[0], self.min_max[1] + 2))
            self.histogram_dict = {f"hist_{bin_left_edge}": count 
                                   for count, bin_left_edge in zip(self.histogram[0], self.histogram[1][:-1])
                                  }

    @classmethod
    def get_glcm_feature_names(cls) -> list:
        """Renvoie la liste des features GLCM"""
        glcm_feature_names = []
        for prop in cls.GLCM_PROPERTIES:
            for distance in cls.GLCM_DISTANCES:
                for angle in cls.GLCM_ANGLES:
                    # On stocke dans les noms des features un label du type : contrast_d1_a45
                    glcm_feature_names.append(f"{prop}_d{distance}_a{np.degrees(angle):.0f}")
        return glcm_feature_names
    
    def extract_glcm(self):
        """Extrait le vecteur GLCM de la matrice de températures"""

        if not self.glcm_vector:

            glcm_feature_names = self.get_glcm_feature_names()

            glcm_vector = []
            # Calcul de la matrice GLCM (256 x 256 x distances x angles)
            glcm = graycomatrix(
                self.array,
                distances=self.GLCM_DISTANCES,
                angles=self.GLCM_ANGLES,
                levels=256
            )
            # Pour chaque propriété GLCM
            for prop in self.GLCM_PROPERTIES:
                # On la calcule pour les différentes distances et les différents angles 
                prop_matrix = graycoprops(glcm, prop)  # Matrice distances x angles
                # On transforme la matrice en vecteur qu'on stocke dans glcm_vector
                glcm_vector.extend(prop_matrix.flatten())

            self.glcm_vector = dict(zip(glcm_feature_names, glcm_vector))

    def extract_hotspots(self):
        """Extrait les hotspots de la matrice de températures"""

        if not self.hot_spots:

            hot_spots = {}

            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(self.array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological cleaning (optional but recommended)
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            hot_spots["cleaned"] = cleaned

            # Connected components
            num_labels, labels_im = cv2.connectedComponents(cleaned)
            hot_spots["labels_im"] = labels_im

            hot_spots["spots"] = []
            # Calculate area and mean value of each hot spot
            for label in range(1, num_labels):  # skip background label 0
                mask = (labels_im == label)
                area = np.sum(mask)
                mean = np.mean(self.array[mask])
                hot_spots["spots"].append({
                    "area": area,
                    "mean": mean,
                })

            self.hot_spots = hot_spots

    def plot_hot_spots(self):
        """Affiche les hot spots du module"""

        plt.figure(figsize=(5, 2))
        plt.subplot(1, 2, 1)
        plt.title("Cleaned Binary")
        plt.imshow(self.hot_spots["cleaned"], cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Labeled Hot Spots")
        plt.imshow(self.hot_spots["labels_im"], cmap='nipy_spectral')
        plt.show()

    def __str__(self):
        """Permet d'afficher les infos principales de l'objet"""

        content = []
        content.append(f"Image: {self.image_path.stem}")
        content.append(f"Format: {self.format}")
        content.append(f"Split d'origine: {self.original_split}")
        content.append(f"Statut: {self.status}")
        content.append(f"Taille: {self.stats['size']}")
        content.append(f"Min/max: {self.stats['min']}/{self.stats['max']}")

        return "\n".join(content)
    
        