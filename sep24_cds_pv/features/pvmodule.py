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
from skimage.filters.rank import entropy
from skimage.morphology import disk
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
    edges: np.array = field(init=False, default=None)  # Carte des contours
    edge_density: float = field(init=False, default=None)  # Edge density (Canny)
    entropy: np.array = field(init=False, default=None)  # Carte d'entropie
    entropy_vector: dict = field(init=False, default_factory=dict)  # Entropy dict {entropy_feature: value}
    hot_spots: dict = field(init=False, default_factory=dict)  # Hot spots
    hot_spots_features: dict = field(init=False, default_factory=dict)  # Hot spots features

    # Class variables
    # ---------------
    
    # min_max is used to set the min & max pixel values on all modules
    min_max: ClassVar[tuple] = (0, 255)

    # Statistical indicators constants
    STATS_PERCENTILES = np.arange(0.05, 1., 0.05)

    # GLCM constants    
    GLCM_DISTANCES = [8]
    GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    # Edge density - Canny thresholds
    CANNY_THRESHOLD1 = 70
    CANNY_THRESHOLD2 = 140

    # Entropy constants
    ENTROPY_RADIUS = 4
    ENTROPY_HIST_BINS = 20

    # the "vault" is used to record all loaded modules, referenced by the path of the image file
    _vault: ClassVar[dict[str, Self]] = {}
        
    def __post_init__(self):
        """Enregistre l'objet créé dans le 'coffre-fort' :-)"""

        self._vault[self.image_path.as_posix()] = self

    @classmethod
    def get_module(cls, image_path: Path | str) -> Self:
        """Renvoie l'objet PVModule correspondant à l'image passée en paramètre"""

        image_path = Path(image_path)

        module = cls._vault.get(image_path.as_posix())

        if module is None:
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

    @classmethod
    def get_known_modules(cls) -> list[Self]:
        """Renvoie la liste de tous les modules connus"""

        return cls._vault.values()
    
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

    @classmethod
    def get_stats_feature_names(cls) -> list:
        """Renvoie la liste des features 'indicateurs statistiques'"""
        
        stats_feature_names = [
            # Indicateurs classiques
            "mean", "median", "max", "std", "min", "ptp", 
            # Indicateurs additionnels
            "skewness", "kurtosis", "iqr_25_75",
        ]
        # Percentiles
        for p in cls.STATS_PERCENTILES:
            label = f"p_{p:.2f}"
            stats_feature_names.append(label)
        
        return stats_feature_names

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
            for p in self.STATS_PERCENTILES:
                label = f"p_{p:.2f}"
                self.stats[label] = np.nanquantile(self.array, p)

            # Array size
            self.stats["size"] = np.sum(~np.isnan(self.array))

    @classmethod
    def get_histogram_feature_names(cls) -> list:
        """Renvoie la liste des features liés à l'histogramme de la matrice de températures"""
        histogram_feature_names = [f"hist_{idx}" 
                                    for idx in range(cls.min_max[0],cls.min_max[1]+1)
                                    ]
        return histogram_feature_names

    def extract_histogram(self):
        """Extrait l'histogramme de la matrice de températures"""

        if self.histogram is None:
            self.histogram = np.histogram(self.array, bins=np.arange(self.min_max[0], self.min_max[1] + 2))
            self.histogram_dict = {f"hist_{bin_left_edge}": count 
                                   for count, bin_left_edge in zip(self.histogram[0], self.histogram[1][:-1])
                                  }

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

    @classmethod
    def get_edge_density_feature_names(cls) -> list:
        """Renvoie la liste des features liés à la densité de contours"""
        return ["edge_density"]
    
    def extract_edge_density(self):
        """Extrait la densité de contours à l'aide d'un filtre de Canny"""

        if self.edges is None:

            height, width = self.array.shape
            # On enlève les bords pour ne pas les détecter...
            cropped_array = self.array[2:height-1, 2:width-1]
            # Extraction de la densité de contours à l'aide d'un filtre de Canny
            self.edges = cv2.Canny(cropped_array, self.CANNY_THRESHOLD1, self.CANNY_THRESHOLD2)
            self.edge_density = np.sum(self.edges > 0) / self.edges.size     

    def plot_edges(self):
        """Affiche l'image des contours"""

        plt.figure(figsize=(2,2))
        im = plt.imshow(self.edges, cmap="gray")
        plt.title("Contours")
        plt.show()

    @classmethod
    def get_entropy_feature_names(cls) -> list:
        """Renvoie la liste des features liés à l'entropie"""
        entropy_feature_names = ['mean', 'std', 'min','max', 'median', 'skew', 'kurtosis']
        entropy_feature_names.extend([f'hist_bin{i+1}' for i in range(cls.ENTROPY_HIST_BINS)])
        return entropy_feature_names

    def extract_entropy(self):
        """Extrait les caractéristiques de l'entropie"""

        if self.entropy is None:

            entropy_feature_names = self.get_entropy_feature_names()

            # Calcul de la carte d'entropie
            self.entropy = entropy(self.array, disk(self.ENTROPY_RADIUS))
            # Calcul des statistiques associées
            entropy_vector = [
                    np.mean(self.entropy),
                    np.std(self.entropy),
                    np.min(self.entropy),
                    np.max(self.entropy),
                    np.median(self.entropy),
                    skew(self.entropy.ravel()),
                    kurtosis(self.entropy.ravel())
                ]
            hist, _ = np.histogram(self.entropy, bins=self.ENTROPY_HIST_BINS, range=(0, np.max(self.entropy)), density=True)
            # On l'ajoute au vecteur des caractéristiques de l'entropie
            entropy_vector.extend(list(hist))

            self.entropy_vector = dict(zip(entropy_feature_names, entropy_vector))

    def plot_entropy(self):
        """Affiche la carte d'entropie"""

        plt.figure(figsize=(2,2))
        im = plt.imshow(self.entropy, cmap="coolwarm")
        plt.title("Entropie")
        plt.show()

    @staticmethod
    def _get_morpho_features_from_mask(mask):
        """
        Extrait les features morphologiques (aire, circularité, excentricité)
        à partir d'un masque binaire d'un hot spot.
        
        Paramètres :
            mask (np.ndarray): image binaire (0 et 255)
            
        Retour :
            dict avec area, circularity, eccentricity
        """
        # S’assurer que le masque est bien binaire
        mask = (mask > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'contour_relative_area': 0, 'circularity': 0, 'eccentricity': 0}

        cnt = max(contours, key=cv2.contourArea)  # On suppose un seul hot spot principal

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Circularité : 4π × (aire) / (périmètre²)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # Excentricité via ellipse
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            _, axes, _ = ellipse
            major, minor = max(axes), min(axes)
            eccentricity = np.sqrt(1 - (minor / major) ** 2)
        else:
            eccentricity = 0

        return {
            'contour_relative_area': area / np.sum(~np.isnan(mask)),  # relative à la surface totale de l'image
            'contour_circularity': circularity,
            'contour_eccentricity': eccentricity
        }

    @staticmethod
    def get_hot_spots_feature_names() -> list:
        """Renvoie la liste des features liés aux hot spots"""
        hot_spots_feature_names = [
            "number",
            "mean_rel_areas",
            "std_rel_areas",
            "mean_means",
            "std_means",
            "mean_contour_rel_areas",
            "std_contour_rel_areas",
            "mean_contour_circ",
            "std_contour_circ",
            "mean_contour_ecc",
            "std_contour_ecc",
        ]
        return hot_spots_feature_names
    
    def extract_hotspots(self):
        """Extrait les hotspots de la matrice de températures"""

        if not self.hot_spots:

            hot_spots = {}

            # Applique un seuillage Otsu
            _, thresh = cv2.threshold(self.array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Nettoyage morphologique (optionnel mais recommandé)
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            hot_spots["cleaned"] = cleaned

            # Détection des pixesl connectés
            num_labels, labels_im = cv2.connectedComponents(cleaned)
            hot_spots["labels_im"] = labels_im

            hot_spots["spots"] = []
            # Calcul de la surface et de la valeur moyenne de chaque hot spot
            for label in range(1, num_labels):  # on skippe le fond (label 0)
                mask = (labels_im == label)
                area = np.sum(mask) / np.sum(~np.isnan(self.array))  # relative à la surface totale de l'image
                mean = np.mean(self.array[mask])
                hot_spot_desc = {
                    "relative_area": area,
                    "mean": mean,
                }
                # Add morphological features
                hot_spot_desc.update(self._get_morpho_features_from_mask(mask))
                
                hot_spots["spots"].append(hot_spot_desc)

            self.hot_spots = hot_spots

            # Extraction des features synthétiques
            number = len(hot_spots["spots"])

            relative_areas = [hs_desc["relative_area"] for hs_desc in hot_spots["spots"]]
            mean_relative_areas = np.mean(relative_areas) if hot_spots["spots"] else 0.
            std_relative_areas = np.std(relative_areas) if hot_spots["spots"] else 0.
            
            means = [hs_desc["mean"] for hs_desc in hot_spots["spots"]]
            mean_means = np.mean(means) if hot_spots["spots"] else 0.
            std_means = np.std(means) if hot_spots["spots"] else 0.
            
            contour_relative_areas = [hs_desc["contour_relative_area"] for hs_desc in hot_spots["spots"]]
            mean_contour_relative_areas = np.mean(contour_relative_areas) if hot_spots["spots"] else 0.
            std_contour_relative_areas = np.std(contour_relative_areas) if hot_spots["spots"] else 0.
            
            contour_circularities = [hs_desc["contour_circularity"] for hs_desc in hot_spots["spots"]]
            mean_contour_circularities = np.mean(contour_circularities) if hot_spots["spots"] else 0.
            std_contour_circularities = np.std(contour_circularities) if hot_spots["spots"] else 0.
            
            contour_eccentricities = [hs_desc["contour_eccentricity"] for hs_desc in hot_spots["spots"]]
            mean_contour_eccentricities = np.mean(contour_eccentricities) if hot_spots["spots"] else 0.
            std_contour_eccentricities = np.std(contour_eccentricities) if hot_spots["spots"] else 0.

            self.hot_spots_features = {
                "number": number,
                "mean_rel_areas": mean_relative_areas,
                "std_rel_areas": std_relative_areas,
                "mean_means": mean_means,
                "std_means": std_means,
                "mean_contour_rel_areas": mean_contour_relative_areas,
                "std_contour_rel_areas": std_contour_relative_areas,
                "mean_contour_circ": mean_contour_circularities,
                "std_contour_circ": std_contour_circularities,
                "mean_contour_ecc": mean_contour_eccentricities,
                "std_contour_ecc": std_contour_eccentricities,
            }

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
        content.append(f"Taille: {self.array.shape}")

        return "\n".join(content)
    
        