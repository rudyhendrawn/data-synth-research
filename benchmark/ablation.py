import itertools
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AblationComponent:
    """Single component in ablation study."""

    name: str
    options: List[str]
    baseline: str
    description: str = ""


@dataclass
class AblationExperiment:
    """Single ablation experiment configuration."""

    exp_id: str
    components: Dict[str, str]
    description: str = ""

    def to_dict(self) -> Dict[str, str | Dict[str, str]]:
        return {
            "exp_id": self.exp_id,
            "components": self.components,
            "description": self.description,
        }


class AblationStudyManager:
    """
    Systematic ablation study manager for cross-domain fraud detection.
    """

    def __init__(
        self,
        output_dir: str = "results/ablation",
        components: Optional[Dict[str, AblationComponent]] = None,
    ) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if components is None:
            components = {
                "oversampling": AblationComponent(
                    name="oversampling",
                    options=[
                        "None",
                        "ROS",
                        "SMOTE",
                        "BorderlineSMOTE",
                        "SMOTEENN",
                        "PytorchGAN",
                        "CTGAN",
                        "ConditionalWGAN-GP",
                    ],
                    baseline="None",
                    description="Data balancing technique",
                ),
                "model": AblationComponent(
                    name="model",
                    options=[
                        "LogisticRegression",
                        "DecisionTree",
                        "RandomForest",
                        "XGBoost",
                        "MLP",
                    ],
                    baseline="XGBoost",
                    description="Base classifier architecture",
                ),
                "calibration": AblationComponent(
                    name="calibration",
                    options=["None", "Platt", "Isotonic"],
                    baseline="None",
                    description="Probability calibration method",
                ),
            }

        self.components = components
        self.experiments: List[AblationExperiment] = []

    def generate_single_factor_ablation(self, component_name: str) -> List[AblationExperiment]:
        """
        Generate single-factor ablation: vary one component, fix others at baseline.
        """
        if component_name not in self.components:
            raise ValueError(f"Unknown component: {component_name}")

        component = self.components[component_name]
        experiments: List[AblationExperiment] = []

        baseline_config = {name: comp.baseline for name, comp in self.components.items()}

        for option in component.options:
            config = baseline_config.copy()
            config[component_name] = option
            exp = AblationExperiment(
                exp_id=f"ablation_{component_name}_{option}",
                components=config,
                description=f"Ablate {component_name}: {option}",
            )
            experiments.append(exp)

        logger.info(
            "Generated %s single-factor ablation experiments for %s",
            len(experiments),
            component_name,
        )
        return experiments

    def generate_pairwise_ablation(self, component1: str, component2: str) -> List[AblationExperiment]:
        """
        Generate pairwise ablation: vary two components, fix others.
        """
        if component1 not in self.components or component2 not in self.components:
            raise ValueError("Unknown component")

        comp1 = self.components[component1]
        comp2 = self.components[component2]

        baseline_config = {name: comp.baseline for name, comp in self.components.items()}
        experiments: List[AblationExperiment] = []

        for opt1, opt2 in itertools.product(comp1.options, comp2.options):
            config = baseline_config.copy()
            config[component1] = opt1
            config[component2] = opt2
            exp = AblationExperiment(
                exp_id=f"ablation_{component1}_{opt1}_{component2}_{opt2}",
                components=config,
                description=f"Pairwise: {component1}={opt1}, {component2}={opt2}",
            )
            experiments.append(exp)

        logger.info("Generated %s pairwise ablation experiments", len(experiments))
        return experiments

    def generate_full_factorial(self, components: List[str]) -> List[AblationExperiment]:
        """
        Generate full factorial design for specified components.
        """
        if not all(c in self.components for c in components):
            raise ValueError("Unknown component in list")

        options_lists = [self.components[c].options for c in components]
        combinations = list(itertools.product(*options_lists))

        baseline_config = {
            name: comp.baseline for name, comp in self.components.items() if name not in components
        }

        experiments: List[AblationExperiment] = []
        for combo in combinations:
            config = baseline_config.copy()
            for comp_name, option in zip(components, combo):
                config[comp_name] = option

            exp_id = "_".join([f"{c}_{o}" for c, o in zip(components, combo)])
            exp = AblationExperiment(
                exp_id=f"factorial_{exp_id}",
                components=config,
                description=f"Factorial: {dict(zip(components, combo))}",
            )
            experiments.append(exp)

        logger.info("Generated %s factorial experiments", len(experiments))
        return experiments

    def save_experiments(self, experiments: List[AblationExperiment], filename: str) -> None:
        """Save experiment configurations to JSON."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump([exp.to_dict() for exp in experiments], f, indent=2)

        logger.info("Saved %s experiments to %s", len(experiments), filepath)
